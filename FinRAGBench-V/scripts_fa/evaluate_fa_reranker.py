from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 4
MAX_LENGTH = 512

def main():
    paths = sorted(
        (ROOT / "retrieval/results/fa_evaluation").glob("candidates_*.json")
    )
    if not paths:
        raise RuntimeError("No candidate report found")
    candidate_path = paths[-1]
    candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))

    visual_path = ROOT / candidate_report["visual_report"]
    visual_report = json.loads(visual_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / visual_report["dataset"]
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != \
            visual_report["dataset_sha256"]:
        raise RuntimeError("Evaluation dataset has changed")

    text_records = [
        json.loads(line)
        for line in (ROOT / "data/fa/text/pages_text.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = {r["corpus-id"]: r["text_normalized"] for r in text_records}

    manifest = json.loads(
        (ROOT / "models/reranker_manifest.json").read_text(encoding="utf-8")
    )
    load_options = {
        "cache_dir": str(ROOT / "models/hf-cache"),
        "local_files_only": True,
    }
    if manifest.get("revision"):
        load_options["revision"] = manifest["revision"]

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_id"], **load_options
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        manifest["model_id"],
        torch_dtype=torch.float16,
        **load_options,
    ).to("cuda:0").eval()

    rows = []
    for item in candidate_report["results"]:
        candidates = item["candidates"]
        scores = {}
        no_text = []

        usable = []
        for candidate in candidates:
            page_id = candidate["page_id"]
            if page_id not in texts:
                raise RuntimeError(f"Missing page metadata: {page_id}")
            if texts[page_id].strip():
                usable.append(candidate)
            else:
                no_text.append(page_id)

        truncated_pages = []
        for candidate in usable:
            page_id = candidate["page_id"]
            length = len(tokenizer(
                item["query"], texts[page_id], truncation=False
            )["input_ids"])
            if length > MAX_LENGTH:
                truncated_pages.append(page_id)

        for start in range(0, len(usable), BATCH_SIZE):
            batch = usable[start:start + BATCH_SIZE]
            pairs = [
                [item["query"], texts[c["page_id"]]] for c in batch
            ]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation="only_second",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to("cuda:0")

            with torch.inference_mode():
                logits = model(**inputs).logits.float().cpu().reshape(-1)

            if len(logits) != len(batch) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid reranker scores")

            scores.update({
                c["page_id"]: float(score)
                for c, score in zip(batch, logits)
            })
            del inputs, logits

        # Textless pages cannot receive a text-reranker score.
        # Retain them after scored pages and record this limitation.
        ranked = sorted(scores, key=lambda p: (-scores[p], p)) + sorted(no_text)
        relevant = set(item["relevant_page_ids"])
        positions = {p: i for i, p in enumerate(ranked, start=1)}
        reference_ranks = {
            p: positions.get(p) for p in sorted(relevant)
        }
        found = [rank for rank in reference_ranks.values() if rank is not None]
        first_rank = min(found) if found else None

        row = {
            "query_id": item["query_id"],
            "query": item["query"],
            "reference_ranks": reference_ranks,
            "recall_at_5": len(relevant & set(ranked[:5])) / len(relevant),
            "recall_at_20": len(relevant & set(ranked[:20])) / len(relevant),
            "reciprocal_rank": 1 / first_rank if first_rank else 0,
            "textless_pages": no_text,
            "truncated_pages": truncated_pages,
            "ranking": [
                {"rank": rank, "page_id": p, "score": scores.get(p)}
                for rank, p in enumerate(ranked, start=1)
            ],
            "baselines": {
                "colqwen2_reference_ranks": {
                    p: next(
                        (r["rank"] for r in next(
                            v for v in visual_report["results"]
                            if v["query_id"] == item["query_id"]
                        )["ranking"] if r["page_id"] == p), None
                    ) for p in sorted(relevant)
                },
                "bm25_reference_ranks_in_top20": {
                    p: next(
                        (c["bm25_rank"] for c in candidates
                         if c["page_id"] == p), None
                    ) for p in sorted(relevant)
                },
            },
        }
        rows.append(row)

        print(f'\n{item["query_id"]}: reference ranks = {reference_ranks}')
        print("Baselines:", row["baselines"])
        print("Truncated pages:", len(truncated_pages))
        for result in row["ranking"][:5]:
            print(result["rank"], result["page_id"], result["score"])

    summary = {
        "questions": len(rows),
        "mean_recall_at_5": sum(r["recall_at_5"] for r in rows) / len(rows),
        "mean_recall_at_20": sum(r["recall_at_20"] for r in rows) / len(rows),
        "mrr": sum(r["reciprocal_rank"] for r in rows) / len(rows),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / f"retrieval/results/fa_evaluation/reranker_{stamp}.json"
    output.write_text(json.dumps({
        "candidate_report": str(candidate_path.relative_to(ROOT)),
        "model": manifest,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "summary": summary,
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)

if __name__ == "__main__":
    main()
