from pathlib import Path
from datetime import datetime, timezone
import json
import pickle

from retrieve_fa_bm25 import tokenize, TOKENIZER_VERSION
import hashlib

ROOT = Path(__file__).resolve().parents[1]
TOPK = 20

def main():
    reports = sorted(
        (ROOT / "retrieval/results/fa_evaluation").glob("evaluation_*.json")
    )
    if not reports:
        raise RuntimeError("No ColQwen2 evaluation report found")

    visual_path = reports[-1]
    visual_report = json.loads(visual_path.read_text(encoding="utf-8"))

    dataset_path = ROOT / visual_report["dataset"]
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != \
            visual_report["dataset_sha256"]:
        raise RuntimeError("Evaluation dataset changed; rerun visual evaluation")

    with (ROOT / "retrieval/encode/bm25_fa/index.pkl").open("rb") as f:
        text_index = pickle.load(f)

    text_path = ROOT / "data/fa/text/pages_text.jsonl"
    if text_index["source_sha256"] != \
            hashlib.sha256(text_path.read_bytes()).hexdigest():
        raise RuntimeError("Text index is outdated; rebuild BM25 index")
    if text_index["tokenizer_version"] != TOKENIZER_VERSION:
        raise RuntimeError("Tokenizer changed; rebuild BM25 index")

    records = text_index["records"]
    rows = []

    for item in visual_report["results"]:
        query = item["query"]
        relevant = set(item["relevant_page_ids"])
        tokens = tokenize(query)

        visual_top = [
            r["page_id"] for r in item["ranking"][:TOPK]
        ]

        scores = text_index["model"].get_scores(tokens)
        # Keep pages that share at least one query token.
        query_words = set(tokens)
        matched = [
            i for i, record in enumerate(records)
            if query_words & set(tokenize(record["text_normalized"]))
        ]
        text_order = sorted(
            matched,
            key=lambda i: (-float(scores[i]), records[i]["corpus-id"]),
        )
        text_top = [
            records[i]["corpus-id"] for i in text_order[:TOPK]
        ]

        # Membership set only: this is not a fused ranking.
        candidate_ids = sorted(set(visual_top) | set(text_top))

        def recall(pages):
            return len(relevant & set(pages)) / len(relevant)

        candidates = [
            {
                "page_id": page_id,
                "visual_rank": (
                    visual_top.index(page_id) + 1
                    if page_id in visual_top else None
                ),
                "bm25_rank": (
                    text_top.index(page_id) + 1
                    if page_id in text_top else None
                ),
            }
            for page_id in candidate_ids
        ]

        row = {
            "query_id": item["query_id"],
            "query": query,
            "relevant_page_ids": sorted(relevant),
            "visual_recall_at_20": recall(visual_top),
            "bm25_recall_at_20": recall(text_top),
            "union_candidate_recall": recall(candidate_ids),
            "candidate_count": len(candidate_ids),
            "candidates": candidates,
            "missing_reference_pages": sorted(relevant - set(candidate_ids)),
        }
        rows.append(row)

        print(
            f'{row["query_id"]}: '
            f'ColQwen2={row["visual_recall_at_20"]:.2f}, '
            f'BM25={row["bm25_recall_at_20"]:.2f}, '
            f'Union={row["union_candidate_recall"]:.2f}, '
            f'Candidates={row["candidate_count"]}',
            flush=True,
        )

    count = len(rows)
    summary = {
        "questions": count,
        "topk_per_retriever": TOPK,
        "mean_visual_recall_at_20":
            sum(r["visual_recall_at_20"] for r in rows) / count,
        "mean_bm25_recall_at_20":
            sum(r["bm25_recall_at_20"] for r in rows) / count,
        "mean_union_candidate_recall":
            sum(r["union_candidate_recall"] for r in rows) / count,
        "mean_candidate_count":
            sum(r["candidate_count"] for r in rows) / count,
    }

    out_dir = ROOT / "retrieval/results/fa_evaluation"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = out_dir / f"candidates_{stamp}.json"
    output.write_text(json.dumps({
        "visual_report": str(visual_path.relative_to(ROOT)),
        "summary": summary,
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)

if __name__ == "__main__":
    main()
