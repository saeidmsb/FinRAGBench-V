from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickle
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from colpali_engine.models import ColQwen2, ColQwen2Processor
from retrieve_fa_bm25 import tokenize, TOKENIZER_VERSION

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=20)
    args = parser.parse_args()
    if args.topk < 1 or args.candidates < 1:
        raise ValueError("Counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    embedding_path = ROOT / (
        "retrieval/encode/colpali_fa_test/"
        "embeddings.corpus.rank.0.0-122"
    )
    with embedding_path.open("rb") as f:
        vectors, ids = pickle.load(f)
    if len(vectors) != len(ids) or len(set(ids)) != len(ids):
        raise RuntimeError("Invalid corpus embeddings")

    with (ROOT / "retrieval/encode/bm25_fa/index.pkl").open("rb") as f:
        bm25 = pickle.load(f)

    text_path = ROOT / "data/fa/text/pages_text.jsonl"
    content = text_path.read_bytes()
    if bm25["source_sha256"] != hashlib.sha256(content).hexdigest():
        raise RuntimeError("BM25 index is outdated")
    if bm25["tokenizer_version"] != TOKENIZER_VERSION:
        raise RuntimeError("BM25 tokenizer changed")

    pages = {
        r["corpus-id"]: r
        for r in (
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        )
    }
    for page_id in ids:
        if page_id not in pages:
            raise RuntimeError(f"Missing page metadata: {page_id}")

    model_path = ROOT / "retrieval/colqwen2-v1.0-merged"
    print("Loading ColQwen2...", flush=True)
    visual = ColQwen2.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(str(model_path))

    manifest = json.loads(
        (ROOT / "models/reranker_manifest.json").read_text(encoding="utf-8")
    )
    options = {
        "cache_dir": str(ROOT / "models/hf-cache"),
        "local_files_only": True,
    }
    if manifest.get("revision"):
        options["revision"] = manifest["revision"]

    print("Loading reranker...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_id"], **options
    )
    reranker = AutoModelForSequenceClassification.from_pretrained(
        manifest["model_id"], torch_dtype=torch.float16, **options
    ).to("cuda:0").eval()

    text_records = bm25["records"]
    token_sets = [
        set(tokenize(r["text_normalized"])) for r in text_records
    ]
    out_dir = ROOT / "retrieval/results/fa_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nReady. Enter a question; type exit to finish.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        torch.cuda.synchronize()
        start = time.perf_counter()

        inputs = processor.process_queries([query])
        decoded = processor.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        if not decoded.lstrip().startswith("Query:"):
            inputs = processor.process_queries(["Query: " + query])
        inputs = inputs.to("cuda:0")

        with torch.inference_mode():
            query_vector = visual(**inputs)
            visual_scores = processor.score_multi_vector(
                query_vector, vectors, device="cuda:0"
            )[0].detach().float().cpu()

        if not torch.isfinite(visual_scores).all().item():
            raise RuntimeError("Invalid visual scores")
        visual_order = sorted(
            range(len(ids)),
            key=lambda i: (-float(visual_scores[i]), ids[i]),
        )[:args.candidates]
        visual_ranks = {
            ids[i]: rank
            for rank, i in enumerate(visual_order, start=1)
        }

        query_words = set(tokenize(query))
        text_scores = bm25["model"].get_scores(tokenize(query))
        matching = [
            i for i, words in enumerate(token_sets) if words & query_words
        ]
        text_order = sorted(
            matching,
            key=lambda i: (
                -float(text_scores[i]), text_records[i]["corpus-id"]
            ),
        )[:args.candidates]
        text_ranks = {
            text_records[i]["corpus-id"]: rank
            for rank, i in enumerate(text_order, start=1)
        }

        candidate_ids = sorted(set(visual_ranks) | set(text_ranks))
        usable = [
            p for p in candidate_ids if pages[p]["text_normalized"].strip()
        ]
        textless = sorted(set(candidate_ids) - set(usable))

        rerank_scores = {}
        truncated = []
        for p in usable:
            length = len(tokenizer(
                query, pages[p]["text_normalized"], truncation=False
            )["input_ids"])
            if length > 512:
                truncated.append(p)

        for offset in range(0, len(usable), 4):
            batch_ids = usable[offset:offset + 4]
            batch = tokenizer(
                [[query, pages[p]["text_normalized"]] for p in batch_ids],
                padding=True,
                truncation="only_second",
                max_length=512,
                return_tensors="pt",
            ).to("cuda:0")
            with torch.inference_mode():
                scores = reranker(**batch).logits.float().cpu().reshape(-1)
            if len(scores) != len(batch_ids) or not torch.isfinite(scores).all():
                raise RuntimeError("Invalid reranker scores")
            rerank_scores.update({
                p: float(score) for p, score in zip(batch_ids, scores)
            })
            del batch, scores

        ranked = sorted(
            usable, key=lambda p: (-rerank_scores[p], p)
        ) + textless

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results = []
        for rank, page_id in enumerate(ranked, start=1):
            page = pages[page_id]
            results.append({
                "rank": rank,
                "page_id": page_id,
                "reranker_score": rerank_scores.get(page_id),
                "reranker_score_percent": (
                    100.0 * torch.sigmoid(
                        torch.tensor(rerank_scores[page_id], dtype=torch.float64)
                    ).item()
                    if page_id in rerank_scores else None
                ),
                "visual_rank": visual_ranks.get(page_id),
                "bm25_rank": text_ranks.get(page_id),
                "source_pdf": page["source_pdf"],
                "pdf_page_number": page["pdf_page_number"],
                "image_path": page["image_path"],
            })

        for result in results[:args.topk]:
            print(
                f'\n{result["rank"]}. '
                f'{Path(result["source_pdf"]).name} | '
                f'PDF page: {result["pdf_page_number"]}'
            )
            percent = result["reranker_score_percent"]
            print(
                f"   Normalized score: {percent:.2f}%"
                if percent is not None else "   Normalized score: N/A"
            )
            print("   Image:", ROOT / result["image_path"])

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = out_dir / f"query_{stamp}.json"
        output.write_text(json.dumps({
            "query": query,
            "method": "ColQwen2 + BM25 + text reranker",
            "embedding_file": str(embedding_path.relative_to(ROOT)),
            "reranker_model": manifest["model_id"],
            "candidate_count": len(candidate_ids),
            "textless_pages": textless,
            "truncated_pages": truncated,
            "elapsed_seconds": elapsed,
            "ranking": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\nCandidates: {len(candidate_ids)} | Time: {elapsed:.2f}s")
        print("Saved:", output)
        if textless:
            print("Note: textless pages placed after text-scored pages.")
        del inputs, query_vector, visual_scores

    print("\nFinished.")


if __name__ == "__main__":
    main()
