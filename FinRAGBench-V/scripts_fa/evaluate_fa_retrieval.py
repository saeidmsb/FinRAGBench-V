from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import sys
import json
import pickle
import hashlib
import argparse
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from openmatch.modeling.modeling_colqwen.modeling_colqwen_model import ColQwen2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-dir", default="retrieval/encode/colqwen_fa_2048"
    )
    parser.add_argument(
        "--questions", default="data/fa/evaluation/questions_10.json"
    )
    parser.add_argument("--score-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.score_batch_size < 1:
        raise ValueError("score-batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dataset_path = ROOT / args.questions
    dataset_bytes = dataset_path.read_bytes()
    questions = json.loads(
        dataset_bytes.decode("utf-8")
    )["questions"]
    if not questions:
        raise RuntimeError("Evaluation dataset is empty")
    if len({q["query_id"] for q in questions}) != len(questions):
        raise RuntimeError("Duplicate query IDs")

    embedding_dir = ROOT / args.embedding_dir
    files = sorted(
        p for p in embedding_dir.glob("embeddings.corpus.rank.*")
        if p.is_file() and not p.name.endswith(".tmp")
    )
    if not files:
        raise RuntimeError("No corpus embeddings found")

    embeddings, ids = [], []
    for path in files:
        with path.open("rb") as f:
            vectors, page_ids = pickle.load(f)
        if len(vectors) != len(page_ids):
            raise RuntimeError(f"Embedding/ID mismatch: {path}")
        embeddings.extend(vectors)
        ids.extend(page_ids)

    if len(ids) != 122 or len(set(ids)) != len(ids):
        raise RuntimeError("Expected 122 unique corpus pages")

    corpus_ids = set(ids)
    for question in questions:
        full = question.get("relevant_page_ids")
        partial = question.setdefault("per_relevant_page_ids", [])
        if not isinstance(full, list) or not full:
            raise ValueError("Full references must be a nonempty list")
        if not isinstance(partial, list):
            raise ValueError("Partial references must be a list")
        if not all(isinstance(p, str) and p for p in full + partial):
            raise ValueError("Invalid reference page IDs")
        if len(set(full)) != len(full) or len(set(partial)) != len(partial):
            raise ValueError("Duplicate reference pages")
        if set(full) & set(partial):
            raise ValueError("Full and partial references must be disjoint")
        missing = (set(full) | set(partial)) - corpus_ids
        if missing:
            raise ValueError(f"Reference pages missing: {missing}")
    for question in questions:
        relevant = question["relevant_page_ids"]
        if not question["query"].strip() or not relevant:
            raise RuntimeError("Empty query or reference list")
        missing = set(relevant) - corpus_ids
        if missing:
            raise RuntimeError(f"Reference pages missing: {missing}")

    model_path = ROOT / "retrieval/colqwen2-v1.0-merged"
    print("Loading ColQwen2...", flush=True)
    model = ColQwen2Model(str(model_path))

    rows = []
    for question in questions:
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            inputs = model.processor.process_queries(
                [question["query"]]
            ).to("cuda:0")
            query_embedding = model.model(**inputs)
            scores = model.processor.score(
                query_embedding, embeddings, device="cuda", batch_size=args.score_batch_size
            )[0].detach().float().cpu()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if scores.numel() != len(ids) or not torch.isfinite(scores).all():
            raise RuntimeError("Invalid retrieval scores")

        # Fixed ID-based tie-breaking makes equal-score ranks reproducible.
        order = sorted(
            range(len(ids)),
            key=lambda index: (-float(scores[index]), ids[index]),
        )
        ranked_ids = [ids[index] for index in order]
        positions = {
            page_id: rank
            for rank, page_id in enumerate(ranked_ids, start=1)
        }

        relevant = set(question["relevant_page_ids"])
        reference_ranks = {
            page_id: positions[page_id]
            for page_id in sorted(relevant)
        }
        first_rank = min(reference_ranks.values())
        accepted = relevant | set(question["per_relevant_page_ids"])
        first_accepted_rank = min(positions[p] for p in accepted)
        top10 = set(ranked_ids[:10])

        row = {
            **question,
            "reference_ranks": reference_ranks,
            "first_relevant_rank": first_rank,
            "recall_at_10": len(relevant & top10) / len(relevant),
            "recall_at_10_with_partial": len(accepted & top10) / len(accepted),
            "first_accepted_rank": first_accepted_rank,
            "retrieved_page_ids": ranked_ids[:10],
            "reciprocal_rank_at_10_with_partial": (
                1.0 / first_accepted_rank if first_accepted_rank <= 10 else 0.0
            ),
            "reciprocal_rank_at_10": 1.0 / first_rank if first_rank <= 10 else 0.0,
            "elapsed_seconds": elapsed,
            "query_token_count": int(inputs["input_ids"].shape[1]),
            "ranking": [
                {
                    "rank": rank,
                    "page_id": ids[index],
                    "score": float(scores[index]),
                }
                for rank, index in enumerate(order[:10], start=1)
            ],
        }
        rows.append(row)

        print(
            f'{question["query_id"]}: '
            f'rank={first_rank}, '
            f'Recall@10={row["recall_at_10"]:.2f}, '
            f'RR@10={row["reciprocal_rank_at_10"]:.3f}, '
            f'Recall@10+partial={row["recall_at_10_with_partial"]:.2f}',
            flush=True,
        )
        del inputs, query_embedding, scores

    summary = {
        "questions": len(rows),
        "unique_reference_pages": len({
            page_id
            for question in questions
            for page_id in question["relevant_page_ids"]
        }),
        "k": 10,
        "mean_recall_at_10": sum(r["recall_at_10"] for r in rows) / len(rows),
        "mrr_at_10": sum(r["reciprocal_rank_at_10"] for r in rows) / len(rows),
        "mean_recall_at_10_with_partial": sum(
            r["recall_at_10_with_partial"] for r in rows
        ) / len(rows),
        "mrr_at_10_with_partial": sum(
            r["reciprocal_rank_at_10_with_partial"] for r in rows
        ) / len(rows),
    }

    output_dir = ROOT / "retrieval/results/fa_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"evaluation_{Path(args.embedding_dir).name}_{stamp}.json"

    report = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model_path": str(model_path.relative_to(ROOT)),
        "embedding_files": [str(p.relative_to(ROOT)) for p in files],
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "tie_breaking": "score descending, page ID ascending",
        "summary": summary,
        "results": rows,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("Saved report:", output_path)


if __name__ == "__main__":
    main()
