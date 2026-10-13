from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import pickle
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from openmatch.modeling.modeling_colqwen.modeling_colqwen_model import ColQwen2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--embedding-dir", default="retrieval/encode/colqwen_fa_2048"
    )
    args = parser.parse_args()

    if args.topk < 1:
        raise ValueError("topk must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    embedding_dir = ROOT / args.embedding_dir
    files = sorted(
        p for p in embedding_dir.glob("embeddings.corpus.rank.*")
        if p.is_file() and not p.name.endswith(".tmp")
    )
    if not files:
        raise FileNotFoundError("No Persian corpus embeddings found.")

    embeddings, ids = [], []
    for path in files:
        with path.open("rb") as f:
            vectors, page_ids = pickle.load(f)
        if len(vectors) != len(page_ids):
            raise RuntimeError(f"Embedding/ID mismatch: {path}")
        embeddings.extend(vectors)
        ids.extend(page_ids)

    if len(ids) != 122 or len(set(ids)) != 122:
        raise RuntimeError("Expected 122 unique page IDs.")

    metadata = {}
    with (ROOT / "data/fa/pages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata[record["corpus-id"]] = record

    for page_id in ids:
        if page_id not in metadata:
            raise RuntimeError(f"Missing metadata: {page_id}")
        if not (ROOT / metadata[page_id]["image_path"]).is_file():
            raise FileNotFoundError(metadata[page_id]["image_path"])

    print(f"Loaded embeddings for {len(ids)} pages.", flush=True)
    print("Loading ColQwen2...", flush=True)
    model = ColQwen2Model(
        str(ROOT / "retrieval/colqwen2-v1.0-merged")
    )

    output_dir = ROOT / "retrieval/results/fa"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nEnter a Persian question. Type exit to finish.")

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

        with torch.inference_mode():
            inputs = model.processor.process_queries([query]).to("cuda:0")
            query_embedding = model.model(**inputs)
            print("DEBUG question:", repr(query))
            print("DEBUG token IDs:", inputs["input_ids"][0].tolist())
            print("DEBUG embedding shape:", tuple(query_embedding.shape))
            print(
                "DEBUG embedding sample:",
                query_embedding[0, 0, :8].detach().float().cpu().tolist(),
            )
            scores = model.processor.score(
                query_embedding,
                embeddings,
                device="cuda",
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        query_scores = scores.detach().cpu()[0]
        with torch.inference_mode():
            q = query_embedding[0].detach().float().to("cuda:0")
            manual_scores = []

            for page_embedding in embeddings:
                p = page_embedding.float().to("cuda:0")
                score = (q @ p.T).max(dim=1).values.sum()
                manual_scores.append(score.item())

            manual_scores = torch.tensor(manual_scores)
            manual_top = torch.topk(
                manual_scores, k=min(args.topk, len(ids))
            ).indices.tolist()

        print("\nIndependent MaxSim results:")
        for rank, index in enumerate(manual_top, start=1):
            print(rank, ids[index], float(manual_scores[index]))

        print(
            "Maximum score difference:",
            (manual_scores - query_scores.float()).abs().max().item(),
        )
        del q, p
        if query_scores.numel() != len(ids):
            raise RuntimeError("Score count does not match page count.")
        if not torch.isfinite(query_scores).all().item():
            raise RuntimeError("Scores contain NaN or Inf.")

        top_scores, indices = torch.topk(
            query_scores, k=min(args.topk, len(ids))
        )

        results = []
        for rank, (index, score) in enumerate(
            zip(indices.tolist(), top_scores.tolist()), start=1
        ):
            page_id = ids[index]
            record = metadata[page_id]
            image_path = ROOT / record["image_path"]

            result = {
                "rank": rank,
                "corpus_id": page_id,
                "score": score,
                "source_pdf": record["source_pdf"],
                "pdf_page_number": record["pdf_page_number"],
                "image_path": record["image_path"],
            }
            results.append(result)

            print(f"\n{rank}. {Path(record['source_pdf']).name}")
            print(f"   PDF page: {record['pdf_page_number']}")
            print(f"   Score: {score:.4f}")
            print(f"   Image: {image_path}")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result_path = output_dir / f"query_{stamp}.json"
        with result_path.open("x", encoding="utf-8") as f:
            json.dump({
                "query": query,
                "elapsed_seconds": elapsed,
                "corpus_pages": len(ids),
                "results": results,
            }, f, ensure_ascii=False, indent=2)

        print(f"\nQuery encoding + search: {elapsed:.2f} seconds")
        print(f"Saved results: {result_path}")

        del inputs, query_embedding, scores, query_scores

    print("\nFinished.")


if __name__ == "__main__":
    main()
