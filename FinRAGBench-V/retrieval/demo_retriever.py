import argparse
import glob
import os
import pickle

import torch

from openmatch.modeling.modeling_colqwen.modeling_colqwen_model import ColQwen2Model


def load_corpus_embeddings(embedding_dir):
    """
    Load the pre-computed ColQwen2 corpus embeddings.
    """

    corpus_files = sorted(
        glob.glob(
            os.path.join(
                embedding_dir,
                "embeddings.corpus.rank.*"
            )
        )
    )

    if not corpus_files:
        raise FileNotFoundError(
            f"No corpus embeddings found in: {embedding_dir}"
        )

    all_embeddings = []
    all_ids = []

    print(f"Found {len(corpus_files)} corpus embedding file(s).")

    for file_path in corpus_files:
        print(f"Loading: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        embeddings = data[0]
        corpus_ids = data[1]

        if len(corpus_ids) == 0:
            continue

        all_embeddings.extend(embeddings)
        all_ids.extend(corpus_ids)

    print(f"Loaded {len(all_ids)} corpus documents.")

    return all_embeddings, all_ids


def encode_query(model, query):
    """
    Convert a user query into ColQwen2 multi-vector embeddings.
    """

    batch_query = model.processor.process_queries(
        [query]
    )

    batch_query = {
        key: value.to("cuda")
        if torch.is_tensor(value)
        else value
        for key, value in batch_query.items()
    }

    with torch.no_grad():
        query_embeddings = model.model(
            **batch_query
        )

    return query_embeddings


def retrieve(
    model,
    query_embeddings,
    corpus_embeddings,
    corpus_ids,
    topk=5,
):
    """
    Compare query embeddings against all corpus embeddings
    using ColQwen2 MaxSim / late interaction.
    """

    with torch.no_grad():

        scores = model.processor.score(
            query_embeddings,
            corpus_embeddings,
            device="cuda"
        )

        scores = scores.cpu()

    # We have only one query
    query_scores = scores[0]

    topk = min(topk, len(corpus_ids))

    top_scores, top_indices = torch.topk(
        query_scores,
        k=topk,
        largest=True,
        sorted=True,
    )

    results = []

    for rank, (idx, score) in enumerate(
        zip(top_indices, top_scores),
        start=1,
    ):
        corpus_id = corpus_ids[idx.item()]

        results.append(
            {
                "rank": rank,
                "corpus_id": corpus_id,
                "score": score.item(),
            }
        )

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Interactive ColQwen2 Retriever Demo"
    )

    parser.add_argument(
        "--lang",
        choices=["en", "ch"],
        default="en",
        help="Corpus language: en or ch",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of retrieved pages",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text",
    )

    args = parser.parse_args()

    # ---------------------------
    # Paths
    # ---------------------------

    model_path = "./colqwen2-v1.0-merged"

    embedding_dir = (
        f"./encode/colqwen_{args.lang}"
    )

    corpus_image_dir = (
        f"../../mini_corpus/{args.lang}"
    )

    # ---------------------------
    # Query
    # ---------------------------

    if args.query is None:
        print()
        query = input("Enter query: ").strip()
    else:
        query = args.query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    print("\n==============================")
    print("Query:")
    print(query)
    print("==============================\n")

    # ---------------------------
    # Load model
    # ---------------------------

    print("Loading ColQwen2...")

    model = ColQwen2Model(
        model_path
    )

    print("Model loaded.\n")

    # ---------------------------
    # Load corpus embeddings
    # ---------------------------

    corpus_embeddings, corpus_ids = (
        load_corpus_embeddings(
            embedding_dir
        )
    )

    # ---------------------------
    # Encode query
    # ---------------------------

    print("\nEncoding query...")

    query_embeddings = encode_query(
        model,
        query
    )

    print("Query embedding created.")

    # ---------------------------
    # Retrieve
    # ---------------------------

    print("\nSearching corpus...\n")

    results = retrieve(
        model=model,
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        corpus_ids=corpus_ids,
        topk=args.topk,
    )

    # ---------------------------
    # Print results
    # ---------------------------

    print("\n========== TOP RESULTS ==========\n")

    for result in results:

        corpus_id = result["corpus_id"]

        filename = str(corpus_id)

        if not filename.lower().endswith(".png"):
            filename += ".png"

        image_path = os.path.join(
            corpus_image_dir,
            filename
        )

        print(
            f'{result["rank"]}. '
            f'{result["corpus_id"]}'
        )

        print(
            f'   Score: {result["score"]:.4f}'
        )

        print(
            f'   Image: {image_path}'
        )

        print()

    print("=================================\n")


if __name__ == "__main__":
    main()
