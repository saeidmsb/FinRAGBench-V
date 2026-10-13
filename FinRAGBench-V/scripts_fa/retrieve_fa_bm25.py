from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import pickle
import re
import unicodedata

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/fa/text/pages_text.jsonl"
INDEX = ROOT / "retrieval/encode/bm25_fa/index.pkl"
RESULTS = ROOT / "retrieval/results/fa_bm25"
TOKENIZER_VERSION = "fa_v1"

def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(str.maketrans({
        "ي": "ی", "ك": "ک",
        **dict(zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")),
        **dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789")),
    }))
    text = text.replace("\u200c", " ").replace("\u200d", "")
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn" and c != "\u0640"
    )
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    if args.topk < 1:
        raise ValueError("topk must be positive")

    content = SOURCE.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    state = None

    if INDEX.exists():
        with INDEX.open("rb") as f:
            cached = pickle.load(f)
        if (
            cached["source_sha256"] == fingerprint
            and cached["tokenizer_version"] == TOKENIZER_VERSION
        ):
            state = cached
            print("Loaded saved BM25 index.")

    if state is None:
        records = [
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
        ids = [r["corpus-id"] for r in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate page IDs")

        # Pages without searchable text remain in the source dataset,
        # but cannot participate in this text-only retrieval.
        records = [
            r for r in records if tokenize(r["text_normalized"])
        ]
        if not records:
            raise RuntimeError("No searchable text found")

        tokens = [tokenize(r["text_normalized"]) for r in records]
        state = {
            "source_sha256": fingerprint,
            "tokenizer_version": TOKENIZER_VERSION,
            "records": records,
            "model": BM25Okapi(tokens),
            "vocabulary": set(word for page in tokens for word in page),
        }
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        temporary = INDEX.with_suffix(".tmp")
        with temporary.open("wb") as f:
            pickle.dump(state, f, protocol=4)
        temporary.replace(INDEX)
        print("Built and saved BM25 index:", INDEX)

    records = state["records"]
    print("Searchable pages:", len(records))
    print("Pages per PDF:", dict(Counter(r["source_pdf"] for r in records)))
    RESULTS.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            query = input("\nQuestion (exit to finish): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        query_tokens = tokenize(query)
        if not set(query_tokens) & state["vocabulary"]:
            print("No query words were found in the text index.")
            continue

        scores = state["model"].get_scores(query_tokens)
        order = sorted(
            range(len(records)), key=lambda i: float(scores[i]), reverse=True
        )

        results = []
        for rank, index in enumerate(order[:args.topk], start=1):
            record = records[index]
            result = {
                "rank": rank,
                "corpus_id": record["corpus-id"],
                "score": float(scores[index]),
                "source_pdf": record["source_pdf"],
                "pdf_page_number": record["pdf_page_number"],
                "image_path": record["image_path"],
                "text_preview": record["text_normalized"][:500],
            }
            results.append(result)
            print(f"\n{rank}. {Path(record['source_pdf']).name}")
            print("   PDF page:", record["pdf_page_number"])
            print(f"   BM25 score: {scores[index]:.4f}")
            print("   Image:", ROOT / record["image_path"])

        reference = "fa_digipay-1401-fa_p0052.png"
        reference_positions = [
            rank for rank, index in enumerate(order, start=1)
            if records[index]["corpus-id"] == reference
        ]
        if reference_positions:
            print("\nPage 52 reference rank:", reference_positions[0])

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = RESULTS / f"query_{stamp}.json"
        path.write_text(json.dumps({
            "query": query,
            "query_tokens": query_tokens,
            "method": "BM25",
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved results:", path)

if __name__ == "__main__":
    main()
