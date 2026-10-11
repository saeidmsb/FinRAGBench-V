import csv
import json
from pathlib import Path


BASE_DIR = Path("hf_meta")
SUBSET_DIR = Path("subset_metadata")
OUTPUT_DIR = Path("subset_metadata")


def load_selected_queries(lang: str):
    path = SUBSET_DIR / f"selected_queries_{lang}.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qrels(lang: str):
    path = BASE_DIR / "qrels" / f"qrels_{lang}.tsv"

    qrels = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]

            qrels.setdefault(query_id, []).append(corpus_id)

    return qrels


def extract_gold_pages(lang: str):
    selected_queries = load_selected_queries(lang)
    qrels = load_qrels(lang)

    gold_pages = set()
    missing_queries = []

    query_to_pages = {}

    for item in selected_queries:
        query_id = item["query-id"]

        pages = qrels.get(query_id, [])

        if not pages:
            missing_queries.append(query_id)
            continue

        query_to_pages[query_id] = pages

        for page in pages:
            gold_pages.add(page)

    output_json = OUTPUT_DIR / f"query_to_gold_pages_{lang}.json"
    output_txt = OUTPUT_DIR / f"gold_pages_{lang}.txt"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            query_to_pages,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(output_txt, "w", encoding="utf-8") as f:
        for page in sorted(gold_pages):
            f.write(page + "\n")

    print("=" * 60)
    print(f"Language: {lang}")
    print(f"Selected queries: {len(selected_queries)}")
    print(f"Unique gold pages: {len(gold_pages)}")
    print(f"Queries without qrels: {len(missing_queries)}")

    if missing_queries:
        print("\nMissing query IDs:")
        for qid in missing_queries:
            print("  ", qid)

    print(f"\nSaved:")
    print(f"  {output_json}")
    print(f"  {output_txt}")
    print()


if __name__ == "__main__":
    extract_gold_pages("en")
    extract_gold_pages("ch")