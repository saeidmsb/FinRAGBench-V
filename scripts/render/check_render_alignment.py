import json
from pathlib import Path


TARGET_PAGE = "AIM_CRW_2020_71.png"

QUERY_FILE = Path(
    "subset_metadata/selected_queries_en.json"
)

QREL_FILE = Path(
    "subset_metadata/query_to_gold_pages_en.json"
)


with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)

with open(QREL_FILE, "r", encoding="utf-8") as f:
    qrels = json.load(f)


query_map = {
    item["query-id"]: item
    for item in queries
}


found = False

for query_id, pages in qrels.items():

    if TARGET_PAGE not in pages:
        continue

    found = True

    query = query_map[query_id]

    print("=" * 80)
    print("Gold page:")
    print(TARGET_PAGE)

    print("\nQuery ID:")
    print(query_id)

    print("\nQuestion:")
    print(query["query"])

    print("\nAnswer:")
    print(query["answer"])

    print("\nCategory:")
    print(query["category"])


if not found:
    print("No selected query found for:", TARGET_PAGE)