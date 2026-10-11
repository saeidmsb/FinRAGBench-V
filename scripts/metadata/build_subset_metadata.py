import json
import random
from pathlib import Path


RANDOM_SEED = 42

BASE_DIR = Path("hf_meta")
QUERY_DIR = BASE_DIR / "queries"

OUTPUT_DIR = Path("subset_metadata")
OUTPUT_DIR.mkdir(exist_ok=True)


TARGET_COUNTS = {
    "Text Inference": 7,
    "Chart-Information Extraction": 7,
    "Chart-Numerical Calculation": 7,
    "Chart-Time Sensitive": 7,
    "Table-Numerical Calculation": 7,
    "Table-Compare and Sort": 7,
    "MultiPage": 8,
}


def normalize_category(category: str) -> str:
    """
    Convert all MultiPage variants into a single canonical category.
    """
    if "multipage" in category.lower():
        return "MultiPage"

    return category


def build_subset(lang: str):
    random.seed(RANDOM_SEED)

    query_path = QUERY_DIR / f"queries_{lang}.json"

    with open(query_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize category names
    for item in data:
        item["normalized_category"] = normalize_category(
            item["category"]
        )

    selected = []

    for category, target_count in TARGET_COUNTS.items():

        candidates = [
            item
            for item in data
            if item["normalized_category"] == category
        ]

        print(
            f"{lang} | {category}: "
            f"{len(candidates)} available -> "
            f"{target_count} selected"
        )

        sampled = random.sample(
            candidates,
            target_count
        )

        selected.extend(sampled)

    # Shuffle final list
    random.shuffle(selected)

    output_path = (
        OUTPUT_DIR
        / f"selected_queries_{lang}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selected,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Total selected ({lang}): {len(selected)}")
    print(f"Saved to: {output_path}")
    print()


if __name__ == "__main__":

    build_subset("en")
    build_subset("ch")