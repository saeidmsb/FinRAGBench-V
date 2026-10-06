import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path("hf_meta") / "queries"


def analyze_language(lang: str):
    file_path = BASE_DIR / f"queries_{lang}.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(f"Language: {lang}")
    print(f"Total queries: {len(data)}")
    print()

    category_counts = Counter(item["category"] for item in data)

    print("Category distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print()


if __name__ == "__main__":
    analyze_language("en")
    analyze_language("ch")