from pathlib import Path

SUBSET_DIR = Path("subset_metadata")

def inspect(lang):
    path = SUBSET_DIR / f"gold_pages_{lang}.txt"

    with open(path, "r", encoding="utf-8") as f:
        pages = [line.strip() for line in f if line.strip()]

    print("=" * 70)
    print("LANG:", lang)
    print("Gold pages:", len(pages))
    print()

    for page in pages[:20]:
        print(page)

inspect("en")
inspect("ch")