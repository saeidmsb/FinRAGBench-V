from pathlib import Path
from collections import defaultdict


SUBSET_DIR = Path("subset_metadata")


def split_page_name(page_name):
    """
    Supports both:

    AIM_CRW_2020_71.png
        -> document = AIM_CRW_2020
        -> page = 71

    some_document_multipage_44-45.png
        -> document = some_document
        -> page = multipage_44-45
    """

    name = page_name.removesuffix(".png")

    if "_multipage_" in name:
        document_name, page_range = name.rsplit("_multipage_", 1)
        page_info = f"multipage_{page_range}"

    else:
        document_name, page_number = name.rsplit("_", 1)
        page_info = page_number

    return document_name, page_info


def analyze(lang):
    path = SUBSET_DIR / f"gold_pages_{lang}.txt"

    with open(path, "r", encoding="utf-8") as f:
        gold_pages = [
            line.strip()
            for line in f
            if line.strip()
        ]

    documents = defaultdict(list)

    failed = []

    for page in gold_pages:
        try:
            document, page_info = split_page_name(page)
            documents[document].append(page_info)

        except Exception:
            failed.append(page)

    print("=" * 80)
    print(f"Language: {lang}")
    print(f"Gold pages: {len(gold_pages)}")
    print(f"Unique source documents: {len(documents)}")
    print(f"Failed filenames: {len(failed)}")
    print()

    for document, pages in sorted(documents.items()):
        pages = sorted(pages)

        print(document)
        print(f"  Gold pages: {pages}")
        print()

    if failed:
        print("FAILED:")
        for item in failed:
            print(" ", item)


if __name__ == "__main__":
    analyze("en")
    analyze("ch")