import json
import re
from pathlib import Path
from collections import defaultdict


SUBSET_DIR = Path("subset_metadata")


def parse_gold_filename(filename):
    """
    Supported formats:

    document_71.png

    document_multipage_44-45.png
    """

    name = filename.removesuffix(".png")

    # MultiPage
    match = re.match(
        r"^(.*)_multipage_(\d+)-(\d+)$",
        name
    )

    if match:
        return {
            "document": match.group(1),
            "type": "multipage",
            "start_page": int(match.group(2)),
            "end_page": int(match.group(3)),
            "gold_filename": filename,
        }

    # Normal page
    match = re.match(
        r"^(.*)_(\d+)$",
        name
    )

    if match:
        return {
            "document": match.group(1),
            "type": "single",
            "page": int(match.group(2)),
            "gold_filename": filename,
        }

    raise ValueError(
        f"Unknown gold filename format: {filename}"
    )


def build_manifest(lang):
    gold_path = (
        SUBSET_DIR /
        f"gold_pages_{lang}.txt"
    )

    with open(
        gold_path,
        "r",
        encoding="utf-8"
    ) as f:
        gold_files = [
            line.strip()
            for line in f
            if line.strip()
        ]

    documents = defaultdict(list)

    for filename in gold_files:

        parsed = parse_gold_filename(filename)

        document = parsed.pop("document")

        documents[document].append(parsed)

    manifest = {
        "language": lang,
        "target_corpus_size": 500,
        "gold_count": len(gold_files),
        "source_document_count": len(documents),
        "random_seed": 42,
        "documents": dict(documents),
    }

    output = (
        SUBSET_DIR /
        f"corpus_manifest_{lang}.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 70)
    print(f"Language: {lang}")
    print(f"Gold items: {len(gold_files)}")
    print(f"Source documents: {len(documents)}")
    print(f"Saved: {output}")

    multipage_count = 0

    for items in documents.values():
        for item in items:
            if item["type"] == "multipage":
                multipage_count += 1

    print(
        f"MultiPage gold items: {multipage_count}"
    )

    print()


if __name__ == "__main__":
    build_manifest("en")
    build_manifest("ch")