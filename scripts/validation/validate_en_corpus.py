import json
from pathlib import Path
from PIL import Image


CORPUS_DIR = Path("mini_corpus/en")
SELECTION_PATH = Path(
    "subset_metadata/final_corpus_selection_en.json"
)
GOLD_PATH = Path(
    "subset_metadata/gold_pages_en.txt"
)


with open(SELECTION_PATH, "r", encoding="utf-8") as f:
    selection = json.load(f)

expected_files = {
    item["output_filename"]
    for item in selection["items"]
}

actual_files = {
    p.name
    for p in CORPUS_DIR.glob("*.png")
}


with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold_files = {
        line.strip()
        for line in f
        if line.strip()
    }


missing_expected = expected_files - actual_files
unexpected_files = actual_files - expected_files
missing_gold = gold_files - actual_files

zero_byte_files = []
corrupt_images = []

for path in CORPUS_DIR.glob("*.png"):

    if path.stat().st_size == 0:
        zero_byte_files.append(path.name)
        continue

    try:
        with Image.open(path) as img:
            img.verify()

    except Exception as e:
        corrupt_images.append(
            {
                "filename": path.name,
                "error": str(e),
            }
        )


multipage_files = sorted(
    name
    for name in actual_files
    if "_multipage_" in name
)


print("=" * 80)
print("ENGLISH CORPUS VALIDATION")
print("=" * 80)

print(f"Expected files: {len(expected_files)}")
print(f"Actual files: {len(actual_files)}")

print(
    f"Missing expected files: "
    f"{len(missing_expected)}"
)

print(
    f"Unexpected files: "
    f"{len(unexpected_files)}"
)

print(f"Gold files expected: {len(gold_files)}")

print(
    f"Missing gold files: "
    f"{len(missing_gold)}"
)

print(
    f"Zero-byte files: "
    f"{len(zero_byte_files)}"
)

print(
    f"Corrupt images: "
    f"{len(corrupt_images)}"
)

print(
    f"MultiPage files: "
    f"{len(multipage_files)}"
)


if missing_expected:
    print("\nMissing expected:")
    for x in sorted(missing_expected):
        print(x)


if unexpected_files:
    print("\nUnexpected:")
    for x in sorted(unexpected_files):
        print(x)


if missing_gold:
    print("\nMissing gold:")
    for x in sorted(missing_gold):
        print(x)


if corrupt_images:
    print("\nCorrupt images:")
    for x in corrupt_images:
        print(x)


print()
print("=" * 80)

if (
    len(actual_files) == 500
    and not missing_expected
    and not unexpected_files
    and not missing_gold
    and not zero_byte_files
    and not corrupt_images
):
    print("VALIDATION PASSED")
else:
    print("VALIDATION FAILED")