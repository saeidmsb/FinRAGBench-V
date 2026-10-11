import json
from pathlib import Path
from PIL import Image


CORPUS_DIR = Path("mini_corpus/ch")

SELECTION_PATH = Path(
    "subset_metadata/final_corpus_selection_ch.json"
)

GOLD_PATH = Path(
    "subset_metadata/gold_pages_ch.txt"
)


# --------------------------------------------------
# Load expected corpus filenames
# --------------------------------------------------

with open(SELECTION_PATH, "r", encoding="utf-8") as f:
    selection = json.load(f)

expected_files = {
    item["output_filename"]
    for item in selection["items"]
}


# --------------------------------------------------
# Read actual PNG files
# --------------------------------------------------

actual_files = {
    p.name
    for p in CORPUS_DIR.glob("*.png")
}


# --------------------------------------------------
# Load expected gold filenames
# --------------------------------------------------

with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold_files = {
        line.strip()
        for line in f
        if line.strip()
    }


# --------------------------------------------------
# Compare expected vs actual
# --------------------------------------------------

missing_expected = expected_files - actual_files
unexpected_files = actual_files - expected_files
missing_gold = gold_files - actual_files


# --------------------------------------------------
# Check files
# --------------------------------------------------

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


# --------------------------------------------------
# Check MultiPage files
# --------------------------------------------------

multipage_files = sorted(
    name
    for name in actual_files
    if "_multipage_" in name
)


# --------------------------------------------------
# Print validation report
# --------------------------------------------------

print("=" * 80)
print("CHINESE CORPUS VALIDATION")
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


# --------------------------------------------------
# Print problems if found
# --------------------------------------------------

if missing_expected:
    print("\nMissing expected files:")
    for x in sorted(missing_expected):
        print(x)


if unexpected_files:
    print("\nUnexpected files:")
    for x in sorted(unexpected_files):
        print(x)


if missing_gold:
    print("\nMissing gold files:")
    for x in sorted(missing_gold):
        print(x)


if zero_byte_files:
    print("\nZero-byte files:")
    for x in sorted(zero_byte_files):
        print(x)


if corrupt_images:
    print("\nCorrupt images:")
    for x in corrupt_images:
        print(x["filename"])
        print("  Error:", x["error"])


if multipage_files:
    print("\nMultiPage files:")
    for x in multipage_files:
        print(x)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=" * 80)

validation_passed = (
    len(actual_files) == 500
    and len(expected_files) == 500
    and not missing_expected
    and not unexpected_files
    and not missing_gold
    and not zero_byte_files
    and not corrupt_images
)

if validation_passed:
    print("VALIDATION PASSED")
else:
    print("VALIDATION FAILED")