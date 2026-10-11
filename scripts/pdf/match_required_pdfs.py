import json
import tarfile
from pathlib import Path


MANIFEST_PATH = Path("subset_metadata/corpus_manifest_en.json")
ARCHIVE_PATH = Path("data_archives/pdfs_for_QA/pdf_en.tar.gz")
OUTPUT_PATH = Path("subset_metadata/matched_pdfs_en.json")


def normalize_name(name: str) -> str:
    """
    Normalize document/PDF names for comparison.
    """
    name = name.strip()

    if name.lower().endswith(".pdf"):
        name = name[:-4]

    return name.strip()


# --------------------------------------------------
# Load required documents from our manifest
# --------------------------------------------------

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

required_documents = list(manifest["documents"].keys())

print("=" * 80)
print(f"Required documents: {len(required_documents)}")
print()


# --------------------------------------------------
# Read filenames from TAR without extracting
# --------------------------------------------------

with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
    members = [
        member.name
        for member in tar.getmembers()
        if member.isfile() and member.name.lower().endswith(".pdf")
    ]


print(f"PDF files inside archive: {len(members)}")
print()


# --------------------------------------------------
# Build PDF name index
# --------------------------------------------------

pdf_index = {}

for member in members:
    filename = Path(member).name
    stem = normalize_name(filename)

    pdf_index.setdefault(stem, []).append(member)


# --------------------------------------------------
# Match required documents
# --------------------------------------------------

matched = {}
missing = []
ambiguous = {}

for document in required_documents:

    normalized_document = normalize_name(document)

    candidates = pdf_index.get(normalized_document, [])

    if len(candidates) == 1:
        matched[document] = candidates[0]

    elif len(candidates) == 0:
        missing.append(document)

    else:
        ambiguous[document] = candidates


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 80)
print("MATCH RESULTS")
print("=" * 80)

print(f"Exact matches: {len(matched)}")
print(f"Missing: {len(missing)}")
print(f"Ambiguous: {len(ambiguous)}")
print()


if missing:
    print("=" * 80)
    print("MISSING DOCUMENTS")
    print("=" * 80)

    for document in missing:
        print(document)

    print()


if ambiguous:
    print("=" * 80)
    print("AMBIGUOUS DOCUMENTS")
    print("=" * 80)

    for document, candidates in ambiguous.items():
        print(document)

        for candidate in candidates:
            print(f"    -> {candidate}")

        print()


# --------------------------------------------------
# Save mapping
# --------------------------------------------------

output = {
    "required_document_count": len(required_documents),
    "matched_count": len(matched),
    "missing_count": len(missing),
    "ambiguous_count": len(ambiguous),

    "matched": matched,
    "missing": missing,
    "ambiguous": ambiguous,
}


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


print("=" * 80)
print(f"Saved: {OUTPUT_PATH}")