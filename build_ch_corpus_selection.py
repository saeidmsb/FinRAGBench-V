import json
import random
from pathlib import Path

from pypdf import PdfReader


SEED = 42
TARGET_CORPUS_SIZE = 500

MANIFEST_PATH = Path(
    "subset_metadata/corpus_manifest_ch.json"
)

PDF_DIR = Path(
    "extracted_pdfs/ch"
)

OUTPUT_PATH = Path(
    "subset_metadata/final_corpus_selection_ch.json"
)


def find_pdf(document_name):
    """
    Find PDF case-insensitively:
    supports both .pdf and .PDF
    """
    for path in PDF_DIR.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
            and path.stem == document_name
        ):
            return path

    return None


with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)


documents = manifest["documents"]

gold_items = []
protected_pages = {}
pdf_page_counts = {}
errors = []


# --------------------------------------------------
# Read PDFs and validate gold pages
# --------------------------------------------------

for document_name, items in documents.items():

    pdf_path = find_pdf(document_name)

    if pdf_path is None:
        errors.append(
            f"Missing PDF: {document_name}"
        )
        continue

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)

    except Exception as e:
        errors.append(
            f"Could not read {pdf_path.name}: {e}"
        )
        continue

    pdf_page_counts[document_name] = page_count
    protected_pages[document_name] = set()

    for item in items:

        if item["type"] == "single":

            page_number = item["page"]

            if not (1 <= page_number <= page_count):
                errors.append(
                    f"Invalid gold page: "
                    f"{document_name} -> {page_number} "
                    f"(PDF has {page_count} pages)"
                )
                continue

            protected_pages[document_name].add(
                page_number
            )

            gold_items.append(
                {
                    "type": "gold",
                    "gold_type": "single",
                    "document": document_name,
                    "page": page_number,
                    "output_filename":
                        item["gold_filename"],
                }
            )

        elif item["type"] == "multipage":

            start_page = item["start_page"]
            end_page = item["end_page"]

            if not (
                1 <= start_page <= end_page <= page_count
            ):
                errors.append(
                    f"Invalid multipage gold: "
                    f"{document_name} -> "
                    f"{start_page}-{end_page}"
                )
                continue

            for page_number in range(
                start_page,
                end_page + 1
            ):
                protected_pages[
                    document_name
                ].add(page_number)

            gold_items.append(
                {
                    "type": "gold",
                    "gold_type": "multipage",
                    "document": document_name,
                    "start_page": start_page,
                    "end_page": end_page,
                    "output_filename":
                        item["gold_filename"],
                }
            )


# --------------------------------------------------
# Stop on structural errors
# --------------------------------------------------

if errors:

    print("=" * 80)
    print("ERRORS FOUND")
    print("=" * 80)

    for error in errors:
        print(error)

    raise SystemExit(
        "\nFix errors before sampling."
    )


# --------------------------------------------------
# Build distractor candidate pool
# --------------------------------------------------

candidate_pages = []

for document_name, page_count in pdf_page_counts.items():

    blocked = protected_pages[document_name]

    for page_number in range(
        1,
        page_count + 1
    ):

        if page_number in blocked:
            continue

        candidate_pages.append(
            {
                "type": "distractor",
                "document": document_name,
                "page": page_number,
                "output_filename":
                    f"{document_name}_{page_number}.png",
            }
        )


# --------------------------------------------------
# Sample distractors
# --------------------------------------------------

distractors_needed = (
    TARGET_CORPUS_SIZE - len(gold_items)
)

if len(candidate_pages) < distractors_needed:
    raise RuntimeError(
        f"Not enough distractor candidates. "
        f"Available={len(candidate_pages)}, "
        f"Needed={distractors_needed}"
    )


rng = random.Random(SEED)

selected_distractors = rng.sample(
    candidate_pages,
    distractors_needed
)


# --------------------------------------------------
# Final corpus
# --------------------------------------------------

final_items = gold_items + selected_distractors

final_items = sorted(
    final_items,
    key=lambda x: x["output_filename"]
)


filenames = [
    item["output_filename"]
    for item in final_items
]

duplicate_count = (
    len(filenames) - len(set(filenames))
)

assert len(final_items) == TARGET_CORPUS_SIZE
assert duplicate_count == 0


output = {
    "language": "ch",
    "random_seed": SEED,
    "target_corpus_size": TARGET_CORPUS_SIZE,
    "source_document_count": len(pdf_page_counts),
    "available_pdf_pages": sum(
        pdf_page_counts.values()
    ),
    "gold_item_count": len(gold_items),
    "distractor_candidate_count":
        len(candidate_pages),
    "selected_distractor_count":
        len(selected_distractors),
    "duplicate_count": duplicate_count,
    "items": final_items,
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


single_gold = sum(
    1 for x in gold_items
    if x["gold_type"] == "single"
)

multipage_gold = sum(
    1 for x in gold_items
    if x["gold_type"] == "multipage"
)


print("=" * 80)
print("CHINESE MINI-CORPUS SELECTION")
print("=" * 80)

print(f"Source documents: {len(pdf_page_counts)}")
print(
    f"Available PDF pages: "
    f"{sum(pdf_page_counts.values())}"
)
print(f"Single-page gold items: {single_gold}")
print(f"MultiPage gold items: {multipage_gold}")
print(f"Total gold items: {len(gold_items)}")
print(
    f"Distractor candidates: "
    f"{len(candidate_pages)}"
)
print(
    f"Selected distractors: "
    f"{len(selected_distractors)}"
)
print(f"Final corpus items: {len(final_items)}")
print(f"Duplicate filenames: {duplicate_count}")
print(f"Random seed: {SEED}")

print()
print(f"Saved: {OUTPUT_PATH}")