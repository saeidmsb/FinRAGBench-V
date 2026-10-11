import io
import json
from pathlib import Path

import pymupdf
from PIL import Image


DPI = 150

SELECTION_PATH = Path(
    "subset_metadata/final_corpus_selection_ch.json"
)

PDF_DIR = Path(
    "extracted_pdfs/ch"
)

OUTPUT_DIR = Path(
    "mini_corpus/ch"
)

SUMMARY_PATH = Path(
    "subset_metadata/render_summary_ch.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pdf(document_name):
    for path in PDF_DIR.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
            and path.stem == document_name
        ):
            return path
    return None


def pixmap_to_pil(pix):
    png_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def render_single_page(pdf_path, page_number, output_path):
    doc = pymupdf.open(pdf_path)

    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=DPI, alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(output_path)

        return {
            "width": pix.width,
            "height": pix.height,
            "file_size_bytes": output_path.stat().st_size,
        }

    finally:
        doc.close()


def render_multipage_vertical(pdf_path, start_page, end_page, output_path):
    doc = pymupdf.open(pdf_path)

    try:
        rendered_images = []

        for page_number in range(start_page, end_page + 1):
            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=DPI, alpha=False)
            rendered_images.append(pixmap_to_pil(pix))

        total_width = max(img.width for img in rendered_images)
        total_height = sum(img.height for img in rendered_images)

        canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

        y_offset = 0
        for img in rendered_images:
            canvas.paste(img, (0, y_offset))
            y_offset += img.height

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

        return {
            "width": total_width,
            "height": total_height,
            "file_size_bytes": output_path.stat().st_size,
            "component_page_count": len(rendered_images),
        }

    finally:
        doc.close()


with open(SELECTION_PATH, "r", encoding="utf-8") as f:
    selection = json.load(f)

items = selection["items"]

rendered = []
errors = []

print("=" * 80)
print("RENDERING CHINESE MINI-CORPUS")
print("=" * 80)
print(f"Items to render: {len(items)}")
print()

for idx, item in enumerate(items, start=1):

    output_filename = item["output_filename"]
    output_path = OUTPUT_DIR / output_filename
    document_name = item["document"]
    pdf_path = find_pdf(document_name)

    print(f"[{idx}/{len(items)}] {output_filename}")

    if pdf_path is None:
        errors.append(
            {
                "output_filename": output_filename,
                "error": f"Missing PDF for document: {document_name}"
            }
        )
        print("  ERROR: Missing PDF")
        continue

    try:
        if item["type"] == "distractor":
            meta = render_single_page(
                pdf_path=pdf_path,
                page_number=item["page"],
                output_path=output_path,
            )

        elif item["type"] == "gold":
            if item["gold_type"] == "single":
                meta = render_single_page(
                    pdf_path=pdf_path,
                    page_number=item["page"],
                    output_path=output_path,
                )
            elif item["gold_type"] == "multipage":
                meta = render_multipage_vertical(
                    pdf_path=pdf_path,
                    start_page=item["start_page"],
                    end_page=item["end_page"],
                    output_path=output_path,
                )
            else:
                raise ValueError(f"Unknown gold_type: {item['gold_type']}")
        else:
            raise ValueError(f"Unknown item type: {item['type']}")

        rendered.append(
            {
                "output_filename": output_filename,
                "document": document_name,
                "type": item["type"],
                **meta,
            }
        )

        print(
            f"  OK | {meta['width']}x{meta['height']} | "
            f"{meta['file_size_bytes'] / 1024:.1f} KB"
        )

    except Exception as e:
        errors.append(
            {
                "output_filename": output_filename,
                "error": str(e)
            }
        )
        print(f"  ERROR: {e}")


output_files = list(OUTPUT_DIR.glob("*.png"))
total_size_bytes = sum(p.stat().st_size for p in output_files)

summary = {
    "language": "ch",
    "dpi": DPI,
    "selection_path": str(SELECTION_PATH),
    "output_dir": str(OUTPUT_DIR),
    "expected_item_count": len(items),
    "rendered_item_count": len(rendered),
    "error_count": len(errors),
    "output_file_count": len(output_files),
    "total_output_size_bytes": total_size_bytes,
    "rendered": rendered,
    "errors": errors,
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print()
print("=" * 80)
print("RENDER SUMMARY")
print("=" * 80)
print(f"Expected items: {len(items)}")
print(f"Rendered items: {len(rendered)}")
print(f"Errors: {len(errors)}")
print(f"Output files: {len(output_files)}")
print(f"Total output size: {total_size_bytes / (1024 * 1024):.2f} MB")
print(f"Saved summary: {SUMMARY_PATH}")

if errors:
    print("\nSome renders failed. Inspect render_summary_ch.json")