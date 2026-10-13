from pathlib import Path
from io import BytesIO
import json

import pymupdf
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data/fa/pdfs/fa-data"
IMAGE_DIR = ROOT / "data/fa/images"
PARQUET_PATH = ROOT / "retrieval/corpus/fa/corpus.parquet"
METADATA_PATH = ROOT / "data/fa/pages.jsonl"
DPI = 150

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if len(pdfs) != 2:
        raise RuntimeError(f"Expected 2 PDFs, found {len(pdfs)}")

    if PARQUET_PATH.exists() or METADATA_PATH.exists():
        raise RuntimeError("Final output already exists; refusing to overwrite.")

    counts = {}
    for path in pdfs:
        with pymupdf.open(path) as doc:
            counts[path.name] = len(doc)

    total = sum(counts.values())
    if total != 122:
        raise RuntimeError(f"Expected 122 pages, found {total}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = []

    for pdf_path in pdfs:
        with pymupdf.open(pdf_path) as doc:
            print(f"{pdf_path.name}: {len(doc)} pages", flush=True)

            for index, page in enumerate(doc):
                page_number = index + 1
                page_id = f"fa_{pdf_path.stem}_p{page_number:04d}.png"
                image_path = IMAGE_DIR / page_id

                pix = page.get_pixmap(
                    dpi=DPI,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                image_bytes = pix.tobytes("png")
                image_path.write_bytes(image_bytes)

                rows.append({
                    "corpus-id": page_id,
                    "image": image_bytes,
                })
                metadata.append({
                    "corpus-id": page_id,
                    "language": "fa",
                    "source_pdf": pdf_path.relative_to(ROOT).as_posix(),
                    "pdf_page_number": page_number,
                    "image_path": image_path.relative_to(ROOT).as_posix(),
                    "width": pix.width,
                    "height": pix.height,
                    "render_dpi": DPI,
                })

                if page_number % 10 == 0 or page_number == len(doc):
                    print(f"  Rendered {page_number}/{len(doc)}", flush=True)

    schema = pa.schema([
        pa.field("corpus-id", pa.string()),
        pa.field("image", pa.binary()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)

    # Write to a temporary file and validate before publishing it.
    temp_path = PARQUET_PATH.with_suffix(".parquet.tmp")
    pq.write_table(table, temp_path, compression="snappy")
    saved = pq.read_table(temp_path)

    ids = saved["corpus-id"].to_pylist()
    assert saved.num_rows == total
    assert len(set(ids)) == total
    assert saved.schema.equals(schema)

    for record, image_bytes in zip(
        metadata, saved["image"].to_pylist()
    ):
        with Image.open(BytesIO(image_bytes)) as image:
            assert image.mode == "RGB"
            assert image.size == (record["width"], record["height"])
            image.verify()

        disk_path = ROOT / record["image_path"]
        assert disk_path.read_bytes() == image_bytes

    with METADATA_PATH.open("x", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_path.rename(PARQUET_PATH)

    print("\nValidation: OK")
    print("Pages:", saved.num_rows)
    print("Unique IDs:", len(set(ids)))
    print("Schema:", saved.schema)
    print("Parquet:", PARQUET_PATH)
    print("Metadata:", METADATA_PATH)
    print("Images:", IMAGE_DIR)

if __name__ == "__main__":
    main()
