from pathlib import Path
import json
import unicodedata
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data/fa/pages.jsonl"
OUTPUT = ROOT / "data/fa/text/pages_text.jsonl"

def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"ي": "ی", "ك": "ک"}))
    return text

def main():
    records = [
        json.loads(line)
        for line in METADATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    documents = {}
    extracted = []

    try:
        for record in records:
            pdf_path = ROOT / record["source_pdf"]
            if pdf_path not in documents:
                documents[pdf_path] = pymupdf.open(pdf_path)

            page = documents[pdf_path][record["pdf_page_number"] - 1]
            raw_text = page.get_text("text", sort=False)
            text = normalize(raw_text)

            extracted.append({
                **record,
                "text_raw": raw_text,
                "text_normalized": text,
                "text_chars": len(text.strip()),
            })
    finally:
        for document in documents.values():
            document.close()

    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in extracted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved pages:", len(extracted))
    print("Pages with no extracted text:",
          sum(r["text_chars"] == 0 for r in extracted))
    print("Output:", OUTPUT)

    samples = [
        "fa_digipay-1400-fa_p0003.png",
        "fa_digipay-1401-fa_p0052.png",
    ]
    for page_id in samples:
        record = next(r for r in extracted if r["corpus-id"] == page_id)
        sample_path = OUTPUT.parent / f"{Path(page_id).stem}.txt"
        sample_path.write_text(
            record["text_normalized"], encoding="utf-8"
        )
        print("\nSample:", page_id)
        print(record["text_normalized"][:2000])
        print("Sample file:", sample_path)

if __name__ == "__main__":
    main()
