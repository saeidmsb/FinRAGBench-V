from pathlib import Path
from pypdf import PdfReader


PDF_DIR = Path("extracted_pdfs/en")

total_pages = 0
pdf_count = 0

print("=" * 80)

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    try:
        reader = PdfReader(str(pdf_path))
        pages = len(reader.pages)

        pdf_count += 1
        total_pages += pages

        print(f"{pdf_path.name}")
        print(f"  Pages: {pages}")

    except Exception as e:
        print(f"ERROR: {pdf_path.name}")
        print(f"  {e}")

print("=" * 80)
print(f"PDF count: {pdf_count}")
print(f"Total pages: {total_pages}")