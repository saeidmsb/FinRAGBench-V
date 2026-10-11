from pathlib import Path
import fitz


PDF_DIR = Path("extracted_pdfs/en")
OUTPUT_DIR = Path("render_test")
OUTPUT_DIR.mkdir(exist_ok=True)

# One known document/page from our gold set
DOCUMENT = "AIM_CRW_2020"
PAGE_NUMBER = 71

pdf_path = PDF_DIR / f"{DOCUMENT}.pdf"

doc = fitz.open(pdf_path)

# PDF page numbers in our metadata are 1-based
page = doc[PAGE_NUMBER - 1]

# Render at 150 DPI
pix = page.get_pixmap(dpi=150, alpha=False)

output_path = OUTPUT_DIR / f"{DOCUMENT}_{PAGE_NUMBER}.png"

pix.save(output_path)

print("Saved:", output_path)
print("Width:", pix.width)
print("Height:", pix.height)
print("File size:", output_path.stat().st_size / 1024, "KB")