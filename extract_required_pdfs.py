import json
import tarfile
from pathlib import Path


MATCH_PATH = Path("subset_metadata/matched_pdfs_en.json")
ARCHIVE_PATH = Path("data_archives/pdfs_for_QA/pdf_en.tar.gz")
OUTPUT_DIR = Path("extracted_pdfs/en")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(MATCH_PATH, "r", encoding="utf-8") as f:
    match_data = json.load(f)

matched = match_data["matched"]

members_to_extract = list(matched.values())

print(f"PDFs to extract: {len(members_to_extract)}")


with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:

    for i, member_name in enumerate(members_to_extract, start=1):

        print(
            f"[{i}/{len(members_to_extract)}] "
            f"Extracting: {member_name}"
        )

        member = tar.getmember(member_name)

        source = tar.extractfile(member)

        if source is None:
            raise RuntimeError(
                f"Could not read: {member_name}"
            )

        output_path = OUTPUT_DIR / Path(member_name).name

        with open(output_path, "wb") as out_file:
            out_file.write(source.read())


print()
print("Extraction complete.")
print(f"Saved to: {OUTPUT_DIR}")