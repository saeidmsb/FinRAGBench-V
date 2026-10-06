from huggingface_hub import list_repo_files

REPO_ID = "zhaosuifeng/FinRAGBench-V"

files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

qa_pdf_files = [
    f for f in files
    if f.startswith("pdfs_for_QA/")
]

print(f"Total files under pdfs_for_QA/: {len(qa_pdf_files)}")
print()

for f in qa_pdf_files:
    print(f)