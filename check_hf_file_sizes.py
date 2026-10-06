from huggingface_hub import HfApi

REPO_ID = "zhaosuifeng/FinRAGBench-V"

api = HfApi()

info = api.repo_info(
    repo_id=REPO_ID,
    repo_type="dataset",
    files_metadata=True,
)

targets = {
    "pdfs_for_QA/pdf_en.tar.gz",
    "pdfs_for_QA/pdf_ch.tar.gz",
}

for file in info.siblings:
    if file.rfilename in targets:
        size_bytes = file.size

        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)

        print(file.rfilename)
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Size: {size_gb:.2f} GB")
        print()
        