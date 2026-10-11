from huggingface_hub import list_repo_files

repo_id = "zhaosuifeng/FinRAGBench-V"

files = list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"
)

print("=== English corpus files ===")
for f in files:
    if f.startswith("corpus/en/"):
        print(f)

print("\n=== Chinese corpus files ===")
for f in files:
    if f.startswith("corpus/ch/"):
        print(f)