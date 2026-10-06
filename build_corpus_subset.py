import json
import random
from pathlib import Path

from huggingface_hub import list_repo_files


RANDOM_SEED = 42
TARGET_CORPUS_SIZE = 500

SUBSET_DIR = Path("subset_metadata")


def load_gold_pages(lang: str):
    path = SUBSET_DIR / f"gold_pages_{lang}.txt"

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_all_corpus_files(lang: str):
    files = list_repo_files(
        repo_id="zhaosuifeng/FinRAGBench-V",
        repo_type="dataset"
    )

    # این بخش را بعداً ممکن است با ساختار واقعی corpus اصلاح کنیم
    # فعلاً فقط فایل‌هایی را می‌گیریم که زیر corpus باشند
    corpus_files = [
        f for f in files
        if f.startswith("corpus/")
    ]

    return corpus_files


def build_subset(lang: str):
    random.seed(RANDOM_SEED)

    gold_pages = set(load_gold_pages(lang))

    all_corpus_files = get_all_corpus_files(lang)

    print("=" * 60)
    print(f"Language: {lang}")
    print(f"Gold pages: {len(gold_pages)}")
    print(f"All corpus files found: {len(all_corpus_files)}")

    # فقط نام فایل را جدا می‌کنیم
    corpus_names = {
        Path(f).name: f
        for f in all_corpus_files
    }

    missing_gold = [
        page for page in gold_pages
        if page not in corpus_names
    ]

    print(f"Gold pages missing from corpus listing: {len(missing_gold)}")

    if missing_gold:
        print("\nExample missing pages:")
        for page in missing_gold[:10]:
            print("  ", page)

    available_distractors = [
        name
        for name in corpus_names.keys()
        if name not in gold_pages
    ]

    need = TARGET_CORPUS_SIZE - len(gold_pages)

    if need < 0:
        raise ValueError("Gold pages exceed target corpus size")

    sampled_distractors = random.sample(
        available_distractors,
        need
    )

    final_pages = sorted(
        list(gold_pages) + sampled_distractors
    )

    output_path = SUBSET_DIR / f"corpus_subset_{lang}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": lang,
                "target_size": TARGET_CORPUS_SIZE,
                "gold_count": len(gold_pages),
                "distractor_count": len(sampled_distractors),
                "pages": final_pages
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Final subset size: {len(final_pages)}")
    print(f"Saved to: {output_path}")
    print()


if __name__ == "__main__":
    build_subset("en")
    build_subset("ch")