import json
import csv
from pathlib import Path


def build_subset_qrels(
    selected_queries_path,
    qrels_path,
    output_path,
):
    selected_queries_path = Path(selected_queries_path)
    qrels_path = Path(qrels_path)
    output_path = Path(output_path)

    with open(
        selected_queries_path,
        "r",
        encoding="utf-8"
    ) as f:
        selected = json.load(f)

    selected_ids = {
        str(item["query-id"])
        for item in selected
    }

    rows = []

    with open(
        qrels_path,
        "r",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        fieldnames = reader.fieldnames

        for row in reader:
            if str(row["query-id"]) in selected_ids:
                rows.append(row)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t"
        )

        writer.writeheader()
        writer.writerows(rows)

    found_query_ids = {
        str(row["query-id"])
        for row in rows
    }

    missing = selected_ids - found_query_ids

    print("=" * 70)
    print(f"Selected queries: {len(selected_ids)}")
    print(
        f"Queries found in subset qrels: "
        f"{len(found_query_ids)}"
    )
    print(f"Qrels rows: {len(rows)}")
    print(f"Missing queries: {len(missing)}")
    print(f"Saved: {output_path}")

    if missing:
        print("\nMissing query IDs:")
        for qid in sorted(missing):
            print(qid)


print("\nENGLISH")
build_subset_qrels(
    "subset_metadata/selected_queries_en.json",
    "hf_meta/qrels/qrels_en.tsv",
    "subset_metadata/qrels_en_subset.tsv",
)

print("\nCHINESE")
build_subset_qrels(
    "subset_metadata/selected_queries_ch.json",
    "hf_meta/qrels/qrels_ch.tsv",
    "subset_metadata/qrels_ch_subset.tsv",
)