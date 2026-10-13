# Persian Financial Retrieval

Run commands from the inner FinRAGBench-V project directory.

## Data
Two Digipay PDFs, 122 pages total.
RGB page images are rendered at 150 DPI.
Parquet columns: corpus-id (string), image (binary).
Page metadata: data/fa/pages.jsonl.
Text extraction uses sort=False. OCR has not been applied.

## ColQwen2-only
Use the original OpenMatch virtual environment.

Encode:
    python scripts_fa/encode_fa_corpus.py --image-tokens 2048

Evaluate:
    python scripts_fa/evaluate_fa_retrieval.py --embedding-dir retrieval/encode/colqwen_fa_2048 --questions data/fa/evaluation/questions_10.json

Interactive retrieval:
    python scripts_fa/retrieve_fa.py --embedding-dir retrieval/encode/colqwen_fa_2048 --topk 5

Encoding uses FP16, SDPA and one page per batch.
Per-page checkpoints support resuming interrupted runs.
Completed embedding files are not overwritten.

## BM25 baseline
    python scripts_fa/retrieve_fa_bm25.py --topk 5

BM25 searches extracted text, not embeddings.
112 pages currently contain searchable text.

## Hybrid retrieval
Use the separate .venv-colpali-test environment.

    .venv-colpali-test/bin/python scripts_fa/download_fa_reranker.py
    .venv-colpali-test/bin/python scripts_fa/retrieve_fa_hybrid.py --topk 5

Pipeline: official ColQwen2 top-20 plus BM25 top-20,
deduplication, then BAAI/bge-reranker-v2-m3 text reranking.

Required local artifacts:
- retrieval/encode/colpali_fa_test/embeddings.corpus.rank.0.0-122
- retrieval/encode/bm25_fa/index.pkl
- models/hf-cache/
- models/reranker_manifest.json
- data/fa/text/pages_text.jsonl

The script that created official ColPali embeddings is retained
in the local diagnostics archive. These embeddings are separate
from the OpenMatch 2048-token embeddings.

Reranking processes at most 512 tokens per question/page pair.
Textless pages are placed after text-scored pages.
Displayed percentage = 100 * sigmoid(raw score).
It is not calibrated confidence. Raw scores are preserved.

## Development results
Ten questions covering only two reference pages.

| Image token budget | Recall@5 | Recall@20 | MRR |
|---|---:|---:|---:|
| 768 | 0.30 | 0.40 | 0.123779 |
| 1536 | 0.50 | 0.80 | 0.386600 |
| 2048 | 0.80 | 1.00 | 0.619708 |

3072 failed with GPU out-of-memory; no complete evaluation exists.
2048 is the provisional development configuration.
No model fine-tuning was performed.
Image processing, batch size and explicit attention settings differ
across some experiments.

Hybrid retrieval ranked the reference first for all four initial
questions covering the same two pages. This is a small development
test, not evidence of general superiority over BM25.

## Storage and reproducibility
Environment snapshots: environment_original.txt and environment_colpali.txt.
Code, questions, metadata and evaluation reports are versioned.
PDFs, images, Parquet, weights, embeddings, environments and backups
remain local and are excluded from ordinary Git.
Only load trusted pickle files.

## Remaining work
- Diverse questions and a separate held-out evaluation set.
- Latency and peak GPU memory measurements.
- Handling pages without extracted text.
- Answer generation and citation evaluation.
