<img src="assets/logo.png" alt="FinRAGBench-V Logo" width="80" align="left" />

# FinRAGBench-V: A Benchmark for Visual RAG in the Financial Domain

FinRAGBench-V is a **comprehensive benchmark for visual retrieval-augmented generation (RAG) in finance**, addressing the challenge that most existing financial RAG research focuses predominantly on text while overlooking rich visual content in financial documents. By integrating multimodal data and providing **visual citation**, FinRAGBench-V ensures traceability and supports robust evaluation of Multimodal Large Language Models (MLLMs).

This repository also includes a **Persian document retrieval implementation** using Digipay financial reports. This extension explores page-level visual retrieval for Persian financial documents and is documented separately from the original benchmark.

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/zhaosuifeng/FinRAGBench-V">Dataset</a>
  · 📄 <a href="https://arxiv.org/abs/2505.17471">Original Paper</a>
  · 🇮🇷 <a href="docs/reports/fa/FinRAGBench_FA_Report.pdf">Persian Implementation Report</a>
</p>

<p align="center">
  <img src="assets/main_fig.png" alt="FinRAGBench-V main figure" width="100%" />
</p>

## Benchmark: FinRAGBench-V

- 📊 **Multimodal Retrieval Corpus:** We construct a multimodal financial corpus by collecting documents from various real-world financial sources, including research reports, financial statements, prospectuses, academic papers, financial magazines, and financial news. The corpus contains 60,780 Chinese pages and 51,219 English pages from 1,104 Chinese and 1,105 English documents.

- 📝 **High-Quality QA Dataset:** We construct a high-quality, human-annotated question-answering (QA) dataset spanning heterogeneous data types (charts, tables, and texts) and seven question categories, including time-sensitive questions, numerical calculations, comparison and sorting, and multi-page queries.

## Baseline: RGenCite

We provide **RGenCite**, a multimodal RAG baseline that seamlessly integrates:

- 🔍 **Retrieval:** Efficiently retrieves relevant textual and visual information from the dataset.
- ✍️ **Generation:** Produces high-quality answers grounded in retrieved content.
- 📌 **Fine-Grained Visual Citation:** Provides precise visual evidence at the page and block levels to support answers.

## Automatic Citation Evaluation

We propose an **automatic evaluation method** for visual citation, which:

- Measures **precision and recall** at multiple citation levels.
- Uses **bounding boxes** and **image cropping** to assess the alignment of cited visual evidence.

## Persian Financial Document Retrieval

![Persian visual retrieval — Digipay case study](assets/digipay_visual_retrieval.png)
This extension applies page-level visual retrieval to **Persian financial reports**, using Digipay reports as a case study.

Unlike a text-only pipeline, this approach represents each PDF page as an image, preserving its original text, tables, charts, and layout. Persian queries are then used to retrieve relevant page images.

### Implementation Overview

The Persian retrieval pipeline consists of:

1. **PDF preparation:** Convert financial report pages into images and retain document and page identifiers.
2. **Corpus construction:** Store page images in Parquet format with associated metadata.
3. **Visual encoding:** Generate multi-vector page representations using **ColQwen2**.
4. **Query encoding and retrieval:** Encode Persian queries and rank pages using **MaxSim** scoring.
5. **Evidence inspection:** Return the highest-ranked pages for inspection and retrieval evaluation.

### Scope

This extension focuses on **retrieving relevant pages from Persian financial reports**.

It does not, by itself, constitute a complete Persian version of the original benchmark. Answer generation and fine-grained visual citation evaluation are separate stages and should be assessed independently.

Retrieval performance should also be distinguished from the ability to interpret financial evidence or predict future company performance.

### Persian Report

The report documents the Persian implementation, data preparation, retrieval workflow, experimental results, and limitations.

📄 **[View the Persian Implementation and Evaluation Report (PDF)](docs/reports/fa/FinRAGBench_FA_Report.pdf)**

### توضیحات فارسی

<div dir="rtl" align="right">

این بخش، پیاده‌سازی بازیابی بصری صفحات گزارش‌های مالی فارسی را با استفاده از گزارش‌های دیجی‌پی معرفی می‌کند.

در این روش، هر صفحهٔ PDF به تصویر تبدیل می‌شود تا متن، جدول‌ها، نمودارها و چیدمان اصلی آن حفظ شود. سپس مدل ColQwen2 بازنمایی برداری صفحات و پرسش را تولید می‌کند و صفحات مرتبط با پرسش فارسی، با استفاده از امتیازدهی MaxSim رتبه‌بندی می‌شوند.

مراحل اصلی پیاده‌سازی:

- تبدیل صفحات گزارش‌های مالی به تصویر و ثبت شناسهٔ سند و صفحه.
- آماده‌سازی مجموعهٔ صفحات در قالب Parquet.
- تولید بازنمایی‌های برداری چندگانه برای صفحات و پرسش‌های فارسی.
- بازیابی صفحات مرتبط و بررسی نتایج و محدودیت‌های ارزیابی.

تمرکز این پیاده‌سازی بر بازیابی صفحه است؛ کیفیت تولید پاسخ و استناد بصری باید به‌صورت جداگانه ارزیابی شود. همچنین این گزارش، مستند پیاده‌سازی و آزمایش‌های فارسی است و ترجمهٔ مقالهٔ اصلی محسوب نمی‌شود.

<a href="docs/reports/fa/FinRAGBench_FA_Report.pdf">📄 مشاهدهٔ گزارش پیاده‌سازی و ارزیابی فارسی</a>

</div>

## Installation

```bash
git clone https://github.com/zhaosuifeng/FinRAGBench-V.git
cd FinRAGBench-V

conda create --name finragbench python=3.11 -y
conda activate finragbench

pip install -r requirements.txt
pip install -e .
```

> The clone URL above points to the original repository. To work with the Persian extension, clone the repository containing this extension.

## Dataset Download and Preprocessing

First, download all dataset files, including **corpus**, **queries**, and **qrels**, from [zhaosuifeng/FinRAGBench-V](https://huggingface.co/datasets/zhaosuifeng/FinRAGBench-V/tree/main) and place them into the corresponding folders under `./data/`.

The corpus is provided in multiple split parts (`part_*.tar.gz`). Please merge them before extraction.

### Example: English Corpus

```bash
# Create the corpus folder
mkdir -p ./data/corpus/en
cd ./data/corpus/en

# Merge all parts into a single archive
cat part_*.tar.gz > corpus_en.tar.gz

# Extract the archive
tar -xzvf corpus_en.tar.gz

# Navigate to the preprocessing directory
cd ../../../prepare_data

# Convert corpus and queries into Parquet format
python generate_parquet.py
```

The instructions above concern the original benchmark dataset. The Persian corpus preparation workflow is documented in the [Persian implementation report](docs/reports/fa/FinRAGBench_FA_Report.pdf).

## Retrieval

From the repository root:

```bash
cd retrieval
python eval_mm_retriever.py
```

Use `encode_config.json` for encoding and `retrieve_config.json` for retrieval.

For the Persian implementation, model settings and corpus paths should match the prepared Persian data. See the [Persian report](docs/reports/fa/FinRAGBench_FA_Report.pdf) for implementation details.

## Generation

From the repository root:

```bash
cd generation
python generate.py
```

## Evaluation

From the repository root:

```bash
cd eval
python eval_generation.py
python eval_citation.py
```

These commands belong to the original generation and citation evaluation workflow. They should not be interpreted as evidence that these stages have been evaluated for the Persian extension.

## Documentation

| Document | Description |
|---|---|
| [Original Paper](https://arxiv.org/abs/2505.17471) | Original FinRAGBench-V benchmark, methodology, and evaluation |
| [Persian Implementation Report](docs/reports/fa/FinRAGBench_FA_Report.pdf) | Persian financial document preparation, visual retrieval implementation, experiments, and limitations |

## Other Related Projects

- [OpenMatch](https://github.com/thunlp/OpenMatch)

## Attribution

The original **FinRAGBench-V benchmark**, **RGenCite baseline**, and **visual citation evaluation methodology** are the work of the original authors.

The Persian implementation is an extension built upon this project. Its implementation details and experiments are described in the Persian report.

## Citation

If you find the original FinRAGBench-V work useful, please cite:

```bibtex
@article{zhao2025finragbench,
  title={FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain},
  author={Zhao, Suifeng and Jin, Zhuoran and Li, Sujian and Gao, Jun},
  journal={arXiv preprint arXiv:2505.17471},
  year={2025}
}
```
