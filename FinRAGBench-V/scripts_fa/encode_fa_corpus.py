from pathlib import Path
from io import BytesIO
import sys
import json
import pickle
import hashlib
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import pyarrow.parquet as pq
from PIL import Image
from openmatch.modeling.modeling_colqwen import ColQwen2, ColQwen2Processor

MODEL = ROOT / "retrieval/colqwen2-v1.0-merged"
CORPUS = ROOT / "retrieval/corpus/fa/corpus.parquet"


def save_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        pickle.dump(value, f, protocol=4)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.image_tokens < 1:
        raise ValueError("image-tokens must be positive")

    TOKENS = args.image_tokens
    OUT = (
        ROOT / args.output_dir if args.output_dir
        else ROOT / f"retrieval/encode/colqwen_fa_{TOKENS}"
    )
    PARTS = OUT / "page_checkpoints"
    FINAL = OUT / "embeddings.corpus.rank.0.0-122"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if FINAL.exists():
        raise RuntimeError("Final embeddings already exist; refusing to overwrite")

    records = pq.read_table(CORPUS).to_pylist()
    ids = [r["corpus-id"] for r in records]
    assert len(ids) == 122 and len(set(ids)) == 122
    PARTS.mkdir(parents=True, exist_ok=True)

    settings = {
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "image_token_budget": TOKENS,
        "max_pixels": TOKENS * 28 * 28,
        "batch_size": 1,
        "dtype": "float16",
        "attention": "sdpa",
        "model_files": {
            p.name: {
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in sorted(MODEL.glob("*"))
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}
        },
    }
    settings_path = OUT / "resume_settings.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if previous != settings:
            raise RuntimeError("Resume settings changed; use a new output folder")
    else:
        settings_path.write_text(json.dumps(settings, indent=2))

    print("Loading model with SDPA...", flush=True)
    model = ColQwen2.from_pretrained(
        str(MODEL),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(str(MODEL))

    processor.max_pixels = settings["max_pixels"]
    processor.image_processor.max_pixels = settings["max_pixels"]
    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, dict):
        for key in ("max_pixels", "longest_edge"):
            if key in size:
                size[key] = settings["max_pixels"]

    print("Attention:", model.config._attn_implementation, flush=True)
    print("Vision attention class:",
          type(model.visual.blocks[0].attn).__name__, flush=True)

    for index, record in enumerate(records):
        checkpoint = PARTS / f"page_{index:04d}.pkl"
        if checkpoint.exists():
            with checkpoint.open("rb") as f:
                saved = pickle.load(f)
            if saved["page_id"] != record["corpus-id"]:
                raise RuntimeError("Checkpoint ID mismatch")
            vector = saved["embedding"]
            if vector.ndim != 2 or not torch.isfinite(vector).all().item():
                raise RuntimeError("Invalid checkpoint")
            print(f"Reused {index + 1}/122", flush=True)
            del saved, vector
            continue

        with Image.open(BytesIO(record["image"])) as source:
            image = source.convert("RGB")
        try:
            inputs = processor.process_images([image]).to("cuda:0")
            grid = inputs["image_grid_thw"][0].tolist()

            with torch.inference_mode():
                result = model(**inputs)

            vector = result[0].detach().cpu().clone()
            if vector.ndim != 2 or not torch.isfinite(vector).all().item():
                raise RuntimeError("Invalid embedding")

            save_atomic(checkpoint, {
                "page_id": record["corpus-id"],
                "embedding": vector,
                "image_grid_thw": grid,
            })
            print(
                f"Saved {index + 1}/122 | "
                f"{record['corpus-id']} | shape={tuple(vector.shape)}",
                flush=True,
            )
            del inputs, result, vector
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(
                "\nGPU memory insufficient for this page. "
                "Completed pages remain saved; no resolution fallback applied.",
                flush=True,
            )
            raise
        finally:
            image.close()

    vectors = []
    for index, page_id in enumerate(ids):
        with (PARTS / f"page_{index:04d}.pkl").open("rb") as f:
            saved = pickle.load(f)
        assert saved["page_id"] == page_id
        vectors.append(saved["embedding"])

    save_atomic(FINAL, (vectors, ids))
    with FINAL.open("rb") as f:
        checked_vectors, checked_ids = pickle.load(f)
    assert checked_ids == ids
    assert len(checked_vectors) == len(vectors)
    assert all(torch.equal(a, b) for a, b in zip(vectors, checked_vectors))

    (OUT / "manifest.json").write_text(json.dumps({
        **settings,
        "pages": len(ids),
        "embedding_file": FINAL.name,
        "shapes": [list(v.shape) for v in vectors],
    }, indent=2))

    print("\nValidation: OK")
    print("Saved pages:", len(ids))
    print("Embedding file:", FINAL)


if __name__ == "__main__":
    main()
