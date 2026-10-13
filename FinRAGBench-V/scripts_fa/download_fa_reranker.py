from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "BAAI/bge-reranker-v2-m3"
CACHE = ROOT / "models/hf-cache"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    CACHE.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=str(CACHE)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE),
        torch_dtype=torch.float16,
    ).to("cuda:0").eval()

    inputs = tokenizer(
        [["تعداد تماس‌های ورودی چقدر بوده است؟",
          "تعداد تماس‌های ورودی بیش از ۵۰۰ هزار تماس بوده است."]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cuda:0")

    with torch.inference_mode():
        score = model(**inputs).logits.float().cpu().reshape(-1)

    if score.numel() != 1 or not torch.isfinite(score).all().item():
        raise RuntimeError("Invalid test score")

    info = {
        "model_id": MODEL_ID,
        "revision": getattr(model.config, "_commit_hash", None),
        "cache_dir": str(CACHE),
        "test_score": score.item(),
    }
    manifest = ROOT / "models/reranker_manifest.json"
    manifest.write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    print("\nReranker loading and inference: OK")
    print("Model cache:", CACHE)
    print("Manifest:", manifest)
    print("Test score:", score.item())

if __name__ == "__main__":
    main()
