import argparse
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


def model_weights_path(model_dir):
    return Path(model_dir) / "model.safetensors"


def main():
    parser = argparse.ArgumentParser(description="Download and save a local SentenceTransformers model for offline ranking.")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", default="models/all-MiniLM-L6-v2")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model_name)
    model.save(str(out_path))

    weights = model_weights_path(out_path)
    if not weights.is_file() or weights.stat().st_size < 1_000_000:
        raise SystemExit(f"Model save failed: expected weights at {weights}")

    probe = SentenceTransformer(str(out_path))
    dim_fn = getattr(probe, "get_embedding_dimension", probe.get_sentence_embedding_dimension)
    print(f"Saved embedding model to {out_path} (dimension={dim_fn()}).")


if __name__ == "__main__":
    main()
