import os
import gzip
import json
from pathlib import Path
from sentence_transformers import CrossEncoder

from rank_candidates import (
    JD_SEMANTIC_ANCHOR,
    evaluate_stage1_boolean,
    open_candidate_stream,
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
LOCAL_MODEL_DIR = os.path.join("models", "ms-marco-MiniLM-L-12-v2")
OUTPUT_PATH = os.path.join("data", "crossencoder_scores.json.gz")
INPUT_DATA_PATH = os.path.join("data", "candidates.jsonl")

def get_cross_encoder():
    """Loads CrossEncoder locally if available, otherwise downloads and saves it."""
    if os.path.exists(LOCAL_MODEL_DIR):
        print(f" -> Loading local CrossEncoder from: {LOCAL_MODEL_DIR}")
        return CrossEncoder(LOCAL_MODEL_DIR)
    else:
        print(f" -> Downloading CrossEncoder model '{MODEL_NAME}'...")
        os.makedirs(os.path.dirname(LOCAL_MODEL_DIR), exist_ok=True)
        model = CrossEncoder(MODEL_NAME)
        print(f" -> Saving CrossEncoder model to: {LOCAL_MODEL_DIR}")
        model.save(LOCAL_MODEL_DIR)
        return model

def main():
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"CRITICAL FILE ERROR: Input dataset path '{INPUT_DATA_PATH}' does not exist.")
        return 1

    # 1. Load Cross-Encoder
    model = get_cross_encoder()

    # 2. Extract Stage 1 survivors
    print("[1/3] Scanning candidates dataset for Stage 1 survivors...")
    surviving_candidates = []
    candidate_texts = []
    
    records = []
    try:
        with open_candidate_stream(INPUT_DATA_PATH) as f:
            header_chars = f.read(100).strip()
            
        with open_candidate_stream(INPUT_DATA_PATH) as f:
            if header_chars.startswith('['):
                records = json.load(f)
            else:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
    except Exception as e:
        print(f"CRITICAL FILE ERROR: Failed to read candidate data: {e}")
        return 1

    for record in records:
        if evaluate_stage1_boolean(record):
            surviving_candidates.append(record)
            
            # Reconstruct candidate text exactly as in rank_candidates.py
            hist_text = " ".join([f"{j.get('title','')} {j.get('description','')}" for j in record.get('career_history', [])])
            summary_text = record.get('profile', {}).get('summary', '') or ''
            headline_text = record.get('profile', {}).get('headline', '') or ''
            combined_clean_text = f"{headline_text} {summary_text} {hist_text}"
            candidate_texts.append(combined_clean_text)

    num_survivors = len(surviving_candidates)
    print(f" -> Found {num_survivors} Stage 1 survivors.")
    if num_survivors == 0:
        print("CRITICAL ERROR: No candidates passed Stage 1.")
        return 1

    # 3. Compute Cross-Encoder relevance scores
    print("[2/3] Computing Cross-Encoder relevance scores paired with JD anchor...")
    pairs = [(JD_SEMANTIC_ANCHOR, text) for text in candidate_texts]
    
    # Predict in batches
    raw_scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # 4. Normalize scores to [0, 1] across the surviving pool
    print("[3/3] Normalizing scores and writing output...")
    min_score = float(raw_scores.min())
    max_score = float(raw_scores.max())
    denom = max_score - min_score if max_score > min_score else 1.0
    
    print(f" -> Raw scores range: [{min_score:.4f}, {max_score:.4f}]")
    
    normalized_scores = {}
    for idx, record in enumerate(surviving_candidates):
        cid = record.get('candidate_id')
        raw_val = float(raw_scores[idx])
        norm_val = (raw_val - min_score) / denom
        normalized_scores[cid] = norm_val

    # 5. Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(normalized_scores, f)
        
    print(f"SUCCESS: Cross-Encoder scores saved to {OUTPUT_PATH} (entries: {len(normalized_scores)}).")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
