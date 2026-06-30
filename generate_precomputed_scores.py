import argparse
import gzip
import json
import os

from sentence_transformers import SentenceTransformer, util

from rank_candidates import JD_SEMANTIC_ANCHOR, evaluate_stage1_boolean

def main():
    parser = argparse.ArgumentParser(description="Regenerate precomputed semantic similarity scores.")
    parser.add_argument("--model", default=os.path.join("models", "all-MiniLM-L6-v2"))
    parser.add_argument("--output", default=os.path.join("data", "semantic_scores.json.gz"))
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Local model path '{model_path}' not found.")
        print("Run: python prepare_embedding_model.py --out models/all-MiniLM-L6-v2")
        return 1

    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.isfile(weights_path):
        print(f"Error: Model weights missing at '{weights_path}'.")
        print("Run: python prepare_embedding_model.py --out models/all-MiniLM-L6-v2")
        return 1
        
    print("Loading SentenceTransformers model...")
    model = SentenceTransformer(model_path)
    jd_embedding = model.encode(JD_SEMANTIC_ANCHOR, convert_to_tensor=True)
    
    # Process both candidates.jsonl and sample_candidates.jsonl
    files_to_process = [
        "data/candidates.jsonl",
        "data/sample_candidates.jsonl"
    ]
    
    survivors = {}
    
    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping.")
            continue
            
        print(f"Scanning {file_path} for survivors...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if evaluate_stage1_boolean(record):
                    cid = record.get('candidate_id')
                    if cid and cid not in survivors:
                        hist_text = " ".join([f"{j.get('title','')} {j.get('description','')}" for j in record.get('career_history', [])])
                        summary_text = record.get('profile', {}).get('summary', '') or ''
                        combined_clean_text = f"{summary_text} {hist_text}"
                        survivors[cid] = combined_clean_text

    print(f"Found {len(survivors)} unique surviving candidates across both datasets.")
    
    # Compute embeddings and similarities
    cids = list(survivors.keys())
    texts = [survivors[cid] for cid in cids]
    
    print("Computing embeddings...")
    embeddings = model.encode(texts, convert_to_tensor=True, batch_size=128, show_progress_bar=True)
    
    print("Calculating similarity scores...")
    similarities = util.cos_sim(embeddings, jd_embedding).cpu().numpy().flatten()
    
    semantic_scores = {}
    for cid, sim in zip(cids, similarities):
        semantic_scores[cid] = float(sim)
        
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving precomputed scores to {args.output}...")
    with gzip.open(args.output, "wt", encoding="utf-8") as f:
        json.dump(semantic_scores, f)

    print(f"Done! Saved {len(semantic_scores)} precomputed scores.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
