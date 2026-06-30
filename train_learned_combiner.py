import os
import gzip
import json
import math
from datetime import datetime
import numpy as np
import lightgbm as lgb

from rank_candidates import (
    evaluate_stage1_boolean,
    calculate_contextual_cooccurrence,
    BM25Okapi,
    clean_and_tokenize,
    open_candidate_stream,
    REFERENCE_DATE,
    TARGET_KEYWORDS,
)

from signal_calibration import (
    calculate_advanced_multipliers,
    blend_relevance,
    calculate_skill_trust,
    _normalize_inverse_band,
    _normalize_to_band,
    _intent_multiplier,
    _market_validation_multiplier,
    _company_scale_multiplier,
    ACTIVITY_SIGMOID_LAMBDA,
    ACTIVITY_SIGMOID_TAU,
    ACTIVITY_FLOOR,
    RESPONSE_TIME_OBS_MIN,
    RESPONSE_TIME_OBS_MAX,
    RESPONSE_TIME_MULT_MIN,
    RESPONSE_TIME_MULT_MAX,
    GITHUB_OBS_MIN,
    GITHUB_OBS_MAX,
    GITHUB_MULT_MIN,
    GITHUB_MULT_MAX,
    OAR_PENALTY_THRESHOLD,
)

INPUT_DATA_PATH = os.path.join("data", "candidates.jsonl")
CROSS_ENC_SCORES_PATH = os.path.join("data", "crossencoder_scores.json.gz")
MODEL_OUT_PATH = "learned_model.txt"

def extract_features_and_targets():
    if not os.path.exists(INPUT_DATA_PATH):
        raise FileNotFoundError(f"Missing input dataset: {INPUT_DATA_PATH}")
    if not os.path.exists(CROSS_ENC_SCORES_PATH):
        raise FileNotFoundError(f"Missing Cross-Encoder scores: {CROSS_ENC_SCORES_PATH}")

    # 1. Load precomputed Cross-Encoder scores
    print(" -> Loading precomputed Cross-Encoder scores...")
    with gzip.open(CROSS_ENC_SCORES_PATH, "rt", encoding="utf-8") as f:
        crossencoder_scores = json.load(f)

    # 2. Scanning for survivors
    print(" -> Ingesting candidate stream and filtering survivors...")
    surviving_candidates = []
    corpus_tokens = []
    
    records = []
    try:
        import sys
        with open_candidate_stream(INPUT_DATA_PATH) as f:
            header_chars = f.read(100).strip()
            
        with open_candidate_stream(INPUT_DATA_PATH) as f:
            if header_chars.startswith('['):
                try:
                    records = json.load(f)
                except Exception as e:
                    raise ValueError(f"Failed to parse candidate JSON array: {e}")
            else:
                malformed_errors = []
                for line_idx, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception as e:
                        snippet = line.strip()[:100]
                        err_msg = f"Line {line_idx} is malformed: {e}. Snippet: '{snippet}'"
                        print(f"WARNING: {err_msg}", file=sys.stderr)
                        malformed_errors.append(err_msg)
                
                if malformed_errors:
                    raise ValueError(
                        f"Encountered {len(malformed_errors)} malformed JSON lines in '{INPUT_DATA_PATH}'. "
                        f"First error: {malformed_errors[0]}"
                    )
    except Exception as e:
        print(f"CRITICAL FILE ERROR: Failed to read or parse candidate data from '{INPUT_DATA_PATH}': {e}")
        raise e

    for record in records:
        if evaluate_stage1_boolean(record):
            surviving_candidates.append(record)
            hist_text = " ".join([f"{j.get('title','')} {j.get('description','')}" for j in record.get('career_history', [])])
            summary_text = record.get('profile', {}).get('summary', '') or ''
            headline_text = record.get('profile', {}).get('headline', '') or ''
            combined_clean_text = f"{headline_text} {summary_text} {hist_text}"
            corpus_tokens.append(clean_and_tokenize(combined_clean_text))

    print(f" -> Found {len(surviving_candidates)} survivors.")

    # 3. Compute BM25 and Co-occurrence scores
    print(" -> Computing BM25 and Co-occurrence scores...")
    query_unigrams_raw = []
    for token in TARGET_KEYWORDS:
        query_unigrams_raw.extend(clean_and_tokenize(token))
    seen_tokens = set()
    query_tokens = [t for t in query_unigrams_raw if not (t in seen_tokens or seen_tokens.add(t))]

    bm25_index = BM25Okapi(corpus_tokens)
    bm25_all_scores = bm25_index.get_scores(query_tokens)
    co_scores = [calculate_contextual_cooccurrence(c, TARGET_KEYWORDS) for c in surviving_candidates]

    bm25_max = max(bm25_all_scores) if bm25_all_scores and max(bm25_all_scores) > 0 else 1.0
    co_max = max(co_scores) if co_scores and max(co_scores) > 0 else 1.0

    # 4. Extract 15 features and compute relevance target for each survivor
    print(" -> Extracting features and generating continuous targets...")
    feature_matrix = []
    target_scores = []
    cids = []

    for idx, candidate in enumerate(surviving_candidates):
        cid = candidate.get('candidate_id')
        signals = candidate.get("redrob_signals", {})
        
        # Relevance sub-components
        bm25_norm = bm25_all_scores[idx] / bm25_max
        cross_score = crossencoder_scores.get(cid, 0.0)  # default fallback if missing
        co_norm = co_scores[idx] / co_max

        # Heuristic relevance and multipliers
        relevance = blend_relevance(bm25_norm, cross_score, co_norm)
        behavioral_multiplier = calculate_advanced_multipliers(candidate, REFERENCE_DATE)
        
        # Target relevance label: current continuous hybrid heuristic score
        y_target = relevance * behavioral_multiplier
        
        # Individual features
        skill_trust = calculate_skill_trust(candidate)
        
        # Activity Decay
        last_act_str = signals.get("last_active_date", REFERENCE_DATE)
        try:
            delta_days = max(0, (datetime.strptime(REFERENCE_DATE, "%Y-%m-%d") - datetime.strptime(last_act_str, "%Y-%m-%d")).days)
        except Exception:
            delta_days = 100
        sigmoid_activity = 1.0 / (1.0 + math.exp(ACTIVITY_SIGMOID_LAMBDA * (delta_days - ACTIVITY_SIGMOID_TAU)))
        activity_decay = max(ACTIVITY_FLOOR, sigmoid_activity)

        # Recruiter RR
        recruiter_rr = signals.get("recruiter_response_rate", 1.0)
        
        # Response Time
        rt = signals.get("avg_response_time_hours", -1)
        if rt is not None and rt != -1:
            rt_norm = _normalize_inverse_band(float(rt), RESPONSE_TIME_OBS_MIN, RESPONSE_TIME_OBS_MAX, RESPONSE_TIME_MULT_MIN, RESPONSE_TIME_MULT_MAX)
        else:
            rt_norm = 1.0

        intent_score = _intent_multiplier(signals)
        market_val = _market_validation_multiplier(signals)
        company_scale = _company_scale_multiplier(candidate)
        icr = signals.get("interview_completion_rate", 1.0)
        
        # Notice period
        notice_days = signals.get("notice_period_days", 0)
        notice_norm = 0.92 if notice_days > 120 else 1.0

        # GitHub
        gh = signals.get("github_activity_score", -1)
        if gh != -1:
            gh_norm = _normalize_to_band(gh, GITHUB_OBS_MIN, GITHUB_OBS_MAX, GITHUB_MULT_MIN, GITHUB_MULT_MAX)
        else:
            gh_norm = 1.0

        # Contact verified
        email_verified = signals.get("verified_email", True)
        phone_verified = signals.get("verified_phone", True)
        contact_verified = 0.88 if (email_verified is False and phone_verified is False) else 1.0

        # OAR
        oar_val = signals.get("offer_acceptance_rate", -1)
        oar = 0.95 if (oar_val != -1 and oar_val < OAR_PENALTY_THRESHOLD) else 1.0

        # Build feature list matching ranker features order
        feats = [
            bm25_norm, cross_score, co_norm,
            skill_trust, activity_decay, recruiter_rr,
            rt_norm, intent_score, market_val,
            company_scale, icr, notice_norm,
            gh_norm, contact_verified, oar
        ]

        feature_matrix.append(feats)
        target_scores.append(y_target)
        cids.append(cid)

    return np.array(feature_matrix), np.array(target_scores), cids

def main():
    print("=== Training LightGBM Regression Combiner ===")
    try:
        X, y, cids = extract_features_and_targets()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return 1

    # Shuffle and split into 80% train and 20% validation sets
    np.random.seed(204)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f" -> Total survivors: {len(X)}")
    print(f" -> Training set size: {len(X_train)} (shape: {X_train.shape})")
    print(f" -> Validation set size: {len(X_val)} (shape: {X_val.shape})")

    # Define explicit feature names
    feature_names = [
        'bm25_norm', 'crossencoder_score', 'co_norm',
        'skill_trust', 'activity_decay', 'recruiter_rr',
        'rt_norm', 'intent_score', 'market_validation',
        'company_scale', 'icr', 'notice_norm',
        'github_norm', 'contact_verified', 'oar'
    ]

    # Create Datasets with locked feature names to guarantee stability
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 16,          # shallow GBDT trees to prevent overfitting
        'min_data_in_leaf': 10,
        'learning_rate': 0.05,
        'verbose': -1,
        'seed': 204,
        'deterministic': True,     # Enforce bit-wise determinism across runs
        'num_threads': 1           # Use single thread to prevent multi-threaded ordering variances
    }

    print(" -> Fitting LightGBM regression ranker with early stopping...")
    # Train booster
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )

    print(f" -> Saving trained model to: {MODEL_OUT_PATH}")
    bst.save_model(MODEL_OUT_PATH)
    
    # Feature importance printout
    importance = bst.feature_importance(importance_type='gain')
    feature_names = [
        'bm25_norm', 'crossencoder_score', 'co_norm',
        'skill_trust', 'activity_decay', 'recruiter_rr',
        'rt_norm', 'intent_score', 'market_validation',
        'company_scale', 'icr', 'notice_norm',
        'github_norm', 'contact_verified', 'oar'
    ]
    
    print("\n=== Feature Importance (Gain) ===")
    sorted_idx = np.argsort(importance)[::-1]
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:<20} : {importance[idx]:.4f}")

    print("\nSUCCESS: GBDT regression model trained and saved successfully.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
