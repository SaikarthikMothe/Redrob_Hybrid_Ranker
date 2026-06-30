# Redrob Hybrid Ranker

CPU-only submission ranker for 
the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Files

- `rank_candidates.py` generates the submission CSV.
- `team_Jarvis.csv` is the generated and validator-passing submission file.
- `validate_submission.py` is the official CSV format validator.
- `signal_calibration.py` holds empirically tuned multiplier constants (see below).
- `analyze_signals.py` reproduces correlation/distribution analysis on the surviving pool.
- `docs/SIGNAL_CALIBRATION.md` is the judge-facing calibration report.
- `data/candidates.jsonl` is the candidate dataset used by the ranker.
- `data/sample_candidates.jsonl` is a lightweight sample dataset (2969 candidates) designed so that exactly 100 candidates survive the Stage 1 gates, ensuring the format validator passes.
- `sandbox.ipynb` is a runnable Jupyter Notebook designed to run end-to-end in Google Colab.
- `submission_metadata.yaml` contains metadata for the portal and Stage 3 review.

## Reproduce

To reproduce our official submission using the pure Heuristic ranker (which avoids GBDT circularity and BM25 lexical dominance, and is our primary submitted system):
```bash
python rank_candidates.py --out team_Jarvis.csv
python validate_submission.py team_Jarvis.csv
python analyze_signals.py   # reproduce empirical calibration stats
```

## Alternative Run (GBDT LambdaMART Combiner)

The pipeline also supports the machine-learned GBDT ranker trained on heuristic targets (distilled rules):
```bash
python rank_candidates.py --use-learned-combiner --out team_Jarvis.csv
python validate_submission.py team_Jarvis.csv
python analyze_signals.py   # reproduce empirical calibration stats
```

### Advanced Custom Runs & Embedding Regeneration
If you run the pipeline on custom/unseen datasets or wish to regenerate the Cross-Encoder scores from scratch:
1. Prepare your environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Cross-Encoder scorer locally on CPU (uses `models/ms-marco-MiniLM-L-12-v2`):
   ```bash
   python generate_crossencoder_scores.py
   ```
3. Run the ranker. If a candidate is not found in the precomputed scores, the pipeline will automatically fall back to computing the embedding dynamically:
   ```bash
   python rank_candidates.py --embedding-model models/ms-marco-MiniLM-L-6-v2
   ```
4. If you wish to disable the semantic signal entirely and run in lexical-only mode:
   ```bash
   python rank_candidates.py --no-embeddings
   ```

## Sandbox (Google Colab)

A hosted, runnable sandbox of the ranking system is available on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiKarthikMothe/Redrob_Hybrid_Ranker/blob/main/sandbox.ipynb)

By default, the Google Colab environment is configured to clone this repository, set up the environment, and run the pipeline end-to-end.

---

## System Architecture

The ranking engine employs a multi-stage pipeline designed for execution speed, low memory footprints on CPU-only hosts, and robust signal integration.

```mermaid
graph TD
    A[Input: Candidates Dataset] --> B[Stage 1: Boolean Hard Gates]
    B -->|Filter: YOE, Location, Relocation, Work Mode, Banned Titles| C[Gate Survivors: ~3% Footprint]
    C --> D[Stage 2: Hybrid Scoring Engine]
    D -->|Lexical Retrieval: Okapi BM25| E[Raw Relevance Scores]
    D -->|Contextual Proximity: Semantic Token Patterns| E
    E --> F[Stage 3: Behavioral Modifiers & Honeypot Filter]
    F -->|Assessment-First Skill Trust Matrix| G[Multiplier Score Tuning]
    F -->|Platform Active/Response Curve| G
    F -->|GitHub, Verified Contact Verification| G
    F -->|Anti-Honeypot Trap Matrix Penalty| G
    G --> H[Stage 4: Shortlisting & Deterministic Tie-Breaking]
    H -->|Alphanumeric CAND_ID Tie Break| I[Output Shortlist: Top 100 Candidates]
    I --> J[CSV Writer & Explainable Reasoning Generator]
    J --> K[Final Output: team_Jarvis.csv]
```

### Pipeline Details:

1. **Stage 1 (Boolean Hard Gates):** Instantly discards non-fits based on strict constraints (Years of Experience between 4 and 13, onsite preference or willingness to relocate to Pune/Noida, non-technical title exclusions, completeness score $\ge 50\%$, and blacklist checks).
2. **Stage 2 (Hybrid Scoring Engine):**
   * **Okapi BM25:** Computes lexical scores against target JD keywords using corpus-wide statistics.
   * **Contextual Proximity Matcher:** Awards high-gradient boosts if target skills are written in the same sentence as senior action verbs (*"built"*, *"scaled"*, *"optimized"*).
   * **Headline-aware Matching:** Pulls profile headlines into the text-match path so explicit stack keywords can influence BM25 and semantic scoring.
   * **Optional MiniLM Semantic Scoring:** Blends CPU sentence-transformer cosine similarity against the JD anchor when a local model is prepared.
3. **Stage 3 (Behavioral Modifiers & Honeypot Check):**
   * Applies activity decay (sigmoid centred on p50 inactivity = 88 days), recruiter response rates, recruiter response time, intent signals, recruiter-market validation, company-scale progression, and assessment-first skill trust blended with experience and endorsements.
   * All multiplier thresholds and coefficients are empirically calibrated — see `docs/SIGNAL_CALIBRATION.md` and `signal_calibration.py`.
   * GitHub activity uses a minimal ±3% band (not a linear boost): dataset analysis shows r(ICR, GitHub) = +0.06, so it is treated as a code-artifact nudge, not an outcome predictor.
   * Contact verification penalises **both** unverified channels only (6.8% of pool); an OR-penalty would hit 46% and was rejected.
   * Defensively catches honeypots (Expert skills with 0 months experience) and applies a severe `0.001` multiplier, completely dropping them from the shortlisted ranks.
4. **Stage 4 (Shortlist & Explainability):**
   * Sorts candidates and breaks ties deterministically using alphanumeric candidate IDs.
   * Generates factual, hallucination-free, candidate-specific reasoning notes based on resume metadata.

---

## Video Demo Recording Guide

To create your screencast for the submission checklist:

### 1. Preparation
* Open your terminal or command prompt.
* Open your repository folder containing the code.
* Open the **Google Colab Sandbox** in a browser tab.

### 2. What to Record (Duration: ~60–90 seconds)
1. **Show the Sandbox (15s):** Start the recording showing the Colab sandbox badge. Scroll down to show the notebook structure.
2. **Run the Code (30s):** Highlight Step 2 and Step 3. Click **Run All** or run the cells in sequence:
   * Run the setup to clone/load the ranker.
   * Run `python rank_candidates.py --candidates data/sample_candidates.jsonl --out sample_submission.csv`. Show the command outputting the audit metrics in the console.
   * Run `python validate_submission.py sample_submission.csv` to show the validator outputting **`Submission is valid.`**
3. **Show Explainability (20s):** Scroll to Step 4 and show the pandas table rendering the ranked shortlist. Highlight the `reasoning` column containing candidate facts (e.g. Swiggy experience, recruiter response rates, notice periods) to demonstrate **explainable rankings**.
4. **End the Video (10s):** Show the `submission_metadata.yaml` file loaded, showing the team configuration.
