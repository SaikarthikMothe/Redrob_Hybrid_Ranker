# Redrob Hybrid Ranker — Full Run Process

**Team:** `team_204`  
**Last verified run:** 2026-06-17  
**Environment:** Windows 10, Python 3.x, CPU-only (no GPU, no network calls at runtime)

---

## 1. Purpose

This document records the end-to-end procedure to reproduce the submission from a clean checkout. A full run verifies prerequisites, executes the ranking pipeline on the official 100,000-candidate dataset, validates output, audits reasoning quality, and optionally runs the lightweight sample demo path.

---

## 2. Project Layout

| Path | Role |
|---|---|
| `data/candidates.jsonl` | Official dataset (100,000 candidates) |
| `data/sample_candidates.jsonl` | Demo dataset (2,969 candidates → exactly 100 gate survivors) |
| `data/semantic_scores.json.gz` | Precomputed MiniLM cosine scores (679 survivors) |
| `models/all-MiniLM-L6-v2/` | Local SentenceTransformers model (384-dim) |
| `rank_candidates.py` | Main ranking pipeline |
| `signal_calibration.py` | Empirically tuned behavioral multipliers |
| `validate_submission.py` | Official CSV format validator |
| `audit_submission.py` | Internal reasoning-quality audit |
| `verify_setup.py` | Prerequisite checker |
| `analyze_signals.py` | Signal distribution / correlation reproduction |
| `team_204.csv` | Final submission output |

---

## 3. Prerequisites

Install dependencies (one-time):

```bash
pip install -r requirements.txt
```

If the embedding model or precomputed scores are missing:

```bash
python prepare_embedding_model.py --out models/all-MiniLM-L6-v2
python generate_precomputed_scores.py
```

---

## 4. Full Run Procedure

### Step 1 — Verify Setup

```bash
python verify_setup.py
```

**Expected output:**

```
Redrob Hybrid Ranker setup verification
==================================================
  [OK] full candidate dataset: 100000 records, 679 Stage 1 survivors
  [OK] sample candidate dataset: 2969 records, 100 Stage 1 survivors
  [OK] Embedding model loads successfully (dimension=384).
  [OK] Precomputed semantic scores present (679 entries).
  [OK] Submission file present: team_204.csv

All technical prerequisites are satisfied.
```

**Exit code:** `0`

---

### Step 2 — Run Ranking Pipeline (Official Dataset)

```bash
python rank_candidates.py --out team_204.csv
```

This is the **default semantic run**. It loads precomputed scores from `data/semantic_scores.json.gz` and completes in ~30 seconds on CPU.

**Pipeline stages (console output):**

| Phase | Console label | What happens |
|---|---|---|
| 1/4 | Ingesting + Stage 1 gates | Stream 100k candidates; 679 survive boolean filters |
| 2/4 | Consolidating scores | BM25 + contextual co-occurrence + semantic blend + behavioral multipliers |
| 3/4 | Sorting + tie-break | Descending score; alphanumeric `CAND_ID` tie-break |
| 4/4 | CSV writer | Write top 100 with comparative + factual reasoning |

**Expected audit block:**

```
====================================================================
                      PIPELINE AUDIT METRICS
====================================================================
 Total Shortlisted Candidates Exported : 100
 Flagged Honeypot Traps Contained      : 0 / 100
  STATUS: Safety threshold verified. Sandbox validation PASSED.
====================================================================
SUCCESS: Pipeline iteration completed cleanly. Data target established at: team_204.csv
```

**Exit code:** `0`  
**Runtime observed:** ~31 s

---

### Step 3 — Official Format Validation

```bash
python validate_submission.py team_204.csv
```

**Expected output:** `Submission is valid.`  
**Exit code:** `0`

Checks enforced:
- Header exactly: `candidate_id,rank,score,reasoning`
- Exactly 100 data rows (ranks 1–100)
- Unique candidate IDs matching `CAND_XXXXXXX`
- No empty reasoning cells

---

### Step 4 — Internal Submission Audit

```bash
python audit_submission.py
```

**Expected results (2026-06-17 run):**

| Check | Status |
|---|---|
| Required columns | PASS |
| 100 rows, ranks 1–100 | PASS |
| Unique candidate IDs | PASS |
| All reasoning populated | PASS |
| Opener repetition (>3×) | PASS — 100 unique openers |
| Honeypot keywords in reasoning | PASS |
| Response-rate descriptor variety | PASS — 15 distinct lead words |

**Reasoning length:** Submission Specification v4 defines no character cap. The current
submission ranges from 920 to 1,302 characters (1,133 average). The official validator
does not reject reasoning by character count; the specification's relevant guidance is
to provide specific, factual, varied justification, with 1–2 sentences recommended.

**Final status:** `SUBMISSION STATUS: READY`

---

### Step 5 — Reasoning Opener Diversity Check

```bash
python check_openers.py
```

**Expected:** 100 rows, 100 unique openers (each comparative clause is distinct).

---

### Step 6 — Sample Dataset Demo Run (Optional)

For Colab/sandbox demos with a fast dataset:

```bash
python rank_candidates.py --candidates data/sample_candidates.jsonl --out sample_submission.csv
python validate_submission.py sample_submission.csv
```

**Expected:** 100 gate survivors in, 100 ranked rows out, `Submission is valid.`

---

### Step 7 — Reproduce Signal Calibration (Optional)

```bash
python analyze_signals.py
```

Recomputes distributions and Pearson correlations on the 679-candidate surviving pool. Used to justify constants in `signal_calibration.py` and `docs/SIGNAL_CALIBRATION.md`.

Key findings reproduced:
- Recruiter response rate: mean 0.51, std 0.21
- r(RR, GitHub) = +0.08 (weak) → GitHub limited to ±3% band
- r(RR, ICR) = +0.07 (weak) → ICR used as integrity nudge only
- Both-unverified contact: 6.8% of pool → 0.88× penalty (OR-penalty rejected)

---

## 5. Pipeline Architecture

```
100,000 candidates (candidates.jsonl)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Boolean Hard Gates │  YOE 4–13, location/relocation, work mode,
│  679 survivors (~0.7%)       │  title filter, completeness ≥50%, blacklist
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 2: Hybrid Relevance   │  BM25 (23%) + Cross-Encoder (50%) + Co-occurrence (27%)
│  Corpus-wide Okapi BM25      │  Contextual verb+keyword proximity boosts
│  Precomputed MiniLM scores   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 3: Behavioral Mult.   │  Activity sigmoid, response rate, response time,
│  Honeypot trap filter        │  intent, market validation, company scale, assessment-first skill trust,
│                               │  notice period, GitHub ±3%, contact verify, OAR
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 4: Sort + Explain     │  final_score = relevance × behavioral_multiplier
│  Top 100 + tie-break by ID   │  Comparative reasoning vs. immediate neighbor
└─────────────────────────────┘
        │
        ▼
   team_204.csv
```

---

## 6. Reasoning Format (Post-Comparative Update)

Each `reasoning` cell has two parts:

1. **Comparative opener** — recruiter-style tradeoff vs. immediate neighbor  
   - Rank 1: `Leads the shortlist ahead of CAND_XXXXXXX due to … despite …`
   - Ranks 2–99: `Ranked above CAND_XXXXXXX due to … despite …`
   - Close calls (Δscore < 0.01): `Edges out CAND_XXXXXXX in a close ranking call …`
   - Rank 100: `Ranked below CAND_XXXXXXX primarily due to … though still brings …`

2. **Factual narrative** — career arc, JD skills, response rate, notice period, GitHub (hallucination-free, profile-sourced)

**Example (Rank 1, verified 2026-06-17):**

> Leads the shortlist ahead of CAND_0005260 due to stronger Elasticsearch hybrid-search experience and Qdrant retrieval experience, despite less explicit RAG-specific experience. Based on their career arc from LinkedIn to Genpact AI, this Senior Machine Learning Engineer has accumulated meaningful industry exposure. …

---

## 7. Alternative Run Modes

| Command | Mode | Use case |
|---|---|---|
| `python rank_candidates.py` | Default semantic (precomputed) | **Production submission** |
| `python rank_candidates.py --no-embeddings` | Lexical-only (BM25 + co-occurrence) | Ablation / faster dev iteration |
| `python rank_candidates.py --embedding-model models/all-MiniLM-L6-v2` | Live embedding fallback | Custom/unseen datasets |
| `python generate_precomputed_scores.py` | Regenerate semantic cache | After model or JD anchor change |

---

## 8. Full Run Checklist

| # | Step | Command | Pass criterion |
|---|---|---|---|
| 1 | Verify prerequisites | `python verify_setup.py` | Exit 0, all `[OK]` |
| 2 | Generate submission | `python rank_candidates.py --out team_204.csv` | 100 rows, 0 honeypots in shortlist |
| 3 | Official validation | `python validate_submission.py team_204.csv` | `Submission is valid.` |
| 4 | Internal audit | `python audit_submission.py` | `SUBMISSION STATUS: READY` |
| 5 | Opener diversity | `python check_openers.py` | 100 unique openers |
| 6 | Sample demo (optional) | `python rank_candidates.py --candidates data/sample_candidates.jsonl --out sample_submission.csv` | 100 in → 100 out |
| 7 | Signal reproduction (optional) | `python analyze_signals.py` | Distributions print without error |

---

## 9. Latest Full-Run Results Summary

| Metric | Value |
|---|---|
| Input candidates | 100,000 |
| Stage 1 survivors | 679 |
| Shortlist exported | 100 |
| Honeypots in shortlist | 0 |
| Top candidate | `CAND_0046525` (score 0.75940591) |
| Semantic scores loaded | 679 precomputed |
| Official validator | PASS |
| Internal audit | READY (0 errors, 0 warnings) |
| Unique reasoning openers | 100 / 100 |
| Total pipeline runtime | ~31 s |

---

## 10. Submission Artifacts

After a successful full run, these files are ready for portal upload:

1. `team_204.csv` — ranked shortlist with reasoning
2. `submission_metadata.yaml` — team metadata and methodology summary
3. `docs/SIGNAL_CALIBRATION.md` — judge-facing calibration report (reference only)

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing model directory` | `python prepare_embedding_model.py --out models/all-MiniLM-L6-v2` |
| `Missing semantic_scores.json.gz` | `python generate_precomputed_scores.py` |
| `Only N candidates passed Stage 1 gates` | Check `data/candidates.jsonl` integrity |
| Sample dataset ≠ 100 survivors | `python build_sample_dataset.py` |
| Validator rejects header | Ensure CSV header is exactly `candidate_id,rank,score,reasoning` |

---

*Generated from a live full run on 2026-06-17. Re-run the commands in Section 4 to reproduce.*
