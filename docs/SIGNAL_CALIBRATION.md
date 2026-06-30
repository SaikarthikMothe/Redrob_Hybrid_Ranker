# Signal Calibration Report

Generated from `data\candidates.jsonl` — **679** candidates after Stage 1 gates.
Reference date: `2026-06-16`. Constants live in `signal_calibration.py`.

## Key finding: GitHub is not an outcome predictor

- Interview completion × GitHub activity: **r = +0.0586** (weak)
- High-RR candidates do not have higher GitHub scores on average.
- GitHub is therefore mapped to a minimal **[0.97, 1.03]** band
  (public code artifact nudge for an AI-engineering JD), not a ±15% linear boost.

## Correlation matrix

| Signal pair | r | n | Strength |
|---|---:|---:|---|
| Open to Work x Applications Submitted | +0.1159 | 679 | weak |
| Recruiter RR  x Open to Work | +0.0549 | 679 | weak |
| Recruiter RR  x Applications Submitted | -0.0056 | 679 | weak |
| Recruiter RR  x Profile Views | +0.0728 | 679 | weak |
| Recruiter RR  x Recruiter Saves | +0.1324 | 679 | weak |
| Recruiter RR  x Search Appearances | +0.1406 | 679 | weak |
| Recruiter RR  x Interview Completion Rate | +0.0686 | 679 | weak |
| Recruiter RR  x Offer Acceptance Rate | -0.0325 | 418 | weak |
| Recruiter RR  x GitHub Activity | +0.0772 | 364 | weak |
| Recruiter RR  x Avg Response Time | -0.1237 | 679 | weak |
| Interview CR  x GitHub Activity | +0.0586 | 364 | weak |
| Interview CR  x Avg Response Time | -0.0867 | 679 | weak |
| Interview CR  x Offer Acceptance Rate | +0.0223 | 418 | weak |
| Offer Accept  x GitHub Activity | +0.0678 | 232 | weak |

## Calibrated multiplier summary

| Signal | Threshold / range | Multiplier | Pool impact | Rationale |
|---|---|---:|---|---|
| Platform activity | τ=88d (p50 delta) | sigmoid | median ~0.50× | Inactivity decay centred on observed median |
| Recruiter RR | <30% | 0.35× | 20.5% | Hard floor for disengagement band |
| Recruiter RR | ≥30% | RR^0.8 | — | Exponent tuned vs mean RR=0.52 |
| Skill trust | raw ∈ [0.44, 0.95] | [0.85, 1.20] | 100% differentiated | Assessment-first blend with duration/endorsement confidence |
| Response time | measured ∈ [2.2h, 219.9h] | inverse [0.96, 1.04] | all measured | Faster recruiter replies get a small lift; slower replies taper down |
| Intent | open_to_work + applications | [0.98, 1.04] | 47.9% open_to_work; 17.4% open+10+ apps | Active job seekers get a small urgency lift |
| Market validation | views + saves + search | [0.97, 1.03] | all measured | Recruiter interest / discoverability get a light nudge |
| Company scale | current size + progression | [0.985, 1.015] | current rank [2, 8]; progression delta [-6, 6] | Larger employers and upward movement get a weak boost |
| Interview completion | <50% | 0.85× | 10.6% | Bottom decile operational nudge |
| Notice period | >120d | 0.92× | 6.2% | HR constraint, not quality — light touch |
| GitHub activity | [1.4, 92.4] | [0.97, 1.03] | ±3% max | Weak outcome correlation — minimal weight |
| Verification | both unverified | 0.88× | 6.8% | OR-penalty would hit 46% of pool |
| Offer acceptance | <40% | 0.95× | 29.4% of measured | Soft nudge; missing for 38% of pool |

- Assessment scores on JD-relevant skills can contribute up to a +20% boost, and roughly 10% of surviving candidates receive a non-zero assessment lift.

## GBDT Ranker & Pseudo-Label Circularity Limitations

While the GBDT LambdaMART ranker (`--use-learned-combiner`) integrates multi-signal features, it is important to note the following limitations:
1. **Pseudo-Label Circularity:** Since no human-labeled ground-truth relevance data was available, the LambdaMART model was trained using the calibrated heuristic scores as target labels (pseudo-labels). As a result, the model is optimizing an approximation of our heuristic scoring function rather than independent relevance. Initially, the model achieved a correlation of **0.9859** with the heuristic target, indicating near-perfect memorization of the heuristic logic.
2. **Cross-Encoder Feature Regularization:** Introducing the Cross-Encoder semantic score as a training feature and retraining the GBDT ranker reduced target memorization. Post-retraining, the correlation with the heuristic targets fell to **0.7290**, and the correlation with the Cross-Encoder score was **0.3729**. The Cross-Encoder acts as a helpful semantic regularizer but is not a dominant driver.
3. **Lexical Dominance:** Feature importance analysis shows that lexical matching (`bm25_norm`) remains the dominant splitter (gain: **225.62**), while the Cross-Encoder score has a high split count but low gain. Lexical overlap continues to govern the primary candidate retrieval structure, with semantic and behavioral signals acting as secondary filters.

## Relevance blend (Stage 2)

**BM25 0.23 / Cross-Encoder 0.50 / Co-occurrence 0.27**

Variance share across survivors: BM25 35.3%, cross-encoder 6.9%, co-occurrence 57.8%.
Cross-Encoder is overweighted because it is the direct JD cross-encoder match;
co-occurrence is downweighted to limit keyword+verb gaming.

## Reproduce

```bash
python analyze_signals.py
python rank_candidates.py
```
