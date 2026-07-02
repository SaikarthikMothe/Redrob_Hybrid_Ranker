import os
import gzip
import json
import csv
import sys
from pathlib import Path

from rank_candidates import (
    evaluate_stage1_boolean,
    open_candidate_stream
)
from signal_calibration import (
    _company_type_multiplier,
    CONSULTING_ONLY_PENALTY
)

SUBMISSION_PATH = "team_Jarvis2.0.csv"
CANDIDATES_PATH = os.path.join("data", "candidates.jsonl")

def run_audit():
    if not os.path.exists(SUBMISSION_PATH):
        print(f"Error: Submission file {SUBMISSION_PATH} not found. Run ranking pipeline first.")
        return 1
        
    if not os.path.exists(CANDIDATES_PATH):
        print(f"Error: Candidate file {CANDIDATES_PATH} not found.")
        return 1

    # Load top 100 shortlisted candidate IDs
    shortlisted = set()
    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shortlisted.add(row["candidate_id"])

    if len(shortlisted) != 100:
        print(f"Warning: Submission file should contain exactly 100 candidates, but has {len(shortlisted)}.")

    # Load candidates and filter Stage 1 survivors
    survivors = []
    with open_candidate_stream(CANDIDATES_PATH) as f:
        header_chars = f.read(100).strip()
        
    with open_candidate_stream(CANDIDATES_PATH) as f:
        if header_chars.startswith('['):
            records = json.load(f)
        else:
            records = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    for r in records:
        if evaluate_stage1_boolean(r):
            survivors.append(r)

    print(f"Auditing fairness on {len(survivors)} Stage 1 survivors vs. top 100 shortlist...\n")

    # Define groups
    groups = {
        "Consulting-Only Career": {
            "check": lambda c: _company_type_multiplier(c) == CONSULTING_ONLY_PENALTY,
            "disadvantaged_label": "Consulting-only history (Penalised)",
            "advantaged_label": "Product or mixed history (Privileged)"
        },
        "Notice Period >90d": {
            "check": lambda c: (c.get("redrob_signals", {}) or {}).get("notice_period_days", 0) > 90,
            "disadvantaged_label": "Notice period > 90 days (Penalised)",
            "advantaged_label": "Notice period <= 90 days (Privileged)"
        },
        "Low/No GitHub Activity": {
            "check": lambda c: (c.get("redrob_signals", {}) or {}).get("github_activity_score", -1) < 25,
            "disadvantaged_label": "GitHub activity score < 25 (Penalised)",
            "advantaged_label": "GitHub activity score >= 25 (Privileged)"
        },
        "Relocating Candidates": {
            "check": lambda c: "pune" not in ((c.get("profile", {}) or {}).get("location", "") or "").lower() and "noida" not in ((c.get("profile", {}) or {}).get("location", "") or "").lower(),
            "disadvantaged_label": "Must relocate to Pune/Noida (Penalised)",
            "advantaged_label": "Resident of Pune/Noida (Privileged)"
        }
    }

    print(f"{'Audit Metric / Group':<30} | {'Disadvantaged Rate':<22} | {'Privileged Rate':<20} | {'Impact Ratio':<12} | {'Status':<6}")
    print("-" * 100)

    for metric_name, conf in groups.items():
        check_fn = conf["check"]
        
        disadv_total = 0
        disadv_selected = 0
        adv_total = 0
        adv_selected = 0
        
        for c in survivors:
            cid = c.get("candidate_id")
            in_shortlist = cid in shortlisted
            
            if check_fn(c):
                disadv_total += 1
                if in_shortlist:
                    disadv_selected += 1
            else:
                adv_total += 1
                if in_shortlist:
                    adv_selected += 1

        disadv_rate = disadv_selected / disadv_total if disadv_total > 0 else 0.0
        adv_rate = adv_selected / adv_total if adv_total > 0 else 0.0
        
        if adv_rate > 0:
            ratio = disadv_rate / adv_rate
        else:
            ratio = 1.0 if disadv_rate == 0 else float('inf')

        # Status checks
        if ratio < 0.80:
            status = "FAIL"
        else:
            status = "PASS"

        ratio_str = f"{ratio:.3f}" if ratio != float('inf') else "N/A"
        rate_dis_str = f"{disadv_selected}/{disadv_total} ({disadv_rate*100:.1f}%)"
        rate_adv_str = f"{adv_selected}/{adv_total} ({adv_rate*100:.1f}%)"
        
        print(f"{metric_name:<30} | {rate_dis_str:<22} | {rate_adv_str:<20} | {ratio_str:<12} | {status:<6}")

    print("\nFour-Fifths Rule (80% rule): Impact ratio should be >= 0.80 to satisfy basic disparate impact requirements.")
    return 0

if __name__ == "__main__":
    sys.exit(run_audit())
