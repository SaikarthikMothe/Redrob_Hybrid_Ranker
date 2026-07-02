import csv, sys, re
from collections import Counter

REQUIRED_COLS = {"rank", "candidate_id", "reasoning"}
CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "team_Jarvis.csv"

errors   = []
warnings = []

with open(CSV_FILE, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# ── 1. Column check ────────────────────────────────────────────────────────────
cols = set(rows[0].keys()) if rows else set()
missing = REQUIRED_COLS - cols
if missing:
    errors.append(f"Missing required columns: {missing}")
else:
    print(f"[OK] Required columns present: {REQUIRED_COLS}")

# ── 2. Row count ───────────────────────────────────────────────────────────────
if len(rows) != 100:
    errors.append(f"Expected 100 rows, got {len(rows)}")
else:
    print(f"[OK] Exactly 100 rows present")

# ── 3. Rank integrity ──────────────────────────────────────────────────────────
ranks = [int(r["rank"]) for r in rows]
if sorted(ranks) != list(range(1, 101)):
    errors.append(f"Ranks are not a clean 1-100 sequence: {sorted(ranks)[:10]}...")
else:
    print(f"[OK] Ranks are a clean 1-100 sequence")

# ── 3b. Score integrity ────────────────────────────────────────────────────────
invalid_scores = []
for r in rows:
    try:
        s = float(r["score"])
        if not 0.0 <= s <= 1.0:
            invalid_scores.append((r["candidate_id"], s))
    except ValueError:
        invalid_scores.append((r["candidate_id"], r.get("score")))
if invalid_scores:
    errors.append(f"Invalid scores found (outside [0.0, 1.0] range): {invalid_scores}")
else:
    print(f"[OK] All scores are valid and bounded between 0.0 and 1.0")

# ── 4. Duplicate candidate_ids ─────────────────────────────────────────────────
cids = [r["candidate_id"] for r in rows]
dupes = [cid for cid, cnt in Counter(cids).items() if cnt > 1]
if dupes:
    errors.append(f"Duplicate candidate_ids: {dupes}")
else:
    print(f"[OK] All candidate_ids are unique")

# ── 5. Empty reasoning cells ───────────────────────────────────────────────────
empty_rsn = [r["rank"] for r in rows if not r.get("reasoning","").strip()]
if empty_rsn:
    errors.append(f"Empty reasoning at ranks: {empty_rsn}")
else:
    print(f"[OK] All reasoning cells are populated")

# ── 6. Reasoning length check (too short = template skeleton, too long = noise) ─
short = [r["rank"] for r in rows if len(r.get("reasoning","")) < 80]
long  = [r["rank"] for r in rows if len(r.get("reasoning","")) > 1800]
if short:
    warnings.append(f"Very short reasoning (<80 chars) at ranks: {short}")
else:
    print(f"[OK] All reasoning entries are sufficiently detailed")
if long:
    warnings.append(f"Very long reasoning (>1800 chars) at ranks: {long[:5]}... ({len(long)} total)")

# ── 7. Structural repetition — detect identical sentence openers ───────────────
openers = [r["reasoning"][:40] for r in rows]
dup_openers = [o for o, cnt in Counter(openers).items() if cnt > 3]
if dup_openers:
    warnings.append(f"Repeated reasoning openers (>3 times): {dup_openers}")
else:
    print(f"[OK] No structural opener repetition detected")

# ── 8. Honeypot check ──────────────────────────────────────────────────────────
hp_kw = ["honeypot", "INJECTED", "IGNORE_RANKING", "BOOST_SCORE"]
hp_hits = [r["candidate_id"] for r in rows for kw in hp_kw if kw.lower() in r.get("reasoning","").lower()]
if hp_hits:
    errors.append(f"Honeypot keywords found in reasoning: {hp_hits}")
else:
    print(f"[OK] No honeypot content detected in reasoning")

# ── 9. Response rate descriptor variety ───────────────────────────────────────
descriptors = []
pattern = r'(exceptional|flawless|near-perfect|highly responsive|excellent|reliable|consistently active|solid|steady|generally consistent|respectable|moderate|fair|developing|inconsistent|below-average|limited|poor|weak|low)[^.]*(?:response rate|recruiter communication|platform engagement)[^.]*'
for r in rows:
    m = re.search(pattern, r["reasoning"], re.IGNORECASE)
    if m:
        descriptors.append(m.group(0)[:60])
unique_desc = len(set(d.split()[0].lower() for d in descriptors))
print(f"[OK] Response rate descriptor variety: {unique_desc} distinct lead words across {len(descriptors)} entries")

# ── 10. Sample reasoning dump (first 5) ───────────────────────────────────────
print("\n-- Sample Reasoning (Ranks 1-5) ------------------------------------------")
for r in rows[:5]:
    print(f"\n  Rank {r['rank']} [{r['candidate_id']}]:")
    print(f"  {r['reasoning']}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n==================================================")
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors: print(f"  [FAIL] {e}")
else:
    print("  [PASS] ZERO ERRORS")
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings: print(f"  [WARN] {w}")
else:
    print("  [PASS] ZERO WARNINGS")
print("==================================================")
print("SUBMISSION STATUS:", "READY" if not errors else "BLOCKED")
