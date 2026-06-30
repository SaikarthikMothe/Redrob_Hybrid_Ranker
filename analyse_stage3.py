"""
Analyse Stage 3 signal distributions to validate proposed modifications.
"""
import json, math

candidates = {}
with open('data/candidates.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        candidates[c['candidate_id']] = c

# Load stage 1 survivors (same logic as rank_candidates.py)
import sys
sys.path.insert(0, '.')
from rank_candidates import evaluate_stage1_boolean
survivors = [c for c in candidates.values() if evaluate_stage1_boolean(c)]
print(f"Stage 1 survivors: {len(survivors)}")

# --- Notice period distribution ---
notice_vals = [c.get('redrob_signals', {}).get('notice_period_days', 0) or 0 for c in survivors]
notice_vals.sort()
n = len(notice_vals)
print("\n=== NOTICE PERIOD ===")
for band, label in [(30, '<=30d'), (60, '31-60d'), (90, '61-90d'), (120, '91-120d'), (999, '>120d')]:
    count = sum(1 for v in notice_vals if v <= band) - sum(1 for v in notice_vals if v <= (band - (30 if band <= 90 else (30 if band==120 else 0))))
print(f"  <=30d  : {sum(1 for v in notice_vals if v <= 30)} ({sum(1 for v in notice_vals if v <= 30)/n*100:.1f}%)")
print(f"  31-60d : {sum(1 for v in notice_vals if 30 < v <= 60)} ({sum(1 for v in notice_vals if 30 < v <= 60)/n*100:.1f}%)")
print(f"  61-90d : {sum(1 for v in notice_vals if 60 < v <= 90)} ({sum(1 for v in notice_vals if 60 < v <= 90)/n*100:.1f}%)")
print(f"  91-120d: {sum(1 for v in notice_vals if 90 < v <= 120)} ({sum(1 for v in notice_vals if 90 < v <= 120)/n*100:.1f}%)")
print(f"  >120d  : {sum(1 for v in notice_vals if v > 120)} ({sum(1 for v in notice_vals if v > 120)/n*100:.1f}%)")

# --- Last active distribution ---
from datetime import datetime
REF = datetime.strptime("2026-06-16", "%Y-%m-%d")
def delta(c):
    s = c.get('redrob_signals', {}).get('last_active_date', '2026-06-16')
    try:
        return max(0, (REF - datetime.strptime(s, "%Y-%m-%d")).days)
    except:
        return 100

deltas = [delta(c) for c in survivors]
print("\n=== LAST ACTIVE (days before 2026-06-16) ===")
print(f"  <=30d  : {sum(1 for v in deltas if v <= 30)} ({sum(1 for v in deltas if v <= 30)/n*100:.1f}%)")
print(f"  31-90d : {sum(1 for v in deltas if 30 < v <= 90)} ({sum(1 for v in deltas if 30 < v <= 90)/n*100:.1f}%)")
print(f"  91-180d: {sum(1 for v in deltas if 90 < v <= 180)} ({sum(1 for v in deltas if 90 < v <= 180)/n*100:.1f}%)")
print(f"  >180d  : {sum(1 for v in deltas if v > 180)} ({sum(1 for v in deltas if v > 180)/n*100:.1f}%)")

# --- Company type analysis (consulting vs product) ---
CONSULTING = ['tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
              'capgemini', 'hcl', 'hcltech', 'tech mahindra', 'mphasis', 'hexaware',
              'mindtree', 'birlasoft', 'ltimindtree', 'l&t infotech', 'mastech',
              'syntel', 'virtusa', 'zensar', 'persistent', 'cyient', 'kpit',
              'deloitte', 'pwc', 'ey ', 'kpmg', 'ibm global', 'niit tech']
def is_consulting(name):
    name_l = (name or '').lower()
    return any(f in name_l for f in CONSULTING)

def company_profile(c):
    history = c.get('career_history', [])
    all_co = [job.get('company', '') for job in history]
    consulting_count = sum(1 for co in all_co if is_consulting(co))
    total = max(1, len(all_co))
    return consulting_count, total

pure_consulting = sum(1 for c in survivors if company_profile(c)[0] == company_profile(c)[1] and company_profile(c)[0] > 0)
some_consulting = sum(1 for c in survivors if 0 < company_profile(c)[0] < company_profile(c)[1])
no_consulting   = sum(1 for c in survivors if company_profile(c)[0] == 0)
print("\n=== COMPANY TYPE ===")
print(f"  Pure consulting career : {pure_consulting} ({pure_consulting/n*100:.1f}%)")
print(f"  Mixed (some consulting): {some_consulting} ({some_consulting/n*100:.1f}%)")
print(f"  No consulting at all   : {no_consulting} ({no_consulting/n*100:.1f}%)")

# --- OAR distribution ---
oar_vals = [c.get('redrob_signals', {}).get('offer_acceptance_rate', -1) for c in survivors]
oar_measured = [v for v in oar_vals if v != -1]
print("\n=== OFFER ACCEPTANCE RATE ===")
print(f"  Measured : {len(oar_measured)} / {n}")
print(f"  <0.40    : {sum(1 for v in oar_measured if v < 0.40)} ({sum(1 for v in oar_measured if v < 0.40)/len(oar_measured)*100:.1f}% of measured)")
print(f"  <0.50    : {sum(1 for v in oar_measured if v < 0.50)} ({sum(1 for v in oar_measured if v < 0.50)/len(oar_measured)*100:.1f}% of measured)")

# --- Skill trust distribution ---
from signal_calibration import calculate_skill_trust
trust_vals = [calculate_skill_trust(c) for c in survivors]
print("\n=== SKILL TRUST ===")
trust_vals_s = sorted(trust_vals)
print(f"  min={min(trust_vals):.3f}, p25={trust_vals_s[n//4]:.3f}, p50={trust_vals_s[n//2]:.3f}, p75={trust_vals_s[3*n//4]:.3f}, max={max(trust_vals):.3f}")
print(f"  Current band maps this to [{0.85:.2f}, {1.20:.2f}] → after update [{0.80:.2f}, {1.25:.2f}]")

# --- Recent activity sigmoid effect ---
print("\n=== ACTIVITY SIGMOID AT KEY DELTAS ===")
tau, lam = 88.0, 0.08
for d in [0, 15, 30, 45, 60, 90, 120, 180]:
    sig = 1.0 / (1.0 + math.exp(lam * (d - tau)))
    print(f"  delta={d:3d}d → sigmoid={sig:.3f}")
