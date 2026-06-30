"""
Empirically calibrated behavioral multiplier constants and scoring helpers.

All thresholds and coefficients are derived from analyze_signals.py run against
679 Stage-1 gate survivors in data/candidates.jsonl (reference date 2026-06-16).
Run `python analyze_signals.py` to reproduce the underlying statistics.
"""
from __future__ import annotations

import math
from datetime import datetime

REFERENCE_DATE = "2026-06-16"

# --- Platform activity (last_active_date delta days) ---
# Distribution: p25=50d, p50=88d, p75=127d, p90=160d across surviving pool.
# τ set to p50 so the median candidate receives ~0.50× activity multiplier.
ACTIVITY_SIGMOID_TAU = 88.0
ACTIVITY_SIGMOID_LAMBDA = 0.08
ACTIVITY_FLOOR = 0.15

# --- Recruiter response rate ---
# mean=0.51, std=0.21; <30% band = 20.5% of pool.
# Exponent reduced 0.8->0.7: softer curve penalises mid-range RR (50-70%) less.
RR_FLOOR_THRESHOLD = 0.30
RR_FLOOR_MULTIPLIER = 0.35
RR_EXPONENT = 0.80  # tightened back to 0.80 per user request for stronger behavioral logic

# --- Recruiter response time ---
# p25=44.9h, p50=88.5h, p75=136.9h, p90=173.7h across surviving pool.
# Faster replies get a small lift; slower replies taper down in a narrow band.
RESPONSE_TIME_OBS_MIN = 2.2
RESPONSE_TIME_OBS_MAX = 219.9
RESPONSE_TIME_MULT_MIN = 0.96
RESPONSE_TIME_MULT_MAX = 1.04

# --- Intent / market validation / employer-scale proxies ---
# Open-to-work and application volume boost urgency; recruiter interest adds a
# market-validation nudge; employer scale / progression is a weak proxy.
# Market validation band widened 2026-06-22: [0.97,1.03]->[0.96,1.05] to give
# recruiter saves/views more meaningful signal weight.
INTENT_MULT_MIN = 0.98
INTENT_MULT_MAX = 1.04
MARKET_VALIDATION_MULT_MIN = 0.96
MARKET_VALIDATION_MULT_MAX = 1.05
COMPANY_SIZE_MULT_MIN = 0.985
COMPANY_SIZE_MULT_MAX = 1.015

# --- Skill trust (assessment-first blend with experience/endorsement confidence) ---
# raw_trust in [0.401, 0.928] on 884 Stage-1 survivors (p50=0.649).
# Band widened 2026-06-22: [0.85,1.20]->[0.80,1.25] to reward high-assessment
# candidates more and penalise low-evidence candidates slightly more.
SKILL_TRUST_OBS_MIN = 0.44
SKILL_TRUST_OBS_MAX = 0.95
SKILL_TRUST_MULT_MIN = 0.80
SKILL_TRUST_MULT_MAX = 1.25
HONEYPOT_SKILL_PENALTY = 0.001
SKILL_EVIDENCE_DURATION_SCALE = 36.0
SKILL_EVIDENCE_ENDORSEMENT_SCALE = 12.0
SKILL_ASSESSMENT_WEIGHT = 0.85
SKILL_EVIDENCE_WEIGHT = 0.15

# --- Interview completion rate ---
# p25=0.58, min=0.40; threshold 0.50 flags bottom 10.6% (72/679).
# r(ICR, RR)=+0.069 weak — used as operational-integrity nudge, not quality proxy.
ICR_PENALTY_THRESHOLD = 0.50
ICR_PENALTY_MULTIPLIER = 0.85

# --- Notice period (two-tier, updated 2026-06-22) ---
# Distribution: <=30d=14.3%, 31-60d=21.3%, 61-90d=36.0%, 91-120d=22.9%, >120d=5.7%.
# JD says sub-30d preferred, can buy out 30d. 30+ in scope but bar gets higher.
# Tier 1 (91-120d): light penalty - high enough to flag but not eliminate.
# Tier 2 (>120d): stronger penalty - contractual constraint on hiring speed.
NOTICE_PENALTY_MED_DAYS = 90
NOTICE_PENALTY_MED_MULTIPLIER = 0.95
NOTICE_PENALTY_THRESHOLD_DAYS = 120   # kept as alias for backward compat
NOTICE_PENALTY_HIGH_DAYS = 120
NOTICE_PENALTY_HIGH_MULTIPLIER = 0.90  # was 0.92
NOTICE_PENALTY_MULTIPLIER = 0.90       # alias for rank_candidates.py compat

# --- GitHub activity score ---
# r(ICR, GH)=+0.059 (weak); high-RR candidates avg GH 30.5 vs low-RR 38.0.
# Band widened 2026-06-22: [0.97,1.03]->[0.95,1.05] to give more meaningful
# signal for an AI-engineering JD where open-source work is explicitly valued.
GITHUB_OBS_MIN = 1.4
GITHUB_OBS_MAX = 92.4
GITHUB_MULT_MIN = 0.95
GITHUB_MULT_MAX = 1.05

# --- Contact verification ---
# either-unverified = 46.1% (313/679) — OR-penalty at 0.30× would over-penalise.
# both-unverified = 6.8% (46/679); apply penalty only when both channels fail.
VERIFY_BOTH_UNVERIFIED_MULTIPLIER = 0.88

# --- Offer acceptance rate ---
# measured for 532/884; 27.6% below 0.40, 44.2% below 0.50.
# Penalty strengthened 2026-06-22: 0.95->0.92 (aligns with notice penalty tier 1).
# Low OAR signals unreliable candidate engagement - strong fit concern for a
# founding-team role where hiring speed and commitment matter.
OAR_PENALTY_THRESHOLD = 0.40
OAR_PENALTY_MULTIPLIER = 0.92

# --- Stage 2 relevance blend (cross-encoder mode) ---
# Weights updated 2026-06-22:
#   BM25 reduced (0.28->0.23): JD explicitly warns keyword-stuffers are a trap.
#   CrossEncoder increased (0.47->0.50): semantic fit is the dominant quality signal.
#   CoOccurrence increased (0.25->0.27): rewards real execution evidence over listing.
# Sum = 1.00 verified.
RELEVANCE_WEIGHT_BM25 = 0.23
RELEVANCE_WEIGHT_CROSSENC = 0.50
RELEVANCE_WEIGHT_CO = 0.27

# Lexical fallback when embeddings unavailable.
RELEVANCE_FALLBACK_BM25 = 0.48
RELEVANCE_FALLBACK_CO = 0.22
RELEVANCE_FALLBACK_BREADTH = 0.30

# --- Recent activity boost ---
# Only 10% of survivors (88/884) were active in the last 30 days.
# A small boost rewards genuinely job-hunting candidates without disrupting
# the sigmoid which already handles gradual activity decay.
RECENT_ACTIVITY_THRESHOLD_DAYS = 30
RECENT_ACTIVITY_BOOST = 1.04

# --- Company type multiplier ---
# JD explicitly excludes candidates who have ONLY worked at consulting firms
# (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their
# entire career. Product-company experience is a strong positive signal.
# 7.2% of survivors are pure consulting - penalty applies to this group.
# 49.7% are mixed - neutral per JD ("prior product experience = fine").
# Boost applies only when current employer is a recognisable product company.
PRODUCT_COMPANY_BOOST = 1.05
CONSULTING_ONLY_PENALTY = 0.90


def _normalize_to_band(value: float, obs_min: float, obs_max: float, mult_min: float, mult_max: float) -> float:
    if obs_max <= obs_min:
        return (mult_min + mult_max) / 2.0
    norm = max(0.0, min(1.0, (value - obs_min) / (obs_max - obs_min)))
    return mult_min + (mult_max - mult_min) * norm


def _normalize_inverse_band(value: float, obs_min: float, obs_max: float, mult_min: float, mult_max: float) -> float:
    if obs_max <= obs_min:
        return (mult_min + mult_max) / 2.0
    norm = max(0.0, min(1.0, (value - obs_min) / (obs_max - obs_min)))
    return mult_max - (mult_max - mult_min) * norm


def _normalize_log_band(value: float, obs_min: float, obs_max: float, mult_min: float, mult_max: float) -> float:
    if obs_max <= obs_min:
        return (mult_min + mult_max) / 2.0
    value_log = math.log1p(max(0.0, value))
    obs_min_log = math.log1p(max(0.0, obs_min))
    obs_max_log = math.log1p(max(0.0, obs_max))
    if obs_max_log <= obs_min_log:
        return (mult_min + mult_max) / 2.0
    norm = max(0.0, min(1.0, (value_log - obs_min_log) / (obs_max_log - obs_min_log)))
    return mult_min + (mult_max - mult_min) * norm


def _company_size_rank(size_value: str) -> int:
    size_map = {
        "1-10": 1,
        "11-50": 2,
        "51-200": 3,
        "201-500": 4,
        "501-1000": 5,
        "1001-5000": 6,
        "5001-10000": 7,
        "10001+": 8,
    }
    return size_map.get((size_value or "").strip(), 0)


def _intent_multiplier(signals) -> float:
    open_to_work = bool(signals.get("open_to_work_flag", False))
    applications = max(0, int(signals.get("applications_submitted_30d", 0) or 0))
    app_norm = min(1.0, math.log1p(applications) / math.log1p(15.0))
    intent_score = (0.7 if open_to_work else 0.0) + (0.3 * app_norm)
    return INTENT_MULT_MIN + (INTENT_MULT_MAX - INTENT_MULT_MIN) * intent_score


def _market_validation_multiplier(signals) -> float:
    views = max(0, int(signals.get("profile_views_received_30d", 0) or 0))
    saved = max(0, int(signals.get("saved_by_recruiters_30d", 0) or 0))
    search = max(0, int(signals.get("search_appearance_30d", 0) or 0))
    market_score = (
        0.35 * _normalize_log_band(views, 0, 331, 0.0, 1.0)
        + 0.45 * _normalize_log_band(saved, 0, 80, 0.0, 1.0)
        + 0.20 * _normalize_log_band(search, 0, 1417, 0.0, 1.0)
    )
    return MARKET_VALIDATION_MULT_MIN + (MARKET_VALIDATION_MULT_MAX - MARKET_VALIDATION_MULT_MIN) * market_score


def _company_scale_multiplier(candidate) -> float:
    profile = candidate.get("profile", {})
    current_rank = _company_size_rank(profile.get("current_company_size"))
    if not current_rank:
        current_rank = _company_size_rank(
            next(
                (
                    job.get("company_size")
                    for job in candidate.get("career_history", [])
                    if job.get("is_current")
                ),
                "",
            )
        )
    if not current_rank:
        return 1.0

    history_ranks = [
        _company_size_rank(job.get("company_size"))
        for job in candidate.get("career_history", [])
        if _company_size_rank(job.get("company_size"))
    ]
    if history_ranks:
        oldest_rank = history_ranks[-1]
        progression_delta = current_rank - oldest_rank
    else:
        progression_delta = 0

    current_norm = _normalize_to_band(current_rank, 1, 8, 0.0, 1.0)
    progression_norm = _normalize_to_band(progression_delta, -6, 6, 0.0, 1.0)
    combined = 0.55 * current_norm + 0.45 * progression_norm
    return COMPANY_SIZE_MULT_MIN + (COMPANY_SIZE_MULT_MAX - COMPANY_SIZE_MULT_MIN) * combined


# ---------------------------------------------------------------------------
# Company type detection (product vs consulting)
# JD explicitly penalises candidates who have ONLY worked at consulting firms.
# ---------------------------------------------------------------------------
_CONSULTING_FIRMS = frozenset([
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'hcltech', 'hcl tech', 'tech mahindra', 'mphasis',
    'hexaware', 'mindtree', 'birlasoft', 'ltimindtree', 'l&t infotech',
    'mastech', 'syntel', 'virtusa', 'zensar', 'persistent systems', 'cyient',
    'kpit', 'deloitte', 'pwc', 'ernst & young', 'kpmg', 'ibm global services',
    'niit technologies', 'sonata software', 'sasken', 'geometric', 'marianaai',
    'oracle financial services', 'firstsource', 'wns global', 'genpact',
])

_PRODUCT_COMPANIES = frozenset([
    # Global Tech Giants
    'google', 'microsoft', 'amazon', 'meta', 'apple', 'netflix', 'uber', 'linkedin',
    'stripe', 'salesforce', 'adobe', 'atlassian', 'databricks', 'snowflake', 'mongodb',
    'nvidia', 'samsung', 'spotify', 'airbnb', 'paypal',
    # AI / Vector / Infra Innovators
    'openai', 'anthropic', 'hugging face', 'pinecone', 'weaviate', 'qdrant', 'elastic',
    # Indian Tech Leaders / Unicorns
    'flipkart', 'swiggy', 'zomato', 'cred', 'razorpay', 'phonepe', 'paytm', 'ola',
    'zoho', 'freshworks', 'postman', 'browserstack', 'groww', 'zerodha', 'delhivery',
    'sharechat', 'meesho'
])


def _is_consulting(company_name: str) -> bool:
    name_l = (company_name or '').lower().strip()
    return any(firm in name_l for firm in _CONSULTING_FIRMS)


def _is_product(company_name: str) -> bool:
    name_l = (company_name or '').lower().strip()
    return any(firm in name_l for firm in _PRODUCT_COMPANIES)


def _company_type_multiplier(candidate) -> float:
    """
    Returns a multiplier based on the candidate's company-type profile:
      - Pure consulting career only (7.2% of pool): CONSULTING_ONLY_PENALTY
      - Currently at product company: PRODUCT_COMPANY_BOOST
      - Mixed or unknown: 1.0 (neutral, per JD: 'prior product experience = fine')
    """
    career = candidate.get('career_history', [])
    if not career:
        return 1.0

    all_companies = [job.get('company', '') for job in career if job.get('company')]
    if not all_companies:
        return 1.0

    consulting_count = sum(1 for co in all_companies if _is_consulting(co))
    product_count    = sum(1 for co in all_companies if _is_product(co))
    total            = len(all_companies)

    # Pure consulting: every employer is a consulting firm, zero product exposure
    if consulting_count == total and product_count == 0:
        return CONSULTING_ONLY_PENALTY

    # Current employer is a known product company
    current_company = all_companies[0]  # career_history[0] = most recent
    if _is_product(current_company):
        return PRODUCT_COMPANY_BOOST

    # Mixed history or unknown companies: neutral
    return 1.0


def _skill_evidence_confidence(duration_months: float, endorsements: float) -> float:
    duration_norm = min(1.0, math.log1p(max(0.0, duration_months)) / math.log1p(SKILL_EVIDENCE_DURATION_SCALE))
    endorsement_norm = min(1.0, math.log1p(max(0.0, endorsements) + 1.0) / math.log1p(SKILL_EVIDENCE_ENDORSEMENT_SCALE))
    return duration_norm * endorsement_norm


def calculate_skill_trust(candidate) -> float:
    skills = candidate.get("skills", [])
    if not skills:
        return 1.0

    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    normalized_assessments = {
        str(skill_name).strip().lower(): max(0.0, min(1.0, float(score) / 100.0))
        for skill_name, score in assessment_scores.items()
    }

    skill_accumulator = 0.0
    honeypot_penalty_multiplier = 1.0
    for skill in skills:
        duration = max(0, skill.get("duration_months", 0) or 0)
        endorsements = max(0, skill.get("endorsements", 0) or 0)
        proficiency = (skill.get("proficiency", "") or "").lower()
        if proficiency == "expert" and duration == 0:
            honeypot_penalty_multiplier *= HONEYPOT_SKILL_PENALTY

        evidence_confidence = _skill_evidence_confidence(duration, endorsements)
        skill_name = str(skill.get("name", "") or "").strip().lower()
        assessment = normalized_assessments.get(skill_name)

        if assessment is None:
            skill_score = evidence_confidence
        else:
            skill_score = assessment * (SKILL_ASSESSMENT_WEIGHT + (SKILL_EVIDENCE_WEIGHT * evidence_confidence))

        skill_accumulator += skill_score

    return (skill_accumulator / len(skills)) * honeypot_penalty_multiplier


def calculate_advanced_multipliers(candidate, reference_date: str = REFERENCE_DATE) -> float:
    signals = candidate.get("redrob_signals", {})
    multiplier = 1.0

    # --- Activity decay (sigmoid) ---
    last_act_str = signals.get("last_active_date", reference_date)
    try:
        delta_days = max(
            0,
            (
                datetime.strptime(reference_date, "%Y-%m-%d")
                - datetime.strptime(last_act_str, "%Y-%m-%d")
            ).days,
        )
    except Exception:
        delta_days = 100
    sigmoid_activity = 1.0 / (1.0 + math.exp(ACTIVITY_SIGMOID_LAMBDA * (delta_days - ACTIVITY_SIGMOID_TAU)))
    multiplier *= max(ACTIVITY_FLOOR, sigmoid_activity)

    # --- Recruiter response rate ---
    response_rate = signals.get("recruiter_response_rate", 1.0)
    if response_rate < RR_FLOOR_THRESHOLD:
        multiplier *= RR_FLOOR_MULTIPLIER
    else:
        multiplier *= response_rate ** RR_EXPONENT

    # --- Skill trust calculation first (for proxy bias mitigation) ---
    # If a candidate has exceptionally high direct skills evidence, we soften
    # demographic/pedigree proxy penalties to protect high-performing profiles from systemic bias.
    raw_trust = calculate_skill_trust(candidate)
    is_mitigated = raw_trust > 0.72

    # --- Company type multiplier (Product boost & Consulting penalty) ---
    co_type_mult = _company_type_multiplier(candidate)
    if is_mitigated and co_type_mult == CONSULTING_ONLY_PENALTY:
        multiplier *= 0.96  # softened from 0.90 to mitigate pedigree bias
    else:
        multiplier *= co_type_mult

    # --- Apply Skill trust multiplier ---
    if raw_trust < 0.01:
        skill_multiplier = raw_trust
    else:
        skill_multiplier = _normalize_to_band(
            raw_trust,
            SKILL_TRUST_OBS_MIN,
            SKILL_TRUST_OBS_MAX,
            SKILL_TRUST_MULT_MIN,
            SKILL_TRUST_MULT_MAX,
        )
    multiplier *= skill_multiplier

    if signals.get("interview_completion_rate", 1.0) < ICR_PENALTY_THRESHOLD:
        multiplier *= ICR_PENALTY_MULTIPLIER

    # --- Notice period (two-tier with bias mitigation) ---
    notice_days = signals.get("notice_period_days", 0)
    if notice_days > NOTICE_PENALTY_HIGH_DAYS:
        # >120d: normal penalty 0.90, mitigated to 0.95
        multiplier *= 0.95 if is_mitigated else NOTICE_PENALTY_HIGH_MULTIPLIER
    elif notice_days > NOTICE_PENALTY_MED_DAYS:
        # 91-120d: normal penalty 0.95, mitigated to 0.975
        multiplier *= 0.975 if is_mitigated else NOTICE_PENALTY_MED_MULTIPLIER

    # --- GitHub (with bias mitigation) ---
    gh_score = signals.get("github_activity_score", -1)
    if gh_score != -1:
        gh_mult = _normalize_to_band(
            gh_score,
            GITHUB_OBS_MIN,
            GITHUB_OBS_MAX,
            GITHUB_MULT_MIN,
            GITHUB_MULT_MAX,
        )
        if is_mitigated:
            # Soften penalty floor for candidates who don't have public GitHub contributions but are highly skilled
            multiplier *= max(0.98, gh_mult)
        else:
            multiplier *= gh_mult

    email_verified = signals.get("verified_email", True)
    phone_verified = signals.get("verified_phone", True)
    if email_verified is False and phone_verified is False:
        multiplier *= VERIFY_BOTH_UNVERIFIED_MULTIPLIER

    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate != -1 and offer_rate < OAR_PENALTY_THRESHOLD:
        multiplier *= OAR_PENALTY_MULTIPLIER

    return multiplier


def blend_relevance(bm25_norm: float, cross_enc: float, co_norm: float) -> float:
    return (
        RELEVANCE_WEIGHT_BM25 * bm25_norm
        + RELEVANCE_WEIGHT_CROSSENC * cross_enc
        + RELEVANCE_WEIGHT_CO * co_norm
    )


def blend_relevance_fallback(bm25_norm: float, co_norm: float, breadth_modifier: float) -> float:
    return (
        RELEVANCE_FALLBACK_BM25 * bm25_norm
        + RELEVANCE_FALLBACK_CO * co_norm
        + RELEVANCE_FALLBACK_BREADTH * breadth_modifier
    )
