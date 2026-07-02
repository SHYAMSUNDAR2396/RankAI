"""Per-candidate feature extraction for the Redrob candidate-ranking pipeline.

Extracts everything we need to score 100K candidates deterministically from
the competition JSONL schema. All features are pure-Python (no LLM calls,
no embedding model) so the live ranking step fits inside the 5-min/16GB/CPU
budget.

The features are grouped into six dimensions, each returned as a 0–1 score
(or a small structured dict) so the composite scorer in :mod:`score` can
combine them with simple weighted sums.

Feature dimensions:
* **must_have**     – JD must-have skill/experience matches
* **title**         – current title alignment with "Senior AI Engineer / RAG / ranking"
* **career**        – career trajectory, company quality, complexity arc
* **experience**    – YOE proximity to 5–9 range, project evidence in descriptions
* **behavioral**    – Redrob 23-signal multipliers (engagement, response, openness,
                      social proof, technical signals)
* **logistics**     – location fit (Pune/Noida/Mumbai/Delhi/Hyderabad) and notice
                      period

Disqualifier and honeypot detection live in :mod:`honeypot` and are applied
in :mod:`score` as penalties, not as features here.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# JD-derived lookup tables — derived directly from the JD, not guessed
# =============================================================================

#: Companies explicitly called out as a "not a fit" in the JD if entire career.
#: (Sub-section "Things we explicitly do NOT want" — bullet 3.)
DISQUALIFIER_CONSULTING_FIRMS: Set[str] = {
    "tcs", "infosys", "wipro", "hcl", "cognizant", "capgemini",
    "accenture", "mphasis", "tech mahindra", "ltimindtree", "mindtree",
    "deloitte", "ey", "pwc", "kpmg", "mckinsey", "bcg", "bain",
    "hexaware", "dxc technology", "ibm consulting", "persistent",
}

#: Indian cities the JD explicitly welcomes (Pune/Noida-preferred).
LOCATION_BONUS_CITIES: Set[str] = {
    "pune", "noida", "gurgaon", "gurugram", "mumbai", "bombay",
    "delhi", "new delhi", "delhi ncr", "hyderabad", "bangalore", "bengaluru",
}

#: Indian cities still in scope per JD (case-by-case for other metros).
LOCATION_OK_CITIES: Set[str] = {
    "chennai", "kolkata", "ahmedabad", "jaipur", "lucknow", "indore",
    "nagpur", "coimbatore", "kochi", "thiruvananthapuram",
}

#: Tier-1 schools that the dataset uses internally (tier_1 + tier_2 buckets from schema).
TIER1_TIER2_INSTITUTIONS: Set[str] = {
    "iit", "iisc", "iim", "bits pilani", "bits", "iiit",
    "mit", "stanford", "harvard", "cmu", "carnegie mellon",
    "berkeley", "caltech", "oxford", "cambridge", "imperial college",
    "eth zurich", "princeton", "yale", "columbia", "cornell",
    "ucl", "tsinghua", "peking university",
}

#: MUST-HAVE skills per JD — production retrieval/ranking is the core.
MUST_HAVE_SKILL_PATTERNS: Dict[str, float] = {
    # Production retrieval / embeddings (HIGH weight — JD bullet 1)
    "embeddings": 1.0,
    "sentence-transformers": 1.0,
    "sentence_transformers": 1.0,
    "bge": 1.0,
    "e5": 0.9,
    "openai embeddings": 1.0,
    "vector search": 1.0,
    "semantic search": 1.0,
    "hybrid search": 1.0,
    "reranking": 1.0,
    "cross-encoder": 0.9,
    "bi-encoder": 0.9,
    "dense retrieval": 1.0,
    "sparse retrieval": 0.9,
    "bm25": 0.9,
    "tf-idf": 0.5,
    # Vector DB / search infra (HIGH — JD bullet 2)
    "pinecone": 1.0,
    "weaviate": 1.0,
    "qdrant": 1.0,
    "milvus": 1.0,
    "faiss": 1.0,
    "annoy": 0.8,
    "elasticsearch": 0.7,
    "opensearch": 0.7,
    "chroma": 0.9,
    "pgvector": 0.8,
    # RAG & LLMs (HIGH — "modern ML systems" / "embeddings, retrieval, ranking, LLMs, fine-tuning")
    "rag": 1.0,
    "retrieval augmented": 1.0,
    "retrieval-augmented": 1.0,
    "llm": 0.8,
    "large language model": 0.8,
    "fine-tuning": 0.7,
    "finetuning": 0.7,
    "lora": 0.7,
    "qlora": 0.7,
    "peft": 0.7,
    # Eval frameworks (HIGH — JD bullet 4)
    "ndcg": 1.0,
    "mrr": 1.0,
    "map": 0.8,
    "learning to rank": 1.0,
    "ltr": 0.9,
    "lambdarank": 0.9,
    "a/b testing": 0.8,
    "ab testing": 0.8,
    "offline evaluation": 0.8,
    "online evaluation": 0.7,
    # Core ML (required for the role)
    "python": 0.5,
    "pytorch": 0.5,
    "transformers": 0.6,
    "huggingface": 0.6,
    "scikit-learn": 0.4,
    "xgboost": 0.4,
    "lightgbm": 0.4,
    # Distributed systems (for the "scale" lens)
    "kubernetes": 0.3,
    "docker": 0.2,
    "distributed systems": 0.5,
    "spark": 0.3,
    "kafka": 0.3,
}

#: NICE-TO-HAVE patterns — used as bonuses, not required.
NICE_TO_HAVE_SKILL_PATTERNS: Dict[str, float] = {
    "fastapi": 0.3, "mlops": 0.4, "model deployment": 0.4,
    "inference optimization": 0.4, "model serving": 0.4,
    "triton": 0.3, "bentoml": 0.3, "kserve": 0.3,
    "data engineering": 0.2, "airflow": 0.2, "dbt": 0.2,
    "databricks": 0.2, "snowflake": 0.2, "bigquery": 0.2,
    "streamlit": 0.2, "gradio": 0.2,
    "neural networks": 0.3, "transformer architecture": 0.4,
    "rlhf": 0.3, "dpo": 0.3,
    "mlflow": 0.3, "wandb": 0.3, "weights and biases": 0.3,
    "prompt engineering": 0.2,
}

#: ANTI-pattern skills — listed but in clearly non-AI roles (keyword stuffing signal).
#: If a "Marketing Manager" lists these at "advanced" but has zero career evidence,
#: this is strong evidence of a Tier-5 trap candidate.
ADVANCED_AI_SKILLS: Set[str] = {
    "transformers", "pytorch", "tensorflow", "huggingface", "hugging face",
    "fine-tuning", "finetuning", "lora", "qlora", "peft",
    "rag", "embeddings", "vector search", "semantic search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma",
    "sentence-transformers", "sentence_transformers", "bge", "e5",
    "bm25", "learning to rank", "ltr", "lambdarank",
    "ndcg", "mrr",
    "reranking", "cross-encoder", "bi-encoder",
    "dense retrieval", "hybrid search",
}

#: Strong AI/ML title keywords (case-insensitive substring).
STRONG_AI_TITLE_TOKENS: List[str] = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "ai/ml", "mlops", "ai specialist", "applied ml",
    "data scientist", "research engineer", "research scientist",
    "applied scientist", "nlp engineer",
    "llm engineer", "prompt engineer",
    "senior software engineer (ml)", "senior ai", "principal ai",
    "staff ml", "staff ai",
    "machine learning", "deep learning", "data science",
]

#: Mid-tier technical title tokens — get partial credit.
MID_TITLE_TOKENS: List[str] = [
    "senior software engineer", "software engineer", "backend engineer",
    "full stack", "fullstack", "platform engineer", "infrastructure engineer",
    "data engineer", "analytics engineer", "ml infrastructure",
    "engineering manager", "tech lead", "staff engineer", "principal engineer",
    "cloud engineer", "devops engineer", "site reliability",
]

#: Clear non-technical titles — JD explicitly says "title-chasers", "framework enthusiasts",
#: and lists consulting-firm "Marketing Manager" etc. as not-a-fit.
NON_TECH_TITLE_TOKENS: List[str] = [
    "marketing manager", "marketing", "sales", "sales executive",
    "operations manager", "operations", "hr manager", "human resources",
    "accountant", "accounting",
    "content writer", "content", "graphic designer", "designer",
    "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "business analyst", "project manager", "consultant",
    "recruiter", "talent acquisition",
]

#: Career description keywords signaling REAL production retrieval/ranking work.
PRODUCTION_EVIDENCE_PATTERNS: Dict[str, float] = {
    # Highest-signal — building ranking/retrieval/recommendation systems
    "ranking system": 1.0,
    "search system": 0.9,
    "recommendation system": 0.9,
    "retrieval system": 0.95,
    "embedding": 0.6,
    "vector database": 0.8,
    "vector index": 0.7,
    "semantic search": 0.7,
    "hybrid search": 0.7,
    "reranking": 0.8,
    "learning to rank": 1.0,
    "lambdarank": 0.9,
    "ndcg": 0.9,
    "mrr": 0.8,
    "faiss": 0.7,
    "pinecone": 0.7,
    "weaviate": 0.7,
    "qdrant": 0.7,
    "milvus": 0.7,
    "bm25": 0.6,
    # ML deployment / production
    "deployed": 0.4,
    "production": 0.4,
    "a/b test": 0.5,
    "ab test": 0.5,
    "evaluation framework": 0.6,
    "offline evaluation": 0.6,
    "online evaluation": 0.5,
    "latency": 0.3,
    "throughput": 0.3,
    "scale": 0.3,
    # LLM specifics
    "rag": 0.7,
    "retrieval augmented": 0.7,
    "llm": 0.4,
    "fine-tuning": 0.5,
    "finetuning": 0.5,
    "lora": 0.5,
    "qlora": 0.5,
    # Concrete impact language
    "shipped": 0.4,
    "scaled": 0.3,
    "owned": 0.3,
    "led": 0.3,
    "mentored": 0.2,
}

#: Negative signals in career descriptions — research-only, no production deployment.
NEGATIVE_CAREER_PATTERNS: Dict[str, float] = {
    "academic": -0.4,
    "research paper": -0.3,
    "published paper": -0.2,
    "thesis": -0.3,
    "university research": -0.3,
    "no production": -0.5,
    "purely research": -0.5,
    "no deployment": -0.4,
    "langchain tutorial": -0.4,
    "openai api wrapper": -0.3,
    "chatgpt wrapper": -0.4,
}


# =============================================================================
# Helpers
# =============================================================================

def _norm(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_any(text: str, patterns: List[str]) -> bool:
    return any(p in text for p in patterns)


def _sum_dict_hits(text: str, weighted: Dict[str, float], cap: float = 1.0) -> float:
    """Sum weights of patterns found in text, then cap and rescale to 0..1."""
    total = 0.0
    for pat, weight in weighted.items():
        if pat in text:
            total += weight
    # Cap is the "saturating" total — once you have ~5+ distinct matches, you're at 1.0.
    return min(cap, total / 5.0)


# =============================================================================
# Per-dimension feature extractors
# =============================================================================

def _extract_must_have(cand: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """Return (score 0..1, matched_skills, missing_critical)."""
    skill_names = cand["_all_skill_names"]
    skill_names_joined = " ".join(skill_names)
    headline_summary = _norm(cand.get("headline", "") + " " + cand.get("summary", ""))
    career_text = _norm(
        " ".join(r.get("description", "") + " " + r.get("title", "") for r in cand.get("career_history", []))
    )

    # 1) Skill list matches
    skill_hits = 0
    skill_hits_weighted = 0.0
    matched_skills: List[str] = []
    for pat, weight in MUST_HAVE_SKILL_PATTERNS.items():
        if pat in skill_names_joined:
            skill_hits += 1
            skill_hits_weighted += weight
            matched_skills.append(pat)

    # 2) Description / title evidence (production experience language)
    desc_hits_weighted = _sum_dict_hits(career_text, PRODUCTION_EVIDENCE_PATTERNS, cap=2.0) * 1.0
    # Count description hits
    desc_evidence_count = sum(1 for p in PRODUCTION_EVIDENCE_PATTERNS if p in career_text)

    # 3) Headline/summary mentions
    headline_hits = sum(1 for p in MUST_HAVE_SKILL_PATTERNS if p in headline_summary)

    # Combine
    # A candidate with 3+ must-have skill matches AND career description evidence
    # is a strong match. Without ANY skill match, they get 0.
    if skill_hits == 0 and desc_evidence_count == 0:
        return 0.0, matched_skills, list(MUST_HAVE_SKILL_PATTERNS.keys())[:5]

    # Diminishing returns: 1 hit = strong, 5+ hits = saturating
    skill_score = min(1.0, skill_hits / 4.0)
    desc_score = min(1.0, desc_evidence_count / 3.0)
    headline_bonus = min(0.2, headline_hits * 0.05)

    final = 0.55 * skill_score + 0.40 * desc_score + 0.05 + headline_bonus
    return min(1.0, final), matched_skills, []


def _extract_title(cand: Dict[str, Any]) -> Tuple[float, str]:
    """Return (title_relevance 0..1, title_bucket)."""
    title = _norm(cand.get("current_title", ""))
    headline = _norm(cand.get("headline", ""))
    text = f"{title} {headline}"

    # Strong AI match — top of pack
    for tok in STRONG_AI_TITLE_TOKENS:
        if tok in text:
            return 1.0, "strong_ai"

    # Mid technical (SWE, data eng, etc.)
    for tok in MID_TITLE_TOKENS:
        if tok in text:
            return 0.55, "technical_adjacent"

    # Non-technical — these are Tier-5 territory
    for tok in NON_TECH_TITLE_TOKENS:
        if tok in text:
            return 0.10, "non_technical"

    # Unknown — could be a niche title (e.g., "Research Engineer" we missed)
    if title:
        return 0.40, "unknown_technical"
    return 0.30, "unknown_technical"


def _extract_career(cand: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return (career_score 0..1, breakdown dict)."""
    career = cand.get("career_history", [])
    if not career:
        return 0.30, {"roles": 0}

    # Complexity arc — ascending company sizes / scope
    sizes = []
    for r in career:
        s = r.get("company_size", "")
        sizes.append(_company_size_to_int(s))
    if sizes:
        # Compare first/last to detect ascending / descending / stable
        if len(sizes) >= 2:
            diff = sizes[0] - sizes[-1]
            if diff > 1:
                arc = "ascending"  # current company smaller than earlier = descending in size = bad
                # Wait: career_history is in reverse chrono (current first). So sizes[0] = current, sizes[-1] = oldest.
                # We want to detect if older->newer was ascending (growing company size = bad, growing scope = good)
                # Most candidates at product companies START at small startups and grow. We want to reward that.
                arc = "growing_scope" if sizes[0] > sizes[-1] else "stable_or_shrinking"
            else:
                arc = "stable"
        else:
            arc = "single_role"
        # Growth: company size trend older -> newer (sizes[-1] to sizes[0])
        if len(sizes) >= 2 and sizes[-1] > 0:
            growth_ratio = sizes[0] / sizes[-1]
        else:
            growth_ratio = 1.0
    else:
        arc = "unknown"
        growth_ratio = 1.0

    # Career description evidence (production ML / ranking)
    career_text = _norm(
        " ".join(r.get("description", "") + " " + r.get("title", "") for r in career)
    )
    positive = _sum_dict_hits(career_text, PRODUCTION_EVIDENCE_PATTERNS, cap=2.0)
    negative = _sum_dict_hits(career_text, NEGATIVE_CAREER_PATTERNS, cap=1.0)
    desc_score = max(0.0, positive - negative)

    # Complexity arc: prefer growing_scope > stable > shrinking
    arc_score = {
        "growing_scope": 0.9, "stable": 0.6, "single_role": 0.5,
        "stable_or_shrinking": 0.4, "unknown": 0.5,
    }.get(arc, 0.5)

    # Company quality: product companies > consulting (handled separately as penalty)
    # Here, we just count how many roles are at non-consulting companies
    non_consulting_roles = sum(
        1 for r in career
        if r.get("company", "").lower().strip() not in DISQUALIFIER_CONSULTING_FIRMS
    )
    non_consulting_ratio = non_consulting_roles / len(career) if career else 0.0
    company_quality = 0.3 + 0.7 * non_consulting_ratio  # 0.3..1.0

    # Number of roles (more = more experience, but only up to a point)
    role_count_score = min(1.0, len(career) / 4.0)

    final = 0.40 * desc_score + 0.30 * arc_score + 0.20 * company_quality + 0.10 * role_count_score
    return min(1.0, final), {
        "arc": arc,
        "growth_ratio": growth_ratio,
        "desc_score": desc_score,
        "arc_score": arc_score,
        "company_quality": company_quality,
        "role_count": len(career),
    }


def _extract_experience(cand: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return (experience_fit 0..1, breakdown dict)."""
    yoe = cand.get("years_of_experience", 0.0)

    # JD: 5–9 years is the sweet spot; outside gets a soft penalty.
    if 5.0 <= yoe <= 9.0:
        yoe_score = 1.0
    elif 3.0 <= yoe < 5.0:
        yoe_score = 0.6 + 0.1 * (yoe - 3.0)  # 0.6..0.8
    elif 9.0 < yoe <= 12.0:
        yoe_score = 0.8 - 0.15 * (yoe - 9.0)  # 0.8..0.35
    elif yoe > 12.0:
        yoe_score = 0.30  # Very senior — possible overqualification per JD's "title-chasers" vibe
    elif 2.0 <= yoe < 3.0:
        yoe_score = 0.35
    else:
        yoe_score = 0.10

    return yoe_score, {
        "yoe": yoe,
        "yoe_score": yoe_score,
    }


def _extract_behavioral(cand: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return (behavioral_score 0..1, breakdown dict).

    Five sub-dimensions:
    * engagement     — profile completeness + activity recency + views
    * responsiveness — response rate + speed + interview completion
    * openness       — open_to_work + notice period + work mode flexibility
    * social_proof   — endorsements + connections + verified channels
    * technical      — GitHub + skill assessments
    """
    sig = cand.get("signals", {})

    # 1. Engagement (profile completeness + recency + views)
    completeness = _clamp(sig.get("profile_completeness_score", 0.0) / 100.0, 0.0, 1.0)
    days_inactive = _days_since(sig.get("last_active_date"))
    recency = 1.0 - _clamp((days_inactive or 365) / 365.0, 0.0, 1.0)
    views = _clamp(sig.get("profile_views_received_30d", 0) / 200.0, 0.0, 1.0)
    search_appear = _clamp(sig.get("search_appearance_30d", 0) / 1000.0, 0.0, 1.0)
    saved = _clamp(sig.get("saved_by_recruiters_30d", 0) / 20.0, 0.0, 1.0)
    engagement = 0.30 * completeness + 0.30 * recency + 0.10 * views + 0.15 * search_appear + 0.15 * saved

    # 2. Responsiveness
    resp_rate = _clamp(sig.get("recruiter_response_rate", 0.0), 0.0, 1.0)
    # Inverse hours: <24h = great, >200h = bad
    speed = 1.0 - _clamp((sig.get("avg_response_time_hours", 200.0) or 200.0) / 200.0, 0.0, 1.0)
    interview = _clamp(sig.get("interview_completion_rate", 0.0), 0.0, 1.0)
    responsiveness = 0.45 * resp_rate + 0.25 * speed + 0.30 * interview

    # 3. Openness (willingness to engage)
    otw = 1.0 if sig.get("open_to_work_flag", False) else 0.3
    notice = 1.0 - _clamp((sig.get("notice_period_days", 90) or 90) / 90.0, 0.0, 1.0)
    work_mode = sig.get("preferred_work_mode", "flexible")
    work_mode_score = {
        "remote": 0.8, "flexible": 1.0, "hybrid": 0.9, "onsite": 0.6,
    }.get(work_mode, 0.7)
    relocate = 0.5 + 0.5 * (1.0 if sig.get("willing_to_relocate", False) else 0.0)
    openness = 0.40 * otw + 0.25 * notice + 0.20 * work_mode_score + 0.15 * relocate

    # 4. Social proof
    endorsements = _clamp(sig.get("endorsements_received", 0) / 50.0, 0.0, 1.0)
    connections = _clamp(sig.get("connection_count", 0) / 500.0, 0.0, 1.0)
    verified = sum([
        1.0 if sig.get("verified_email", False) else 0.0,
        1.0 if sig.get("verified_phone", False) else 0.0,
        1.0 if sig.get("linkedin_connected", False) else 0.0,
    ]) / 3.0
    social_proof = 0.45 * endorsements + 0.25 * connections + 0.30 * verified

    # 5. Technical (GitHub + skill assessments)
    gh = sig.get("github_activity_score", -1)
    if gh == -1 or gh is None:
        gh_score = 0.0  # No GitHub linked — neutral (not a penalty)
    else:
        gh_score = _clamp(gh / 80.0, 0.0, 1.0)
    assessments = sig.get("skill_assessment_scores", {}) or {}
    if assessments:
        # Penalize low-scoring AI/ML assessments (someone claims "expert" but scores 30)
        avg_assessment = sum(assessments.values()) / len(assessments) / 100.0
        ai_assessments = [
            v / 100.0 for k, v in assessments.items()
            if any(p in k.lower() for p in ["ml", "ai", "nlp", "retrieval", "embedding", "rag", "llm", "transform"])
        ]
        ai_avg = sum(ai_assessments) / len(ai_assessments) if ai_assessments else None
        if ai_avg is not None:
            tech_assess = 0.5 * avg_assessment + 0.5 * ai_avg
        else:
            tech_assess = 0.5 * avg_assessment
    else:
        tech_assess = 0.0
    technical = 0.45 * gh_score + 0.55 * tech_assess

    # Combine
    combined = (
        0.20 * engagement
        + 0.20 * responsiveness
        + 0.15 * openness
        + 0.15 * social_proof
        + 0.30 * technical  # technical weight higher for AI/ML role
    )

    return min(1.0, combined), {
        "engagement": engagement,
        "responsiveness": responsiveness,
        "openness": openness,
        "social_proof": social_proof,
        "technical": technical,
        "combined": combined,
    }


def _extract_logistics(cand: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Return (logistics_score 0..1, breakdown)."""
    location = _norm(cand.get("location", ""))
    country = _norm(cand.get("country", ""))

    # Indian candidates strongly preferred (Pune/Noida/Mumbai/Delhi/Hyderabad)
    if any(city in location for city in LOCATION_BONUS_CITIES):
        loc_score = 0.95
    elif any(city in location for city in LOCATION_OK_CITIES):
        loc_score = 0.75
    elif "india" in country:
        loc_score = 0.65
    elif country in ("united states", "canada", "uk", "united kingdom", "germany", "singapore"):
        loc_score = 0.50  # Outside India, case-by-case per JD
    else:
        loc_score = 0.40

    # Salary expectation — JD doesn't list explicit range, but seniority says "Series A founding team"
    # We expect ~30-50 LPA typical. If expected range max is way out of band, slight penalty.
    salary = cand.get("signals", {}).get("expected_salary_range_inr_lpa", {}) or {}
    sal_min = salary.get("min", 0) or 0
    sal_max = salary.get("max", 0) or 0
    sal_center = (sal_min + sal_max) / 2.0 if sal_min and sal_max else 0
    if 25 <= sal_center <= 80:
        salary_score = 1.0
    elif 15 <= sal_center < 25 or 80 < sal_center <= 100:
        salary_score = 0.7
    elif sal_center == 0:
        salary_score = 0.5
    else:
        salary_score = 0.4

    final = 0.6 * loc_score + 0.4 * salary_score
    return min(1.0, final), {
        "location_score": loc_score,
        "salary_score": salary_score,
        "salary_center_lpa": sal_center,
    }


def _extract_disqualifiers(cand: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Return (penalty 0..1, list of triggered disqualifier reasons)."""
    reasons: List[str] = []

    # Disqualifier 1: Entire career at consulting firms (JD explicit)
    career = cand.get("career_history", [])
    if career:
        consulting_roles = sum(
            1 for r in career
            if r.get("company", "").lower().strip() in DISQUALIFIER_CONSULTING_FIRMS
        )
        consulting_ratio = consulting_roles / len(career)
        if consulting_ratio >= 1.0:
            reasons.append("entire_career_consulting")
        elif consulting_ratio >= 0.75:
            reasons.append("mostly_consulting")

    # Disqualifier 2: Pure research — check for "academic" or "thesis" but no industry role
    summary = _norm(cand.get("summary", ""))
    career_text = _norm(
        " ".join(r.get("description", "") for r in career)
    )
    if ("phd" in summary or "thesis" in summary) and "production" not in career_text:
        reasons.append("research_only")

    # Disqualifier 3: "Title chaser" — many short stints at the same seniority
    short_stints = sum(1 for r in career if r.get("duration_months", 0) and r.get("duration_months") < 18)
    if len(career) >= 4 and short_stints / len(career) >= 0.75:
        reasons.append("title_chaser_pattern")

    # Disqualifier 4: LangChain-only AI experience (per JD)
    skills = cand["_all_skill_names"]
    skills_text = " ".join(skills)
    if "langchain" in skills_text and not any(
        p in skills_text for p in ["pytorch", "tensorflow", "huggingface", "transformers", "scikit-learn"]
    ) and not any(p in career_text for p in ["production", "deployed", "scaled"]):
        reasons.append("langchain_only")

    # Disqualifier 5: Non-technical role with many advanced AI skills = keyword stuffing
    title = _norm(cand.get("current_title", ""))
    is_non_tech = any(tok in title for tok in NON_TECH_TITLE_TOKENS)
    advanced_skill_count = sum(1 for s in skills if s in ADVANCED_AI_SKILLS)
    if is_non_tech and advanced_skill_count >= 3:
        reasons.append("keyword_stuffer")

    # Penalty: 0.0 = no penalty, 1.0 = full disqualification
    penalty = 0.0
    if reasons:
        if "entire_career_consulting" in reasons:
            penalty = 0.50
        if "mostly_consulting" in reasons:
            penalty = max(penalty, 0.30)
        if "research_only" in reasons:
            penalty = max(penalty, 0.30)
        if "title_chaser_pattern" in reasons:
            penalty = max(penalty, 0.20)
        if "langchain_only" in reasons:
            penalty = max(penalty, 0.30)
        if "keyword_stuffer" in reasons:
            penalty = max(penalty, 0.40)  # strong signal of Tier-5 trap
    return penalty, reasons


# =============================================================================
# Helpers
# =============================================================================

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _days_since(iso_date: Optional[str]) -> Optional[int]:
    """Return days elapsed since iso_date, or None on parse failure."""
    if not iso_date:
        return None
    try:
        from datetime import date, datetime
        s = str(iso_date)[:10]
        dt = datetime.fromisoformat(s).date()
        # Dataset is dated 2026-07; reference as today
        return (date(2026, 7, 2) - dt).days
    except (ValueError, TypeError):
        return None


def _company_size_to_int(size_str: str) -> int:
    """Convert a company size bucket to an ordinal integer for arc analysis."""
    if not size_str:
        return 0
    s = str(size_str).lower().strip()
    mapping = {
        "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
        "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8,
        "startup <50": 1, "scaleup 50-500": 4, "enterprise 500+": 6,
    }
    return mapping.get(s, 0)


# =============================================================================
# Public API
# =============================================================================

def extract_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all per-candidate features for scoring.

    Returns a flat dict with the following structure (all 0..1 unless noted):
    {
        'must_have_score': float,    # JD must-have skill + experience match
        'matched_skills': list[str], # which skills triggered must_have
        'title_score': float,        # current title alignment
        'title_bucket': str,         # 'strong_ai' | 'technical_adjacent' | 'non_technical' | 'unknown_technical'
        'career_score': float,       # career trajectory + production evidence
        'career_breakdown': dict,    # detailed career metrics
        'experience_score': float,   # YOE proximity to 5-9 range
        'experience_breakdown': dict,
        'behavioral_score': float,   # 5-dim Redrob signal combination
        'behavioral_breakdown': dict,
        'logistics_score': float,    # location + salary
        'logistics_breakdown': dict,
        'disqualifier_penalty': float,  # 0..1, applied as soft penalty
        'disqualifier_reasons': list[str],
    }
    """
    must_have, matched, missing = _extract_must_have(cand)
    title, title_bucket = _extract_title(cand)
    career, career_bd = _extract_career(cand)
    experience, exp_bd = _extract_experience(cand)
    behavioral, beh_bd = _extract_behavioral(cand)
    logistics, log_bd = _extract_logistics(cand)
    disq_penalty, disq_reasons = _extract_disqualifiers(cand)

    return {
        "must_have_score": must_have,
        "matched_skills": matched,
        "title_score": title,
        "title_bucket": title_bucket,
        "career_score": career,
        "career_breakdown": career_bd,
        "experience_score": experience,
        "experience_breakdown": exp_bd,
        "behavioral_score": behavioral,
        "behavioral_breakdown": beh_bd,
        "logistics_score": logistics,
        "logistics_breakdown": log_bd,
        "disqualifier_penalty": disq_penalty,
        "disqualifier_reasons": disq_reasons,
    }