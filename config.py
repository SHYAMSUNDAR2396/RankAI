"""Centralized configuration for the Candidate Ranking System.

This module is the single source of truth for every tunable constant used by the
pipeline (Requirement 10). No other module redefines these values; all components
import them from here so that behavior can be adjusted without editing pipeline
logic (Requirement 10.5). Prompt-template text and persona system prompts
(``personas/*.txt``) are intentionally kept outside this module.

Constant groups:
    * Backend / model selection (Ollama, embeddings, ChromaDB).
    * Token limits and scoring temperature for LLM calls.
    * Scoring weights and decision thresholds.
    * Lookup tables used by enrichment and the counterfactual fairness audit.
    * Calibration examples that anchor multi-persona scoring.
"""

from __future__ import annotations

import os

# Backend selection: "ollama" or "groq"
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama").lower()

# Groq API Configuration
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Backend and model selection
# ---------------------------------------------------------------------------

# Per-agent model assignment — optimised for task type
OLLAMA_MODELS = {
    "parser":           "qwen2.5:7b",      # Structured JSON extraction
    "jd_parser":        "qwen2.5:7b",      # Requirement classification
    "orchestrator":     "llama3.1:8b",     # Routing and state management
    "skills_match":     "llama3.2:3b",     # Simple list comparison, RAG output
    "trajectory":       "deepseek-r1:7b",  # Reasoning: career growth arc
    "hiring_manager":   "llama3.1:8b",     # Highest weight persona (0.45)
    "peer_interviewer": "qwen2.5:14b",     # Technical depth evaluation (0.35)
    "devils_advocate":  "deepseek-r1:7b",  # Adversarial reasoning (0.20 inverted)
    "consensus":        "llama3.2:3b",     # Narrative only — math done in Python
    "narrative":        "llama3.2:3b",     # 3-sentence summary
}

# Groq models optimized for tasks
GROQ_MODELS = {
    "parser":           "llama-3.3-70b-versatile",
    "jd_parser":        "llama-3.3-70b-versatile",
    "orchestrator":     "llama-3.3-70b-versatile",
    "skills_match":     "llama-3.1-8b-instant",
    "trajectory":       "llama-3.3-70b-versatile",
    "hiring_manager":   "llama-3.3-70b-versatile",
    "peer_interviewer": "llama-3.3-70b-versatile",
    "devils_advocate":  "llama-3.3-70b-versatile",
    "consensus":        "llama-3.1-8b-instant",
    "narrative":        "llama-3.1-8b-instant",
}

# Hardware-aware override — set True on low-RAM machines (< 20GB)
LOW_MEMORY_MODE = False  # Set True on Intel 16GB machines

# When LOW_MEMORY_MODE is True, replace heavy models with lighter equivalents
OLLAMA_MODELS_LOW_MEM = {
    **OLLAMA_MODELS,
    "peer_interviewer": "qwen2.5:7b",   # 14b → 7b (saves ~4GB RAM)
    "trajectory":       "llama3.1:8b",  # deepseek-r1 → llama (faster on Intel)
    "devils_advocate":  "llama3.1:8b",
}

def get_model(agent_name: str) -> str:
    """Return the appropriate model for a given agent name."""
    if LLM_BACKEND == "groq":
        return GROQ_MODELS.get(agent_name, "llama-3.1-8b-instant")
    model_map = OLLAMA_MODELS_LOW_MEM if LOW_MEMORY_MODE else OLLAMA_MODELS
    return model_map.get(agent_name, "llama3.2:3b")

#: URL of the locally hosted Ollama server (Requirement 1.1, 10.1).
OLLAMA_HOST: str = "http://localhost:11434"

#: Local sentence-transformers embedding model (Requirement 1.2, 10.1).
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"

#: On-disk directory for the ChromaDB PersistentClient (Requirement 1.3, 10.1).
CHROMA_PERSIST_DIR: str = "./chroma_store"


# ---------------------------------------------------------------------------
# Token limits and scoring temperature (Requirement 10.1)
# ---------------------------------------------------------------------------

#: Max tokens for structured profile / JD extraction calls (positive int).
MAX_TOKENS_EXTRACTION: int = 800

#: Max tokens for per-persona scoring calls (positive int).
MAX_TOKENS_SCORING: int = 400

#: Max tokens for the three-sentence narrative call (positive int).
MAX_TOKENS_NARRATIVE: int = 150

#: Sampling temperature for scoring calls; kept low for stability. In [0.0, 1.0].
SCORING_TEMPERATURE: float = 0.2


# ---------------------------------------------------------------------------
# Scoring weights and decision thresholds (Requirement 10.2)
# ---------------------------------------------------------------------------

#: Composite-score weights for the three evaluator personas.
#:
#: The composite score is a plain weighted sum of the persona scores:
#:     composite = sum(PERSONA_WEIGHTS[persona] * persona_score)
#:               = 0.45 * hiring_manager
#:               + 0.35 * peer_interviewer
#:               - 0.20 * devils_advocate
#:
#: NOTE: ``devils_advocate`` carries a NEGATIVE weight (-0.20) on purpose: the
#: devil's-advocate score is *subtracted* from the composite (it penalizes the
#: candidate). Storing the sign here keeps this module the single source of truth
#: so ``score.py`` can compute a straight weighted sum without re-encoding the
#: subtraction (Requirements 6.3, 10.5).
PERSONA_WEIGHTS: dict[str, float] = {
    "hiring_manager": 0.45,
    "peer_interviewer": 0.35,
    "devils_advocate": -0.20,  # subtracted from the composite
}

#: Counterfactual delta above which a candidate is flagged for potential bias.
BIAS_FLAG_THRESHOLD: float = 0.75

#: Panel-variance above which a candidate is routed to human review.
HUMAN_REVIEW_VARIANCE_THRESHOLD: float = 2.5


# ---------------------------------------------------------------------------
# Lookup tables (Requirement 10.3)
# ---------------------------------------------------------------------------

#: Ordinal seniority levels inferred from role titles, used by enrichment to
#: measure leadership progression. Higher values indicate more senior roles.
TITLE_LEVELS: dict[str, int] = {
    "intern": 0,
    "junior": 1,
    "associate": 1,
    "mid": 2,
    "intermediate": 2,
    "senior": 3,
    "staff": 4,
    "lead": 4,
    "principal": 5,
    "architect": 5,
    "manager": 5,
    "director": 6,
    "vp": 7,
    "head": 6,
    "chief": 8,
    "fellow": 8,
}

#: Name pairs used by the counterfactual audit to swap demographic signals.
#: Each tuple is (original_signal_name, swapped_signal_name); the swap is applied
#: in both directions when constructing a counterfactual twin.
COUNTERFACTUAL_NAME_PAIRS: list[tuple[str, str]] = [
    ("James", "Priya"),
    ("Michael", "Fatima"),
    ("David", "Mei"),
    ("John", "Aisha"),
    ("Robert", "Neha"),
    ("William", "Yuki"),
    ("Sarah", "Mohammed"),
    ("Emily", "Raj"),
    ("Jessica", "Kwame"),
    ("Ashley", "Chen"),
]

#: Institution swaps used by the counterfactual audit to substitute prestigious
#: institutions with lower-prestige peers, isolating any prestige-driven bias.
INSTITUTION_SWAPS: dict[str, str] = {
    "Oxford": "University of Birmingham",
    "Cambridge": "University of Leicester",
    "Harvard": "University of Arizona",
    "MIT": "University of New Mexico",
    "Stanford": "San Jose State University",
    "Yale": "University of Connecticut",
    "Princeton": "Rutgers University",
    "Columbia": "CUNY City College",
    "Imperial College": "University of Portsmouth",
    "LSE": "University of Hertfordshire",
}


# ---------------------------------------------------------------------------
# Calibration examples (Requirement 10.4)
# ---------------------------------------------------------------------------

#: Labeled reference candidates that anchor multi-persona scoring via RAG.
#: Contains EXACTLY 10 entries: 5 ``strong_hire`` followed by 5 ``no_hire``.
#: Each entry is a dict with keys ``profile_summary``, ``outcome``
#: (``"strong_hire"`` | ``"no_hire"``), and ``reason``.
CALIBRATION_EXAMPLES: list[dict[str, str]] = [
    {
        "profile_summary": (
            "Sofia Ramirez - Senior Backend Engineer with 8 years building "
            "distributed payment systems in Python and Go. Led a team of 5, "
            "scaled services to 10M daily transactions, and drove a migration "
            "to event-driven architecture."
        ),
        "outcome": "strong_hire",
        "reason": (
            "Deep, directly relevant systems experience with clear ownership "
            "and measurable scale; demonstrated technical leadership and steady "
            "upward trajectory align tightly with the role's must-have skills."
        ),
    },
    {
        "profile_summary": (
            "Marcus Chen - Full-stack engineer with 6 years at two Series B "
            "SaaS startups. Shipped core product features end to end, mentored "
            "two juniors, and introduced automated testing that cut regressions "
            "by 40%."
        ),
        "outcome": "strong_hire",
        "reason": (
            "Proven ability to deliver in a fast-moving startup environment, "
            "strong product sense, and concrete impact on quality; experience "
            "level and culture signals are an excellent match for the team."
        ),
    },
    {
        "profile_summary": (
            "Amara Okafor - Staff Engineer with 10 years across fintech and "
            "infrastructure. Designed a multi-region data platform, set "
            "engineering standards org-wide, and partnered with product on "
            "roadmap planning."
        ),
        "outcome": "strong_hire",
        "reason": (
            "Exceptional depth in architecture and cross-functional influence; "
            "track record of raising the bar for an entire organization exceeds "
            "the seniority markers required for the role."
        ),
    },
    {
        "profile_summary": (
            "Daniel Park - Backend engineer with 7 years in cloud-native "
            "services. Owned the API platform, reduced p99 latency by 60%, and "
            "led the on-call rotation while mentoring new hires."
        ),
        "outcome": "strong_hire",
        "reason": (
            "Strong reliability and performance engineering record with hands-on "
            "ownership and mentorship; technical and domain dimensions map "
            "cleanly onto the job's core requirements."
        ),
    },
    {
        "profile_summary": (
            "Zara Nwosu - Senior Software Engineer with 9 years, transitioned "
            "from data engineering into platform work. Built CI/CD tooling "
            "adopted by 40 engineers and championed a culture of code review "
            "and documentation."
        ),
        "outcome": "strong_hire",
        "reason": (
            "Versatile, high-impact engineer whose tooling and culture "
            "contributions show both technical strength and strong collaboration "
            "signals; growth trajectory is consistently ascending."
        ),
    },
    {
        "profile_summary": (
            "Tom White - Junior developer with 1 year of experience, mostly "
            "small front-end tweaks on a single internal app. Limited exposure "
            "to backend systems, testing, or production operations."
        ),
        "outcome": "no_hire",
        "reason": (
            "Experience level falls well short of the seniority markers and "
            "must-have systems skills; no evidence of the autonomy or scale the "
            "role demands."
        ),
    },
    {
        "profile_summary": (
            "Lisa Brown - Engineer with 4 years but frequent short stints "
            "(three roles under 12 months each) with no clear progression and "
            "no described impact or ownership."
        ),
        "outcome": "no_hire",
        "reason": (
            "Inconsistent tenure and a flat trajectory with no measurable "
            "outcomes raise concerns about fit and follow-through against a "
            "role that expects sustained ownership."
        ),
    },
    {
        "profile_summary": (
            "Kevin Smith - Backend developer with 5 years exclusively in a "
            "legacy monolith using technologies unrelated to the role's stack. "
            "No cloud, distributed-systems, or modern tooling experience."
        ),
        "outcome": "no_hire",
        "reason": (
            "Skills are misaligned with the must-have technical requirements and "
            "there is no signal of adaptability to the team's stack or scale."
        ),
    },
    {
        "profile_summary": (
            "Maria Garcia - Career-changer with 2 years post-bootcamp, strong "
            "enthusiasm but limited to tutorial-style projects. No production "
            "ownership, team collaboration, or scaling experience described."
        ),
        "outcome": "no_hire",
        "reason": (
            "Promising motivation but insufficient depth and production "
            "experience for a senior role; technical and experience-level "
            "dimensions do not yet meet the bar."
        ),
    },
    {
        "profile_summary": (
            "Ryan Johnson - Engineer with 6 years listing many buzzwords but no "
            "concrete accomplishments, metrics, or evidence of ownership. Roles "
            "and responsibilities are vague throughout."
        ),
        "outcome": "no_hire",
        "reason": (
            "Lack of verifiable impact or specificity makes it impossible to "
            "substantiate the claimed seniority; signals do not support the "
            "must-have competencies for the role."
        ),
    },
]


# ---------------------------------------------------------------------------
# Competition mode: deterministic scoring weights & honeypot thresholds
# ---------------------------------------------------------------------------

#: Feature dimension weights for the competition scoring engine.
#: Must sum to 1.0.  All scoring is deterministic — no LLM calls.
COMPETITION_WEIGHTS: dict[str, float] = {
    "skill_match": 0.35,
    "experience_fit": 0.25,
    "behavioral_signals": 0.20,
    "trajectory_education": 0.20,
}

#: Optimal YOE range for the Senior AI Engineer role.
COMPETITION_OPTIMAL_YOE_MIN: float = 5.0
COMPETITION_OPTIMAL_YOE_MAX: float = 9.0

#: Honeypot detection thresholds.
HONEYPOT_CAREER_IMPOSSIBILITY_THRESHOLD: float = 2.0  # years overlap ratio
HONEYPOT_SKILL_CONSISTENCY_MIN: int = 3  # min endorsements for "expert"
HONEYPOT_YOE_CAREER_MISMATCH_YEARS: float = 3.0
HONEYPOT_HONEYPOT_RATE_DISQUALIFY: float = 0.10  # >10% in top 100 = DQ

#: Senior AI Engineer job requirement keywords (case-insensitive).
#: Used by the skill-matching scorer to compute overlap with candidate skills.
COMPETITION_MUST_HAVE_SKILLS: list[str] = [
    "embeddings", "retrieval", "ranking", "llm", "fine-tuning",
    "pytorch", "transformers", "rag", "vector",
]
COMPETITION_NICE_TO_HAVE_SKILLS: list[str] = [
    "fastapi", "kubernetes", "docker", "mlops", "production",
    "evaluation", "a/b testing", "startup",
]

#: Career trajectory keywords for the trajectory/education scorer.
COMPETITION_TRAJECTORY_KEYWORDS: list[str] = [
    "lead", "senior", "staff", "principal", "architect",
    "founded", "built", "scaled", "launched", "shipped",
]


# ---------------------------------------------------------------------------
# Competition mode: title / company / education / location intelligence
# ---------------------------------------------------------------------------

#: Title relevance scores — higher means more aligned with the Senior AI Engineer role.
#: Titles not listed score 0.3 (unknown, neutral).
COMPETITION_TITLE_SCORES: dict[str, float] = {
    # Core AI/ML roles (strong match)
    "ai engineer": 1.0,
    "ml engineer": 1.0,
    "machine learning engineer": 1.0,
    "data scientist": 0.9,
    "research scientist": 0.7,
    "research engineer": 0.8,
    # Software engineering with AI focus
    "senior software engineer": 0.75,
    "staff engineer": 0.8,
    "staff software engineer": 0.8,
    "principal engineer": 0.85,
    "software engineer": 0.65,
    "backend engineer": 0.6,
    "full stack engineer": 0.55,
    "full-stack engineer": 0.55,
    "platform engineer": 0.6,
    "infrastructure engineer": 0.55,
    # Data engineering (adjacent)
    "data engineer": 0.6,
    "analytics engineer": 0.5,
    # Leadership (if technical background)
    "engineering manager": 0.65,
    "vp engineering": 0.6,
    "head of engineering": 0.6,
    "director of engineering": 0.6,
    # Clearly non-technical (low match)
    "marketing manager": 0.15,
    "operations manager": 0.15,
    "accountant": 0.05,
    "hr manager": 0.1,
    "business analyst": 0.2,
    "project manager": 0.15,
    "content writer": 0.1,
    "customer support": 0.1,
    "civil engineer": 0.1,
    "mechanical engineer": 0.1,
    "electrical engineer": 0.15,
    "sales": 0.1,
    "consultant": 0.15,
}

#: Consulting firms explicitly mentioned in the JD as "not a fit" if entire career.
COMPETITION_CONSULTING_FIRMS: set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "deloitte", "ey", "pwc", "kpmg", "mckinsey", "bcg", "bain",
    "tech mahindra", "hcl", "ltimindtree", "mindtree", "mpHASis",
    "hexaware", "mphasis", "dxc technology", "ibm consulting",
}

#: Education field relevance — how well a degree field aligns with AI/ML engineering.
COMPETITION_EDUCATION_FIELD_SCORES: dict[str, float] = {
    # Strong alignment
    "computer science": 1.0,
    "artificial intelligence": 1.0,
    "machine learning": 1.0,
    "data science": 0.9,
    "computer engineering": 0.9,
    "computer science and engineering": 1.0,
    "information technology": 0.7,
    "software engineering": 0.85,
    # Moderate alignment
    "electronics": 0.5,
    "electronics and communication": 0.5,
    "electrical engineering": 0.45,
    "mathematics": 0.5,
    "statistics": 0.55,
    "applied mathematics": 0.6,
    "computational linguistics": 0.7,
    # Low alignment
    "mechanical engineering": 0.2,
    "civil engineering": 0.15,
    "chemical engineering": 0.1,
    "biotechnology": 0.15,
    "bioinformatics": 0.3,
}

#: India locations — candidates in these cities/regions get a boost.
COMPETITION_INDIA_LOCATIONS: set[str] = {
    "pune", "noida", "gurgaon", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "chennai", "kolkata", "ahmedabad", "jaipur",
    "lucknow", "indore", "nagpur", "coimbatore", "kochi", "thiruvananthapuram",
}

#: Keywords in career descriptions that signal relevant ML/search/ranking experience.
COMPETITION_CAREER_DESCRIPTION_KEYWORDS: dict[str, float] = {
    # High-signal keywords (building ranking/search/recommendation systems)
    "ranking system": 1.0,
    "search system": 0.95,
    "recommendation system": 0.9,
    "retrieval system": 0.95,
    "embedding": 0.8,
    "vector database": 0.85,
    "vector search": 0.9,
    "semantic search": 0.9,
    "hybrid search": 0.85,
    "reranking": 0.9,
    "learning to rank": 1.0,
    "ndcg": 0.95,
    "bm25": 0.8,
    "tf-idf": 0.7,
    "annoy": 0.75,
    "faiss": 0.85,
    "pinecone": 0.8,
    "weaviate": 0.8,
    "qdrant": 0.8,
    "milvus": 0.8,
    "elasticsearch": 0.6,
    "opensearch": 0.65,
    # ML deployment / production signals
    "production": 0.6,
    "deployed": 0.5,
    "a/b test": 0.6,
    "ab test": 0.6,
    "evaluation framework": 0.7,
    "offline evaluation": 0.7,
    "online evaluation": 0.65,
    "latency": 0.4,
    "throughput": 0.4,
    "scale": 0.4,
    # LLM-specific
    "fine-tuning": 0.7,
    "finetuning": 0.7,
    "lora": 0.75,
    "qlora": 0.75,
    "peft": 0.7,
    "prompt engineering": 0.4,
    "rag": 0.85,
    "retrieval augmented": 0.85,
    "llm": 0.6,
    "large language model": 0.65,
    # Negative signals (research-only, no production)
    "research paper": -0.3,
    "published paper": -0.2,
    "academic": -0.3,
    "thesis": -0.2,
}

#: Anti-pattern detection: keywords that signal "keyword stuffing" when title is non-technical.
COMPETITION_STUFFING_TITLE_KEYWORDS: set[str] = {
    "marketing manager", "operations manager", "accountant", "hr manager",
    "business analyst", "project manager", "content writer", "customer support",
    "sales", "consultant", "civil engineer", "mechanical engineer",
}
