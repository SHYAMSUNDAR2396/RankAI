# Antigravity Upgrade Prompt — RankAI
# Paste this into Antigravity's agent panel after opening the existing project

---

## CONTEXT — READ THIS FIRST

This is an existing, partially complete project. Do NOT rebuild from scratch.
Do NOT delete or overwrite anything unless explicitly told to.
Read ARCHITECTURE.md and config.py before touching any file.
Every change must be surgical — modify only what is listed below.

The project is called RankAI. It is a local-first AI candidate ranking system.
Stack: Python 3.11, Ollama, sentence-transformers, ChromaDB, FastAPI, React + Vite + TypeScript.

Run `pytest tests/ -v` before starting. All tests must pass before and after your changes.

---

## UPGRADE 1 — Per-agent model selection (HIGHEST PRIORITY)

### Problem
config.py currently uses a single OLLAMA_MODEL = "llama3.2:3b" for all LLM calls.
Every agent — parsing, scoring, reasoning — uses the same weak 3b model.
This degrades output quality on the tasks that matter most for judging.

### What to change in config.py

REPLACE the single OLLAMA_MODEL constant with a per-agent model map:

```python
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
    model_map = OLLAMA_MODELS_LOW_MEM if LOW_MEMORY_MODE else OLLAMA_MODELS
    return model_map.get(agent_name, "llama3.2:3b")
```

Keep all other constants (BIAS_FLAG_THRESHOLD, PERSONA_WEIGHTS, etc.) unchanged.

### What to change in utils/ollama_client.py

The OllamaClient.call() method currently uses a fixed model from config.
Change its signature to accept an optional model parameter:

```python
def call(self, prompt: str, system: str = "", 
         max_tokens: int = 500, model: str | None = None) -> str:
    """
    Call Ollama with optional per-call model override.
    Falls back to config default if model is None.
    """
    resolved_model = model or OLLAMA_MODEL_DEFAULT
    # rest of existing retry logic unchanged — just replace the model reference
```

Add at top of file:
```python
OLLAMA_MODEL_DEFAULT = "llama3.2:3b"  # fallback only
```

### What to change in pipeline/ingest.py

Find every ollama_client.call() or equivalent in ResumeParser and JdParser.
Add model=get_model("parser") and model=get_model("jd_parser") respectively.

```python
from config import get_model

# In ResumeParser._extract_with_llm():
response = self.client.call(prompt, system=system, 
                             max_tokens=MAX_TOKENS_EXTRACTION,
                             model=get_model("parser"))

# In JdParser._classify_requirements():
response = self.client.call(prompt, system=system,
                             max_tokens=MAX_TOKENS_EXTRACTION, 
                             model=get_model("jd_parser"))
```

### What to change in pipeline/enrich.py

Find the seniority_score LLM call in TrajectoryEnricher.
Add model=get_model("trajectory"):

```python
response = self.client.call(prompt, system="",
                             max_tokens=MAX_TOKENS_SCORING,
                             model=get_model("trajectory"))
```

### What to change in pipeline/score.py

This is the most important change. Find the three persona scoring calls.
Each persona must now use its designated model:

```python
from config import get_model

# Hiring manager persona call:
hm_response = self.client.call(
    user_prompt, system=hiring_manager_prompt,
    max_tokens=MAX_TOKENS_SCORING,
    model=get_model("hiring_manager")
)

# Peer interviewer persona call:
peer_response = self.client.call(
    user_prompt, system=peer_interviewer_prompt,
    max_tokens=MAX_TOKENS_SCORING,
    model=get_model("peer_interviewer")
)

# Devil's advocate persona call:
da_response = self.client.call(
    user_prompt, system=devils_advocate_prompt,
    max_tokens=MAX_TOKENS_SCORING,
    model=get_model("devils_advocate")
)

# Narrative generation:
narrative = self.client.call(
    narrative_prompt, system="",
    max_tokens=MAX_TOKENS_NARRATIVE,
    model=get_model("narrative")
)
```

Do NOT change the composite score formula, weighting, or variance logic.
Do NOT change persona prompt files.
Do NOT change the output schema.

### Verify after this change

Run: python -c "from config import get_model; print(get_model('peer_interviewer'))"
Expected output: qwen2.5:14b (or qwen2.5:7b if LOW_MEMORY_MODE=True)

Run: pytest tests/ -v
All tests must still pass (they mock ollama calls so model names don't affect them).

---

## UPGRADE 2 — Pull new Ollama models

After config.py is updated, add a check_models_present() upgrade to main.py.

Find the existing check_models_present() function.
Replace the hardcoded model list with one derived from config:

```python
from config import OLLAMA_MODELS, OLLAMA_MODELS_LOW_MEM, LOW_MEMORY_MODE

def check_models_present() -> None:
    """Verify all required Ollama models are pulled."""
    required = set(
        (OLLAMA_MODELS_LOW_MEM if LOW_MEMORY_MODE else OLLAMA_MODELS).values()
    )
    # existing pull-check logic, but iterate `required` not a hardcoded list
```

Also update README.md. Find the "Prerequisites" or "Setup" section.
Replace any hardcoded `ollama pull llama3.2:3b` with:

```bash
# Pull all required models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b      # Skip if on low-RAM machine, set LOW_MEMORY_MODE=True
ollama pull deepseek-r1:7b

# On Intel 16GB machines: set LOW_MEMORY_MODE = True in config.py
# This automatically substitutes lighter models for the heavy ones.
```

---

## UPGRADE 3 — Empty state / onboarding page

### Problem
The app has no empty state — on first load before any pipeline has run,
the dashboard shows a broken or empty table with no guidance.

### Create frontend/src/pages/EmptyStatePage.tsx

```tsx
import { useNavigate } from 'react-router-dom';

export default function EmptyStatePage() {
  const navigate = useNavigate();
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center px-8">
      <div className="w-16 h-16 rounded-2xl bg-teal-50 flex items-center justify-center">
        {/* Simple document stack icon in teal */}
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <rect x="6" y="10" width="20" height="16" rx="3" fill="#0F6E56" opacity="0.15"/>
          <rect x="4" y="7" width="20" height="16" rx="3" fill="#0F6E56" opacity="0.25"/>
          <rect x="6" y="4" width="20" height="16" rx="3" fill="#0F6E56" opacity="0.9"/>
          <circle cx="24" cy="6" r="4" fill="#EF9F27"/>
          <path d="M22.5 6l1 1 2-2" stroke="white" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
          No candidates ranked yet
        </h1>
        <p className="text-gray-500 max-w-md leading-relaxed">
          Add a job description and upload your candidate files to run your 
          first AI-powered ranking pipeline. Results appear here in minutes.
        </p>
      </div>
      <div className="flex flex-col gap-3 w-full max-w-xs">
        <button
          onClick={() => navigate('/pipeline')}
          className="w-full py-3 px-6 bg-teal-700 text-white rounded-xl font-medium hover:bg-teal-800 transition-colors"
        >
          Run your first pipeline →
        </button>
        <button
          onClick={() => navigate('/pipeline?demo=true')}
          className="w-full py-3 px-6 border border-teal-700 text-teal-700 rounded-xl font-medium hover:bg-teal-50 transition-colors"
        >
          Load sample data
        </button>
      </div>
      <div className="flex gap-8 text-sm text-gray-400 mt-2">
        <span>Privacy-first — runs locally</span>
        <span>Powered by Llama 3.2</span>
        <span>Fairness audit included</span>
      </div>
    </div>
  );
}
```

### Update App.tsx

Find the route for "/" (DashboardPage).
Wrap it with a conditional: if no candidates exist in the API response, render EmptyStatePage instead.

The cleanest way: in DashboardPage.tsx, check the candidates array length after fetch.
If candidates.length === 0 AND pipeline has never run, render <EmptyStatePage /> inline.
Do NOT add a separate route — keep the "/" route as DashboardPage.

In DashboardPage.tsx, find the section that renders the candidates table.
Add above it:

```tsx
if (!isLoading && candidates.length === 0) {
  return <EmptyStatePage />;
}
```

Import EmptyStatePage at the top of DashboardPage.tsx.

---

## UPGRADE 4 — Fix ARCHITECTURE.md duplicate output/ entry

Open ARCHITECTURE.md.
Find the file structure section. There are two `output/` entries:
- One listing `output/__init__.py` and `output/writer.py` (the Python module)
- One listing `output/ranked_candidates.csv` and `output/bias_audit_report.json` (generated files)

Rename the Python module entry to clarify:

```
├── output/                        # Python output module
│   ├── __init__.py
│   └── writer.py                  # rank_candidates(), write_ranked_csv()
│
├── results/                       # Generated pipeline output (gitignored)
│   ├── ranked_candidates.csv      # Final ranked results
│   └── bias_audit_report.json     # Fairness audit report
```

If the actual output path in the code is `output/`, keep `output/` as the folder name but
add a comment clarifying the distinction. Do NOT refactor the actual paths — just fix the doc.

---

## UPGRADE 5 — Update methodology.md model section

Find the section in methodology.md that describes the LLM stack or technical choices.
It currently says something like "we use Llama 3.2 3b for all inference."

Replace with an accurate description of the per-agent model strategy:

```markdown
## Model selection strategy

Rather than a single model for all tasks, Distill assigns each agent the model 
best suited to its cognitive load:

| Agent | Model | Rationale |
|---|---|---|
| Resume + JD parser | qwen2.5:7b | Best open-source JSON adherence |
| Orchestrator | llama3.1:8b | Strong multi-step reasoning |
| Skills match | llama3.2:3b | RAG does heavy lifting; list comparison only |
| Trajectory scorer | deepseek-r1:7b | Reasoning model; infers growth from sparse data |
| Hiring manager | llama3.1:8b | Highest weight; needs nuanced judgment |
| Peer interviewer | qwen2.5:14b | Technical depth requires larger knowledge base |
| Devil's advocate | deepseek-r1:7b | Adversarial reasoning; chain-of-thought finds gaps |
| Narrative | llama3.2:3b | Short text generation; smallest capable model |

A hardware-aware LOW_MEMORY_MODE flag substitutes lighter models on machines 
with less than 20GB RAM, ensuring the pipeline runs on a wider range of hardware
without changing the scoring architecture.
```

---

## FINAL CHECKS — run these in order

```bash
# 1. Confirm config loads correctly
python -c "from config import get_model, OLLAMA_MODELS; print(OLLAMA_MODELS)"

# 2. Confirm all pipeline modules import without error
python -c "
from pipeline.ingest import ResumeParser, JdParser
from pipeline.enrich import TrajectoryEnricher  
from pipeline.embed import VectorStoreManager
from pipeline.score import CandidateScoringPipeline
from audit.counterfactual import CounterfactualAuditor
from output.writer import write_ranked_csv
print('All imports OK')
"

# 3. Run full test suite
pytest tests/ -v

# 4. Start FastAPI server and confirm it responds
uvicorn server:app --host 0.0.0.0 --port 8080 &
sleep 3
curl http://localhost:8080/api/candidates
# Expected: [] or existing results — not an error

# 5. Confirm frontend builds
cd frontend && npm run build
```

All 5 checks must pass before this upgrade is complete.
