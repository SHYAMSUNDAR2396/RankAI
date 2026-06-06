"""
RankAI – FastAPI Backend
Serves ranked-candidate data, bias-audit results, and pipeline control
to the React dashboard.
"""

from __future__ import annotations

import csv
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = OUTPUT_DIR / "ranked_candidates.csv"
AUDIT_PATH = OUTPUT_DIR / "bias_audit_report.json"
JOB_DESC_PATH = DATA_DIR / "sample_job_description.json"

# ── app ──────────────────────────────────────────────────────────────────
app = FastAPI(title="RankAI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ──────────────────────────────────────────────────────────────

# In-memory pipeline run status
_pipeline_status: dict[str, Any] = {"running": False, "last_result": None}


def _parse_csv() -> list[dict[str, Any]]:
    """Read ranked_candidates.csv and return a list of dicts."""
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            # Convert numeric fields
            for key in (
                "composite_score",
                "trajectory_score",
                "hiring_manager_score",
                "peer_interviewer_score",
                "devils_advocate_score",
                "panel_variance",
                "counterfactual_delta",
            ):
                val = row.get(key, "")
                try:
                    row[key] = float(val) if val else None  # type: ignore[assignment]
                except (ValueError, TypeError):
                    row[key] = None  # type: ignore[assignment]

            for key in ("rank",):
                val = row.get(key, "")
                try:
                    row[key] = int(val) if val else None  # type: ignore[assignment]
                except (ValueError, TypeError):
                    row[key] = None  # type: ignore[assignment]

            # Boolean fields
            for key in ("requires_human_review", "bias_flag"):
                val = row.get(key, "")
                row[key] = val.strip().lower() == "true" if val else False  # type: ignore[assignment]

            # List fields (pipe-separated)
            for key in ("strengths", "concerns"):
                val = row.get(key, "")
                row[key] = [s.strip() for s in val.split("|") if s.strip()] if val else []  # type: ignore[assignment]

            rows.append(row)
    return rows


def _load_audit() -> dict[str, Any]:
    """Read bias_audit_report.json."""
    if not AUDIT_PATH.exists():
        return {}
    with open(AUDIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_job_description() -> dict[str, Any]:
    """Read sample_job_description.json."""
    if not JOB_DESC_PATH.exists():
        return {}
    with open(JOB_DESC_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── routes ───────────────────────────────────────────────────────────────


@app.get("/api/candidates")
def get_candidates():
    """Return all ranked candidates sorted by composite_score descending."""
    rows = _parse_csv()
    rows.sort(key=lambda r: r.get("composite_score") or 0, reverse=True)
    return {"candidates": rows, "total": len(rows)}


@app.get("/api/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    """Return a single candidate by their candidate_id."""
    rows = _parse_csv()
    for row in rows:
        if row.get("candidate_id") == candidate_id:
            return row
    raise HTTPException(status_code=404, detail="Candidate not found")


@app.get("/api/audit")
def get_audit():
    """Return the bias audit report."""
    data = _load_audit()
    if not data:
        raise HTTPException(status_code=404, detail="Audit report not found")
    return data


@app.get("/api/job-description")
def get_job_description():
    """Return the job description."""
    data = _load_job_description()
    if not data:
        raise HTTPException(status_code=404, detail="Job description not found")
    return data


@app.get("/api/pipeline/status")
def pipeline_status():
    """Check whether a pipeline run is in progress."""
    return _pipeline_status


def _run_pipeline():
    """Execute main.py in a subprocess."""
    _pipeline_status["running"] = True
    _pipeline_status["last_result"] = None
    try:
        result = subprocess.run(
            [str(BASE_DIR / ".venv" / "bin" / "python"), str(BASE_DIR / "main.py")],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=600,
        )
        _pipeline_status["last_result"] = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }
    except Exception as exc:
        _pipeline_status["last_result"] = {"returncode": -1, "error": str(exc)}
    finally:
        _pipeline_status["running"] = False


@app.post("/api/run")
def run_pipeline(background_tasks: BackgroundTasks):
    """Trigger the scoring pipeline in the background."""
    if _pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    background_tasks.add_task(_run_pipeline)
    return {"message": "Pipeline started"}


@app.post("/api/upload/resumes")
async def upload_resumes(files: list[UploadFile] = File(...)):
    """Upload resume files (PDF/DOCX/JSON) to data/candidates/."""
    candidates_dir = DATA_DIR / "candidates"
    if candidates_dir.exists():
        import shutil
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for file in files:
        dest = candidates_dir / (file.filename or f"{uuid.uuid4()}.pdf")
        content = await file.read()
        dest.write_bytes(content)
        saved.append(str(dest.name))
    return {"uploaded": saved}


@app.post("/api/upload/job-description")
async def upload_job_description(file: UploadFile = File(...)):
    """Upload a job description file (JSON/TXT)."""
    dest = DATA_DIR / (file.filename or "job_description.json")
    content = await file.read()
    dest.write_bytes(content)
    return {"uploaded": str(dest.name)}


@app.get("/api/export/csv")
def export_csv():
    """Download the ranked_candidates.csv file."""
    if not CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(
        path=str(CSV_PATH),
        media_type="text/csv",
        filename="ranked_candidates.csv",
    )
