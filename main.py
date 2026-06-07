"""Command-line entry point for the Candidate Ranking System.

This module owns the orchestration layer described in the design: it parses CLI
arguments, configures logging, and performs startup verification before any
pipeline phase runs. Startup verification has two parts (Requirements 9.2, 9.3,
1.7, 1.8):

* :func:`check_ollama_reachable` confirms the locally hosted Ollama server is
  running and reachable within a connection timeout (default 10 seconds) by
  calling :func:`ollama.list`. On failure it prints clear setup instructions and
  exits with a non-zero status.
* :func:`check_models_present` confirms the configured LLM
  (``config.OLLAMA_MODEL``) is installed locally and performs a lightweight
  check that the embedding model (``config.EMBEDDING_MODEL``) is configured.
  When a required model is missing it names the model and exits *without*
  running any pipeline phase or writing artifacts.

The full pipeline orchestration (loading and parsing inputs, embedding, scoring,
auditing, and writing the ranked CSV / audit JSON) is implemented in
:func:`run_pipeline`. The heavy pipeline collaborators
(``ResumeParser``/``JdParser``, ``TrajectoryEnricher``, ``VectorStoreManager``,
``CandidateScoringPipeline``, ``CounterfactualAuditor``) are imported lazily
*inside* :func:`run_pipeline` so that simply importing this module for the
startup/arg-parsing unit tests does not require those modules' heavy optional
dependencies (``chromadb``, ``sentence_transformers``, ``pydantic``) to be
installed. The rich progress bar, top-N table, and summary are layered onto this
same module by task 13.3; for now phase/per-candidate progress is emitted via
``logging``.

All diagnostic output is routed through the :mod:`logging` library
(Requirement 13.4). The single exception is the fatal Ollama setup-instructions
message, which is also written to ``stderr`` so it remains visible regardless of
logging configuration before exiting.

Testability note: ``ollama`` is imported at module level so that
``ollama.list`` (and ``ollama.chat`` elsewhere) can be mocked with
``unittest.mock.patch`` in the offline test suite (Requirement 12.5).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ollama

import config

if TYPE_CHECKING:
    # Imported only for type checking so importing ``main`` for the
    # startup/arg-parsing unit tests stays light: the pipeline classes pull in
    # heavy optional dependencies (``chromadb``/``sentence_transformers``/
    # ``pydantic``) that are imported lazily inside :func:`run_pipeline`.
    from models.candidate import CandidateProfile
    from pipeline.ingest import ResumeParser

logger = logging.getLogger(__name__)

#: Exit code used for every fatal startup failure (unreachable server or a
#: missing required model). Chosen so the CLI signals failure to the shell.
_EXIT_FAILURE = 1

#: Candidate resume file extensions the CLI loads from the candidates directory
#: (matched case-insensitively) (Requirement 9.4).
_CANDIDATE_EXTENSIONS = frozenset({".json", ".pdf", ".docx"})

#: Sample candidate dataset used when the candidates directory is missing or
#: contains no candidate files (Requirement 9.5). It is an ARRAY of candidate
#: objects, so it is loaded and validated directly rather than through
#: ``ResumeParser.parse_file`` (which treats a ``.json`` file as a single
#: profile).
_SAMPLE_CANDIDATES_PATH = "data/sample_candidates.json"

#: Human-readable setup instructions shown when the Ollama server cannot be
#: reached. Surfaced both via ``logging.error`` and ``stderr`` because this is a
#: fatal, user-facing condition that must be visible before the process exits.
_OLLAMA_SETUP_INSTRUCTIONS = (
    "Could not reach the local Ollama server.\n"
    "The Candidate Ranking System needs a running Ollama server for all LLM "
    "inference.\n"
    "To set it up:\n"
    "  1. Install Ollama from https://ollama.com/download\n"
    "  2. Start the server in a separate terminal:  ollama serve\n"
    "  3. Download the required models (see README.md)\n"
    f"The server is expected at: {config.OLLAMA_HOST}"
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the command-line interface.

    Defines the five flags required by the CLI (Requirement 9.1):
    ``--candidates-dir``, ``--job-description``, ``--output-dir``,
    ``--skip-audit``, and ``--verbose``. Each argument carries help text and the
    path-style arguments carry sensible defaults so the bundled sample data runs
    out of the box.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: A configured parser ready to parse the CLI
        arguments.
    """
    parser = argparse.ArgumentParser(
        prog="candidate-ranking-system",
        description=(
            "Rank candidates against a job description using a local, "
            "open-source stack (Ollama LLM + sentence-transformers embeddings + "
            "ChromaDB). Produces a ranked CSV and a counterfactual bias audit "
            "report."
        ),
    )
    parser.add_argument(
        "--candidates-dir",
        default="./data/candidates/",
        help=(
            "Directory of candidate resume files (.json, .pdf, .docx). Falls "
            "back to data/sample_candidates.json when the directory is missing "
            "or empty. Default: ./data/candidates/"
        ),
    )
    parser.add_argument(
        "--job-description",
        default="./data/sample_job_description.json",
        help=(
            "Path to the job description file (.json or .txt). "
            "Default: ./data/sample_job_description.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./output/",
        help=(
            "Directory where ranked_candidates.csv and bias_audit_report.json "
            "are written. Default: ./output/"
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help=(
            "Run the pipeline without the counterfactual fairness audit. A "
            "bias_audit_report.json documenting that no analysis was performed "
            "is still written."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level log output. Without this flag, logs at INFO and above.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse. When ``None`` (the default),
            :data:`sys.argv` is used, which is the normal behavior when run as a
            program. Passing an explicit list is primarily useful for tests.

    Returns:
        argparse.Namespace: The parsed arguments with attributes
        ``candidates_dir``, ``job_description``, ``output_dir``, ``skip_audit``,
        and ``verbose``.
    """
    return build_arg_parser().parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Configure the root logger for the run.

    Routes all diagnostic output through the :mod:`logging` library
    (Requirement 13.4). The level is DEBUG when ``verbose`` is true and INFO
    otherwise (Requirements 9.11, 9.12). ``force=True`` is used so the
    configuration is applied even if logging was already configured earlier in
    the process (e.g. across repeated invocations in tests).

    Args:
        verbose: When ``True``, set the level to DEBUG; otherwise set it to
            INFO.

    Returns:
        None.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _extract_model_names(list_response: Any) -> list[str]:
    """Extract the locally available model names from an ``ollama.list`` result.

    The Ollama SDK has returned a few response shapes across versions: a mapping
    with a ``"models"`` key whose entries are mappings keyed by ``"model"`` or
    ``"name"``, and object-style responses exposing a ``models`` attribute whose
    entries expose a ``model`` (or ``name``) attribute. Both shapes are handled
    defensively so model-presence verification does not depend on the exact SDK
    version.

    Args:
        list_response: The value returned by :func:`ollama.list`.

    Returns:
        list[str]: The names of the locally installed models. Empty when no
        models are present or the response shape is unrecognized.
    """
    if isinstance(list_response, dict):
        models = list_response.get("models", [])
    else:
        models = getattr(list_response, "models", []) or []

    names: list[str] = []
    for entry in models:
        name: Any = None
        if isinstance(entry, dict):
            name = entry.get("model") or entry.get("name")
        else:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name:
            names.append(str(name))
    return names


def check_ollama_reachable(timeout: float = 10.0) -> None:
    """Verify the local Ollama server is running and reachable.

    Calls :func:`ollama.list` and treats any raised exception (connection
    refused, DNS failure, etc.) or a timeout as "unreachable" (Requirement 9.2).
    The call is run on a worker thread and bounded by ``timeout`` seconds, which
    is the documented 10-second connection timeout from the requirements; if the
    call does not return in time it is treated as a failure. On any failure,
    clear setup instructions (how to install Ollama, run ``ollama serve``, and
    ``ollama pull``) are logged and written to ``stderr`` and the process exits
    with a non-zero status (Requirement 9.3).

    Args:
        timeout: Maximum number of seconds to wait for the reachability check
            before treating the server as unreachable. Defaults to 10 seconds.

    Returns:
        None: Returns normally only when the server is reachable.

    Raises:
        SystemExit: Always raised (via :func:`sys.exit`) with a non-zero status
            when the server cannot be reached.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ollama.list)
            future.result(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - any failure means "unreachable"
        logger.error("%s", _OLLAMA_SETUP_INSTRUCTIONS)
        logger.debug("Ollama reachability check failed: %s", exc)
        print(_OLLAMA_SETUP_INSTRUCTIONS, file=sys.stderr)
        sys.exit(_EXIT_FAILURE)
    logger.info("Ollama server is reachable at %s", config.OLLAMA_HOST)


def check_models_present() -> None:
    """Verify the required local models are present before running the pipeline.

    The primary hard check is the LLM: ``config.OLLAMA_MODEL`` must appear in the
    list of locally installed models reported by :func:`ollama.list`
    (Requirements 1.7, 1.8). If it is missing, an error naming the missing model
    is logged and the process exits without running any pipeline phase or writing
    any artifact.

    The embedding model (``config.EMBEDDING_MODEL``) is verified only lightly:
    a full check would construct a :class:`~sentence_transformers.SentenceTransformer`,
    which is heavy and may download weights. Instead this confirms the model name
    is configured (non-empty) and that ``sentence_transformers`` is importable;
    the weights are downloaded on first use during the EMBED phase, so a clean
    machine will fetch them then rather than failing here.

    Args:
        None.

    Returns:
        None: Returns normally only when the required models are present.

    Raises:
        SystemExit: Always raised (via :func:`sys.exit`) with a non-zero status
            when a required model is missing.
    """
    missing: list[str] = []

    # Primary hard check: the required Ollama LLMs must be installed locally.
    required_models = set(
        (config.OLLAMA_MODELS_LOW_MEM if config.LOW_MEMORY_MODE else config.OLLAMA_MODELS).values()
    )
    available = _extract_model_names(ollama.list())
    for r_model in required_models:
        if r_model not in available:
            missing.append(f"LLM '{r_model}' (install with: ollama pull {r_model})")

    # Lightweight embedding-model check: confirm a name is configured and that
    # sentence_transformers is importable. The weights download on first use.
    if not config.EMBEDDING_MODEL:
        missing.append("Embedding model (config.EMBEDDING_MODEL is not set)")
    else:
        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is None:
                missing.append(
                    "sentence-transformers package (install with: "
                    "pip install sentence-transformers)"
                )
            else:
                logger.debug(
                    "Embedding model '%s' is configured; weights download on "
                    "first use.",
                    config.EMBEDDING_MODEL,
                )
        except Exception as exc:  # noqa: BLE001 - import probing must not crash startup
            logger.debug("Could not probe sentence_transformers: %s", exc)

    if missing:
        logger.error(
            "Required model(s) not available locally; aborting before any "
            "pipeline phase: %s",
            "; ".join(missing),
        )
        print(
            "ERROR: required model(s) not available locally:\n  - "
            + "\n  - ".join(missing),
            file=sys.stderr,
        )
        sys.exit(_EXIT_FAILURE)

    required_llms = ", ".join(
        set((config.OLLAMA_MODELS_LOW_MEM if config.LOW_MEMORY_MODE else config.OLLAMA_MODELS).values())
    )
    logger.info(
        "Required models present (LLMs: %s, embedding model: '%s').",
        required_llms,
        config.EMBEDDING_MODEL,
    )


def verify_startup(timeout: float = 10.0) -> None:
    """Run all startup verification checks in order.

    When LLM_BACKEND is "groq", it confirms the GROQ_API_KEY environment
    variable is set and validates sentence-transformers package installation.
    Otherwise, it confirms the Ollama server is reachable (:func:`check_ollama_reachable`),
    then confirms the required models are installed locally
    (:func:`check_models_present`). Either check exits the process on failure, so
    returning normally means the pipeline is safe to run (Requirements 9.2, 9.3,
    1.7, 1.8).

    Args:
        timeout: Connection timeout in seconds forwarded to the reachability
            check. Defaults to 10 seconds.

    Returns:
        None: Returns normally only when every startup check passes.

    Raises:
        SystemExit: Propagated from a failing check when the server is
            unreachable or a required model is missing.
    """
    if config.LLM_BACKEND == "groq":
        import os
        api_key = config.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY is not set. A Groq API key is required when LLM_BACKEND is 'groq'.")
            print("ERROR: GROQ_API_KEY is not set.", file=sys.stderr)
            sys.exit(_EXIT_FAILURE)

        if not config.EMBEDDING_MODEL:
            logger.error("Embedding model (config.EMBEDDING_MODEL is not set)")
            print("ERROR: config.EMBEDDING_MODEL is not set", file=sys.stderr)
            sys.exit(_EXIT_FAILURE)

        try:
            import importlib.util
            if importlib.util.find_spec("sentence_transformers") is None:
                logger.error("sentence-transformers package is missing.")
                print("ERROR: sentence-transformers package is missing.", file=sys.stderr)
                sys.exit(_EXIT_FAILURE)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not probe sentence_transformers: %s", exc)

        logger.info("Startup check passed for Groq backend.")
    else:
        check_ollama_reachable(timeout=timeout)
        check_models_present()


def _load_candidates_from_dir(
    candidates_dir: Path, parser_resume: "ResumeParser"
) -> list["CandidateProfile"]:
    """Parse every supported candidate file in ``candidates_dir``.

    Scans ``candidates_dir`` for files whose (case-insensitive) suffix is one of
    ``.json``/``.pdf``/``.docx`` (Requirement 9.4) and parses each through
    ``parser_resume.parse_file``. A ``.json`` file in the candidates directory is
    treated as a *single* profile per file (``parse_file`` handles that), unlike
    the sample fallback which is an array. Files that the parser skips (returning
    ``None``) and per-file errors are contained so one bad resume does not abort
    the scan; the failure is logged and the remaining files continue.

    Args:
        candidates_dir: The directory to scan for candidate resume files.
        parser_resume: The :class:`~pipeline.ingest.ResumeParser` used to parse
            each file.

    Returns:
        list[CandidateProfile]: The successfully parsed, non-``None`` profiles,
        ordered by sorted filename for deterministic processing.
    """
    candidate_files = sorted(
        path
        for path in candidates_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _CANDIDATE_EXTENSIONS
    )

    profiles: list["CandidateProfile"] = []
    for path in candidate_files:
        try:
            profile = parser_resume.parse_file(path)
        except Exception as exc:  # noqa: BLE001 - contain per-file parse failures
            logger.warning("Failed to parse candidate file %s: %s", path, exc)
            continue
        if profile is not None:
            profiles.append(profile)
    return profiles


def _load_sample_candidates() -> list["CandidateProfile"]:
    """Load and validate the bundled sample candidate dataset (Requirement 9.5).

    The sample dataset at :data:`_SAMPLE_CANDIDATES_PATH` is a JSON *array* of
    candidate objects (not a single profile), so it is loaded with :mod:`json`
    and each element is validated directly into a
    :class:`~models.candidate.CandidateProfile` rather than being run through
    ``ResumeParser.parse_file``. An element that fails validation is logged and
    skipped so a single malformed entry does not abort the fallback.

    Args:
        None.

    Returns:
        list[CandidateProfile]: The validated sample profiles. Empty when the
        sample file is missing, unreadable, or not a JSON array.
    """
    # Local import: keep the model (and its ``pydantic`` dependency) out of the
    # module import path used by the lightweight startup/arg-parsing tests.
    from models.candidate import CandidateProfile

    sample_path = Path(_SAMPLE_CANDIDATES_PATH)
    try:
        with sample_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not load sample candidates from %s: %s", sample_path, exc)
        return []

    if not isinstance(data, list):
        logger.error(
            "Sample candidates file %s is not a JSON array; got %s",
            sample_path,
            type(data).__name__,
        )
        return []

    profiles: list["CandidateProfile"] = []
    for index, obj in enumerate(data):
        if not isinstance(obj, dict):
            logger.warning(
                "Skipping sample candidate at index %d: not an object", index
            )
            continue
        try:
            profiles.append(CandidateProfile(**obj))
        except Exception as exc:  # noqa: BLE001 - skip a single invalid entry
            logger.warning(
                "Skipping invalid sample candidate at index %d: %s", index, exc
            )
    return profiles


def _load_candidates(
    candidates_dir: Path, parser_resume: "ResumeParser"
) -> list["CandidateProfile"]:
    """Load candidate profiles from the directory, falling back to sample data.

    When ``candidates_dir`` exists, is a directory, and yields at least one
    parsed candidate file, those profiles are returned (Requirement 9.4).
    Otherwise -- the directory is missing, is not a directory, or contains no
    parseable candidate files -- the bundled ``data/sample_candidates.json``
    array is loaded and validated as the fallback dataset (Requirement 9.5).

    Args:
        candidates_dir: The candidates directory supplied on the command line.
        parser_resume: The :class:`~pipeline.ingest.ResumeParser` used to parse
            directory files.

    Returns:
        list[CandidateProfile]: The loaded candidate profiles, from the
        directory when available, otherwise from the sample dataset.
    """
    if candidates_dir.is_dir():
        profiles = _load_candidates_from_dir(candidates_dir, parser_resume)
        if profiles:
            logger.info(
                "Loaded %d candidate(s) from %s", len(profiles), candidates_dir
            )
            return profiles
        logger.info(
            "No candidate files found in %s; falling back to sample data %s",
            candidates_dir,
            _SAMPLE_CANDIDATES_PATH,
        )
    else:
        logger.info(
            "Candidates directory %s does not exist; falling back to sample "
            "data %s",
            candidates_dir,
            _SAMPLE_CANDIDATES_PATH,
        )

    profiles = _load_sample_candidates()
    logger.info(
        "Loaded %d candidate(s) from sample data %s",
        len(profiles),
        _SAMPLE_CANDIDATES_PATH,
    )
    return profiles


#: Number of top-ranked candidates rendered in the results table when more
#: than this many candidates were scored (Requirement 9.8). Fewer scored
#: candidates means every candidate is shown.
_TOP_N = 10

#: rich style names aligned with the RankAI dashboard design system: the
#: dashboard primary is a deep teal used for headers/key highlights, amber marks
#: "Require Human Review", and red is reserved strictly for bias flags.
_STYLE_HEADER = "bold #0F6E56"  # dashboard primary teal
_STYLE_REVIEW = "bold yellow"  # amber for requires_human_review = True
_STYLE_BIAS = "bold red"  # red strictly for bias_flag = True
_STYLE_MUTED = "dim"  # muted "No"/"Clean" markers
_STYLE_CLEAN = "green"  # clean (no bias) marker


def _present_results(
    results: list[dict],
    output_dir: Path,
    csv_path: Path,
    report_path: Path,
    elapsed_seconds: float,
) -> None:
    """Render the top-N results table and the run summary to the console.

    Renders, using :mod:`rich`, the two user-facing artifacts task 13.3 layers
    onto the run (Requirements 9.8, 9.9):

    * A table of the top :data:`_TOP_N` candidates by ``composite_score`` (or all
      candidates when fewer than ``_TOP_N`` were scored), ranked via
      :func:`output.writer.rank_candidates`. The table shows ``rank``, ``name``,
      ``composite_score``, ``verdict_consensus`` (derived per row via
      :func:`output.writer.verdict_consensus`), ``requires_human_review``, and
      ``bias_flag``. Cell styling follows the RankAI dashboard design system: a
      teal header, amber for a ``requires_human_review`` "Yes", and red for a
      ``bias_flag`` flag, with muted/green markers otherwise (Requirement 9.8).
    * A summary panel with the total candidate count, the elapsed time in
      seconds, and the output file paths (the ranked CSV and the bias audit
      report) (Requirement 9.9).

    ``rich`` is imported lazily here so importing this module for the
    startup/arg-parsing unit tests does not require ``rich`` to be installed.

    Args:
        results: The scored candidate result dicts (with ``bias_flag`` populated
            by the audit phase).
        output_dir: The directory the artifacts were written into.
        csv_path: The path to the written ``ranked_candidates.csv``.
        report_path: The path to the written ``bias_audit_report.json``.
        elapsed_seconds: Wall-clock seconds elapsed for the full run, shown in
            the summary.

    Returns:
        None.
    """
    # Lazy rich imports keep ``import main`` light for the startup-only tests.
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from output.writer import rank_candidates, verdict_consensus

    console = Console()

    # Rank (descending composite, candidate_id tie-break) and take the top N, or
    # all candidates when fewer than N were scored (Requirement 9.8).
    ranked = rank_candidates(results)
    top = ranked[:_TOP_N]

    shown = len(top)
    title = (
        f"Top {shown} Candidates"
        if len(ranked) > _TOP_N
        else f"Ranked Candidates ({shown})"
    )
    table = Table(title=title, header_style=_STYLE_HEADER, title_style=_STYLE_HEADER)
    table.add_column("Rank", justify="right")
    table.add_column("Name")
    table.add_column("Composite", justify="right")
    table.add_column("Verdict")
    table.add_column("Human Review", justify="center")
    table.add_column("Bias Flag", justify="center")

    for result in top:
        requires_review = bool(result.get("requires_human_review", False))
        review_cell = (
            f"[{_STYLE_REVIEW}]Yes[/{_STYLE_REVIEW}]"
            if requires_review
            else f"[{_STYLE_MUTED}]No[/{_STYLE_MUTED}]"
        )
        biased = bool(result.get("bias_flag", False))
        bias_cell = (
            f"[{_STYLE_BIAS}]\u26a0 FLAG[/{_STYLE_BIAS}]"
            if biased
            else f"[{_STYLE_CLEAN}]Clean[/{_STYLE_CLEAN}]"
        )
        table.add_row(
            str(result["rank"]),
            str(result["name"]),
            f"{result['composite_score']:.2f}",
            str(verdict_consensus(result["persona_verdicts"])),
            review_cell,
            bias_cell,
        )

    console.print(table)

    # Summary: total candidates, elapsed seconds, and output paths (Req 9.9).
    summary = (
        f"[{_STYLE_HEADER}]Run complete[/{_STYLE_HEADER}]\n"
        f"Candidates scored: {len(results)}\n"
        f"Elapsed: {elapsed_seconds:.2f}s\n"
        f"Ranked CSV: {csv_path}\n"
        f"Bias audit report: {report_path}\n"
        f"Output directory: {output_dir}"
    )
    console.print(Panel(summary, title="Summary", border_style=_STYLE_HEADER))

    logger.info(
        "Run complete: %d candidate(s) scored in %.2fs; ranked CSV at %s; audit "
        "report at %s (output dir %s)",
        len(results),
        elapsed_seconds,
        csv_path,
        report_path,
        output_dir,
    )


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the candidate-ranking pipeline end to end.

    Orchestrates all five phases in sequence (Requirements 9.4, 9.5, 9.6, 9.10,
    13.4, 13.5):

    1. **INGEST** -- parse the job description and load candidate profiles from
       ``--candidates-dir``, falling back to ``data/sample_candidates.json`` when
       the directory is missing or empty (Requirements 9.4, 9.5).
    2. **ENRICH & EMBED** -- embed the job description and the calibration
       examples *before* scoring, then enrich and embed each candidate
       (Requirement 9.6). Enrichment runs before scoring because the scorer reads
       each candidate's trajectory vector.
    3. **SCORE** -- score every enriched candidate through the persona panel.
    4. **AUDIT** -- unless ``--skip-audit`` is set, build and re-score a
       counterfactual twin per candidate to flag potential bias, then write the
       fairness report (when skipped, a report documenting that no analysis was
       performed is still written) (Requirement 9.10).
    5. **OUTPUT** -- write the ranked ``ranked_candidates.csv``.

    All diagnostic and progress output is routed through :mod:`logging`, with
    INFO logs at each phase boundary and per candidate (Requirements 13.4, 13.5).
    The candidate SCORE loop is additionally wrapped in a :mod:`rich` progress
    bar showing candidates processed out of the total (Requirement 9.7), and on
    completion the top-N results table and an elapsed-time summary are rendered
    (Requirements 9.8, 9.9). Per-candidate enrich/embed/score work is wrapped so
    a single bad candidate is logged and skipped rather than aborting the run.
    The heavy pipeline collaborators (and ``rich``) are imported locally so
    importing this module stays light for the startup-only unit tests.

    Args:
        args: The parsed command-line arguments controlling the run
            (``candidates_dir``, ``job_description``, ``output_dir``,
            ``skip_audit``, ``verbose``).

    Returns:
        int: A process exit code; ``0`` indicates success.
    """
    # Local imports: these modules pull in heavy optional dependencies, so they
    # are imported here (not at module load) to keep ``import main`` light for
    # the startup/arg-parsing unit tests.
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    from audit.counterfactual import CounterfactualAuditor, reset_audit_log
    from output.writer import write_ranked_csv
    from pipeline.embed import VectorStoreManager
    from pipeline.enrich import TrajectoryEnricher
    from pipeline.ingest import JdParser, ResumeParser
    from pipeline.score import CandidateScoringPipeline
    from utils.ollama_client import OllamaClient

    # Wall-clock start used for the elapsed-time summary (Requirement 9.9).
    start_time = time.perf_counter()

    output_dir = Path(args.output_dir)

    # Build the shared collaborators once and reuse them across every phase so
    # the audit re-scores twins against the same scorer and vector store.
    ollama_client = OllamaClient()
    parser_resume = ResumeParser(ollama_client)
    parser_jd = JdParser(ollama_client)
    enricher = TrajectoryEnricher(ollama_client)
    store = VectorStoreManager(persist_dir=config.CHROMA_PERSIST_DIR)
    scorer = CandidateScoringPipeline(ollama_client, store)
    auditor = CounterfactualAuditor(parser_resume, enricher, scorer)

    # -- INGEST -------------------------------------------------------------
    logger.info("Phase INGEST: begin")
    jd = parser_jd.parse_file(Path(args.job_description))
    logger.info(
        "Parsed job description %r with %d requirement(s)",
        jd.title,
        len(jd.requirements),
    )

    profiles = _load_candidates(Path(args.candidates_dir), parser_resume)
    logger.info("Phase INGEST: end (%d candidate(s) loaded)", len(profiles))

    # -- EMBED (JD + calibration first, then candidates) --------------------
    logger.info("Phase EMBED: begin")
    store.embed_job_description(jd)
    store.embed_calibration_examples(config.CALIBRATION_EXAMPLES)
    logger.info("Embedded job description and %d calibration example(s)", len(config.CALIBRATION_EXAMPLES))

    # Enrich + embed each candidate. Enrichment must precede scoring because the
    # scorer reads each candidate's trajectory vector. Candidate embedding order
    # does not affect scoring correctness (scoring retrieves JD/calibration
    # context, not other candidates).
    enriched_profiles: list["CandidateProfile"] = []
    for index, profile in enumerate(profiles, start=1):
        try:
            enricher.enrich(profile)
            store.embed_candidate(profile)
        except Exception as exc:  # noqa: BLE001 - skip a single bad candidate
            logger.warning(
                "Skipping candidate %s (%s) after enrich/embed failure: %s",
                profile.candidate_id,
                profile.name,
                exc,
            )
            continue
        enriched_profiles.append(profile)
        logger.info(
            "Enriched and embedded candidate %d/%d: %s",
            index,
            len(profiles),
            profile.name,
        )
    logger.info("Phase EMBED: end (%d candidate(s) embedded)", len(enriched_profiles))

    # -- SCORE --------------------------------------------------------------
    logger.info("Phase SCORE: begin")
    scored: list[tuple["CandidateProfile", dict]] = []
    # Wrap the scoring loop in a rich progress bar showing candidates processed
    # out of the total (Requirement 9.7). Per-candidate try/except resilience is
    # preserved so a single bad candidate is logged and skipped.
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        score_task = progress.add_task(
            "Scoring candidates", total=len(enriched_profiles)
        )
        for index, profile in enumerate(enriched_profiles, start=1):
            try:
                result = scorer.score(profile)
            except Exception as exc:  # noqa: BLE001 - skip a single bad candidate
                logger.warning(
                    "Skipping candidate %s (%s) after scoring failure: %s",
                    profile.candidate_id,
                    profile.name,
                    exc,
                )
                progress.advance(score_task)
                continue
            scored.append((profile, result))
            logger.info(
                "Scored candidate %d/%d: %s (composite=%.2f)",
                index,
                len(enriched_profiles),
                profile.name,
                result["composite_score"],
            )
            progress.advance(score_task)
    results = [result for _, result in scored]
    logger.info("Phase SCORE: end (%d candidate(s) scored)", len(results))

    # -- AUDIT --------------------------------------------------------------
    logger.info("Phase AUDIT: begin (skip_audit=%s)", args.skip_audit)
    # Reset the shared audit log so repeated runs start from a clean slate.
    reset_audit_log()
    if not args.skip_audit:
        for index, (profile, result) in enumerate(scored, start=1):
            try:
                auditor.audit(profile, result)
            except Exception as exc:  # noqa: BLE001 - contain per-candidate audit failures
                logger.warning(
                    "Audit raised for candidate %s (%s); continuing: %s",
                    profile.candidate_id,
                    profile.name,
                    exc,
                )
                continue
            logger.info(
                "Audited candidate %d/%d: %s", index, len(scored), profile.name
            )
    else:
        logger.info("Counterfactual fairness audit skipped by --skip-audit")
    report_path = auditor.write_report(output_dir, skipped=args.skip_audit)
    logger.info("Phase AUDIT: end (report at %s)", report_path)

    # -- OUTPUT -------------------------------------------------------------
    logger.info("Phase OUTPUT: begin")
    csv_path = write_ranked_csv(results, output_dir)
    logger.info("Phase OUTPUT: end (ranked CSV at %s)", csv_path)

    elapsed_seconds = time.perf_counter() - start_time
    _present_results(results, output_dir, csv_path, report_path, elapsed_seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, configure logging, verify startup, and run the pipeline.

    This is the CLI entry point. It parses arguments, configures logging based on
    the ``--verbose`` flag, performs startup verification (which exits the
    process on a fatal condition), and then delegates to :func:`run_pipeline`.

    Args:
        argv: Argument list to parse. When ``None`` (the default),
            :data:`sys.argv` is used. Passing a list is useful for tests.

    Returns:
        int: The process exit code returned by :func:`run_pipeline`.
    """
    args = parse_args(argv)
    configure_logging(args.verbose)
    verify_startup()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
