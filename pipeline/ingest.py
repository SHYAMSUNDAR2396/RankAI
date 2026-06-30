"""INGEST phase: parse resumes (and, later, the job description) into models.

This module owns the INGEST phase of the Candidate Ranking System. It converts
heterogeneous source files into validated Pydantic v2 models that every later
phase (ENRICH, EMBED & STORE, SCORE, AUDIT) can rely on.

It currently hosts :class:`ResumeParser`, which turns a single resume file into
a validated :class:`~models.candidate.CandidateProfile`:

* ``.pdf``  -> raw text via PyMuPDF, then an LLM extraction (Requirement 2.1).
* ``.docx`` -> raw text via python-docx, then an LLM extraction (Requirement 2.2).
* ``.json`` -> a structured profile loaded directly, no LLM (Requirement 2.4).
* anything else -> a logged warning and a skip (Requirement 2.8).

The parser is built to be safe to call in a loop: every per-file failure mode
(unsupported extension, empty/whitespace text, structurally invalid JSON, or a
profile that fails validation twice) results in a logged warning and a ``None``
return rather than a raised exception, so the caller can continue with the
remaining files (Requirements 2.8, 2.11, 2.12).

``JdParser`` (the job-description parser) lives in this same module. Unlike the
resilient, skip-and-continue resume flow, the JD parser is fail-fast: an
unsupported extension or a corrupt/invalid ``.json`` job description raises
rather than returning a placeholder (Requirements 3.3, 3.4). A ``.txt`` job
description is classified by the LLM, and malformed classification output falls
back to a still-valid :class:`~models.job.JobDescription` (Requirement 3.9).

Heavy or optional third-party dependencies (PyMuPDF/``fitz``, python-docx, and
spaCy) are imported lazily inside the methods/helpers that use them, so simply
importing :mod:`pipeline.ingest` does not require those packages to be installed
(Requirement 1.10). All tunable values are read from :mod:`config`; this module
redefines no cross-cutting constants (Requirement 10.5).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

import config
from models.candidate import CandidateProfile, CandidateRole
from models.job import JobDescription, JobRequirement
from utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


#: Exact system prompt that instructs the LLM to emit a structured profile as a
#: single JSON object. Kept verbatim so the extraction schema the model targets
#: matches the :class:`~models.candidate.CandidateProfile` field names
#: (Requirement 2.3).
EXTRACTION_SYSTEM_PROMPT: str = (
    "You are a resume parser. Extract structured data from the resume text. "
    "Return ONLY a valid JSON object. No preamble. No markdown. No explanation. "
    "Use this exact schema: {name, email, years_experience, roles: [{title, "
    "company, company_size_estimate, start_date, end_date, duration_months, "
    "scope_keywords}], skills_claimed, education: [{institution, degree, year}], "
    "raw_text}"
)

#: Instruction appended to the system prompt on the single correction retry that
#: follows a validation failure (Requirement 2.10).
CORRECTION_INSTRUCTION: str = (
    " Your previous output failed validation. Return ONLY valid JSON matching "
    "the schema exactly."
)

#: Simple email matcher used by the spaCy fallback to recover an email address
#: from raw resume text (Requirement 2.6).
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


#: Process-wide cache for the lazily loaded spaCy pipeline. ``_NLP_LOADED``
#: records that a load was attempted (so a missing model is not retried on every
#: call) and ``_NLP`` holds the loaded pipeline or ``None`` when unavailable.
_NLP: Any = None
_NLP_LOADED: bool = False


def _get_nlp() -> Any:
    """Return the process-wide spaCy ``en_core_web_sm`` pipeline, or ``None``.

    The model is imported and loaded lazily on first use and cached at module
    level so the (heavy) pipeline is loaded at most once and never at import
    time. If spaCy or the ``en_core_web_sm`` model is not installed, the failure
    is caught and logged once and ``None`` is returned, so the name/email
    fallback degrades gracefully instead of crashing (Requirement 2.6).

    Args:
        None.

    Returns:
        The loaded spaCy ``Language`` pipeline, or ``None`` when spaCy or the
        ``en_core_web_sm`` model could not be loaded.
    """
    global _NLP, _NLP_LOADED
    if _NLP_LOADED:
        return _NLP
    _NLP_LOADED = True
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    except Exception as exc:  # noqa: BLE001 - any load failure -> graceful fallback
        logger.warning(
            "spaCy en_core_web_sm unavailable; name/email fallback disabled: %s",
            exc,
        )
        _NLP = None
    return _NLP


def _strip_none(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``mapping`` with all ``None``-valued keys removed.

    Dropping ``None`` values lets a missing optional field fall back to its
    defined model default rather than being explicitly set to ``None`` (which
    would fail validation for non-optional typed fields) (Requirement 2.7).

    Args:
        mapping: The dict to filter.

    Returns:
        A new dict containing only the entries whose value is not ``None``.
    """
    return {key: value for key, value in mapping.items() if value is not None}


class ResumeParser:
    """Convert a resume file into a validated :class:`CandidateProfile`.

    The parser dispatches on file extension (case-insensitive). Text-bearing
    formats (``.pdf``/``.docx``) have their raw text extracted and then handed to
    an LLM (via :class:`~utils.ollama_client.OllamaClient`) for structured
    extraction; ``.json`` resumes are loaded directly without any LLM call. Every
    per-file error is contained: the method logs a warning and returns ``None``
    rather than raising, so a caller iterating over a directory can continue with
    the remaining files (Requirements 2.8, 2.11, 2.12).
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """Initialize the parser.

        Args:
            ollama_client: The shared LLM wrapper used to extract a structured
                profile from raw resume text. The same client is reused for the
                correction retry.

        Returns:
            None.
        """
        self.ollama_client = ollama_client

    def parse_file(self, path: Path) -> CandidateProfile | None:
        """Parse one resume file, dispatching on its extension.

        Dispatch (case-insensitive):
            * ``.pdf``  -> extract text with PyMuPDF, then :meth:`parse_text`.
            * ``.docx`` -> extract text with python-docx, then :meth:`parse_text`.
            * ``.json`` -> load the structured profile directly (no LLM).
            * other      -> log a warning and skip (Requirement 2.8).

        For ``.pdf``/``.docx``, if the extracted text is empty or whitespace-only
        the file is skipped with a warning (Requirement 2.12). The method never
        raises on a per-file error; it returns ``None`` so the caller can
        continue with the remaining files.

        Args:
            path: Filesystem path to the resume file.

        Returns:
            A validated :class:`CandidateProfile`, or ``None`` when the file is
            skipped for any reason.
        """
        extension = path.suffix.lower()

        if extension == ".json":
            return self._parse_json_file(path)

        if extension == ".pdf":
            raw_text = self._extract_pdf_text(path)
        elif extension == ".docx":
            raw_text = self._extract_docx_text(path)
        else:
            logger.warning(
                "Skipping resume with unsupported extension %r: %s",
                extension,
                path,
            )
            return None

        if raw_text is None or not raw_text.strip():
            logger.warning(
                "Skipping resume with empty or unreadable text: %s", path
            )
            return None

        return self.parse_text(raw_text, path)

    def parse_text(
        self, raw_text: str, source_path: Path
    ) -> CandidateProfile | None:
        """Extract a validated profile from raw resume text via the LLM.

        Exposed separately from :meth:`parse_file` so the counterfactual audit
        can re-parse a swapped resume body without touching the filesystem. The
        method calls the LLM for a structured profile, builds and validates a
        :class:`CandidateProfile`, and on a validation failure retries the
        extraction exactly once with a correction prompt before giving up
        (Requirements 2.3, 2.9, 2.10, 2.11).

        Args:
            raw_text: The resume text to extract a profile from.
            source_path: Path recorded on the profile's ``source_file`` field.
                The original ``raw_text`` (not the LLM's echoed value) is always
                preserved on the profile so the audit phase can re-parse it.

        Returns:
            A validated :class:`CandidateProfile` with ``is_complete`` set, or
            ``None`` when extraction fails validation twice.
        """
        extracted = self._extract_profile_dict(raw_text, correction=False)
        profile = self._build_profile(extracted, raw_text, source_path)
        if profile is not None:
            return profile

        logger.warning(
            "Profile validation failed for %s; retrying extraction once with a "
            "correction prompt",
            source_path,
        )
        extracted = self._extract_profile_dict(raw_text, correction=True)
        profile = self._build_profile(extracted, raw_text, source_path)
        if profile is None:
            logger.warning(
                "Skipping resume after extraction failed validation twice: %s",
                source_path,
            )
        return profile

    # ------------------------------------------------------------------
    # Text extraction helpers (lazy, dependency-isolated, never raise)
    # ------------------------------------------------------------------

    def _extract_pdf_text(self, path: Path) -> str | None:
        """Extract raw text from a ``.pdf`` file using PyMuPDF (Requirement 2.1).

        PyMuPDF (``fitz``) is imported lazily so importing this module does not
        require the dependency. Any failure to import or read the document is
        caught and logged, returning ``None`` so the caller can skip and
        continue.

        Args:
            path: Path to the ``.pdf`` file.

        Returns:
            The concatenated page text, or ``None`` when extraction fails.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            logger.warning("PyMuPDF (fitz) not installed; cannot read %s: %s", path, exc)
            return None

        try:
            document = fitz.open(str(path))
            try:
                text = "".join(page.get_text() for page in document)
            finally:
                document.close()
            return text
        except Exception as exc:  # noqa: BLE001 - contain per-file extraction errors
            logger.warning("Failed to extract text from PDF %s: %s", path, exc)
            return None

    def _extract_docx_text(self, path: Path) -> str | None:
        """Extract raw text from a ``.docx`` file using python-docx (Req 2.2).

        python-docx is imported lazily so importing this module does not require
        the dependency. Any failure to import or read the document is caught and
        logged, returning ``None`` so the caller can skip and continue.

        Args:
            path: Path to the ``.docx`` file.

        Returns:
            The newline-joined paragraph text, or ``None`` when extraction fails.
        """
        try:
            from docx import Document
        except ImportError as exc:
            logger.warning(
                "python-docx not installed; cannot read %s: %s", path, exc
            )
            return None

        try:
            document = Document(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as exc:  # noqa: BLE001 - contain per-file extraction errors
            logger.warning("Failed to extract text from DOCX %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # JSON resume loading (no LLM)
    # ------------------------------------------------------------------

    def _parse_json_file(self, path: Path) -> CandidateProfile | None:
        """Load a structured ``.json`` resume directly, without the LLM (Req 2.4).

        A ``uuid4`` ``candidate_id`` is assigned when absent (Requirement 2.5),
        ``is_complete`` is set ``True`` to mark a successful build, and the
        profile is validated against the Pydantic v2 model. A structurally
        invalid JSON file (or one whose top level is not an object, or that fails
        validation) is skipped with a warning (Requirement 2.12).

        Args:
            path: Path to the ``.json`` resume file.

        Returns:
            A validated :class:`CandidateProfile`, or ``None`` when the file is
            skipped.
        """
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping structurally invalid JSON resume %s: %s", path, exc
            )
            return None

        if not isinstance(data, dict):
            logger.warning(
                "Skipping JSON resume with a non-object top level: %s", path
            )
            return None

        profile_data = dict(data)
        if not profile_data.get("candidate_id"):
            profile_data["candidate_id"] = str(uuid.uuid4())
        profile_data.setdefault("source_file", str(path))
        profile_data["is_complete"] = True

        try:
            return CandidateProfile(**profile_data)
        except ValidationError as exc:
            logger.warning(
                "Skipping JSON resume that failed validation %s: %s", path, exc
            )
            return None

    # ------------------------------------------------------------------
    # LLM extraction + profile assembly
    # ------------------------------------------------------------------

    def _extract_profile_dict(
        self, raw_text: str, *, correction: bool
    ) -> dict[str, Any]:
        """Call the LLM to extract a structured profile dict from resume text.

        Uses :meth:`OllamaClient.chat_json` with an empty-dict fallback so a
        response that cannot be parsed as JSON yields ``{}`` (which then falls
        back to model defaults) rather than raising. On the correction retry the
        system prompt carries an extra instruction to return valid JSON
        (Requirement 2.10).

        Args:
            raw_text: The resume text passed to the model as the user message.
            correction: When ``True``, append the correction instruction to the
                system prompt for the single validation retry.

        Returns:
            The extracted fields as a dict, or an empty dict when the response
            could not be parsed into a JSON object.
        """
        system_prompt = EXTRACTION_SYSTEM_PROMPT
        if correction:
            system_prompt += CORRECTION_INSTRUCTION

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ]
        result = self.ollama_client.chat_json(
            messages,
            fallback={},
            max_tokens=config.MAX_TOKENS_EXTRACTION,
            model=config.get_model("parser"),
        )
        if not isinstance(result, dict):
            logger.warning(
                "LLM extraction returned a non-object JSON value; using defaults"
            )
            return {}
        return result

    def _build_profile(
        self, extracted: dict[str, Any], raw_text: str, source_path: Path
    ) -> CandidateProfile | None:
        """Assemble and validate a :class:`CandidateProfile` from LLM output.

        Builds the profile from the extracted fields, assigning a ``uuid4``
        ``candidate_id`` (Requirement 2.5), recording ``source_file``, and always
        preserving the original ``raw_text`` (not the LLM's echoed value) so the
        audit phase can re-parse it. Missing optional fields fall back to their
        model defaults (Requirement 2.7). When a usable name or email is missing,
        a spaCy fallback attempts to recover them (Requirement 2.6). Validation
        failures are caught and surfaced as ``None`` so the caller can retry once
        and then skip (Requirements 2.9, 2.11).

        Args:
            extracted: The structured fields returned by the LLM.
            raw_text: The original resume text to store on the profile.
            source_path: Path recorded on the profile's ``source_file`` field.

        Returns:
            A validated :class:`CandidateProfile` with ``is_complete`` set, or
            ``None`` when the assembled profile fails validation.
        """
        profile_data = _strip_none(extracted)

        # Normalize nested list-of-dict fields so stray None values inside roles
        # or education entries fall back to their model defaults too.
        for list_field in ("roles", "education"):
            value = profile_data.get(list_field)
            if isinstance(value, list):
                profile_data[list_field] = [
                    _strip_none(item) if isinstance(item, dict) else item
                    for item in value
                ]

        # Identity and provenance: never trust the LLM's echoed id/raw_text.
        profile_data["candidate_id"] = str(uuid.uuid4())
        profile_data["source_file"] = str(source_path)
        profile_data["raw_text"] = raw_text

        self._apply_name_email_fallback(profile_data, raw_text)

        try:
            profile = CandidateProfile(**profile_data)
        except ValidationError as exc:
            logger.warning("CandidateProfile failed validation: %s", exc)
            return None

        profile.is_complete = True
        return profile

    def _apply_name_email_fallback(
        self, profile_data: dict[str, Any], raw_text: str
    ) -> None:
        """Fill a missing name/email from spaCy, mutating ``profile_data`` (Req 2.6).

        When the LLM dict lacks a usable name or email, a spaCy ``en_core_web_sm``
        pass over ``raw_text`` is used to recover a PERSON entity (name) and an
        email via regex. Any value that remains unresolved is left to its model
        default: the ``name`` key is removed (so the model default
        ``"Unknown Candidate"`` applies) and ``email`` stays absent/``None``.

        Args:
            profile_data: The in-progress profile dict to update in place.
            raw_text: The resume text to run the spaCy/regex fallback against.

        Returns:
            None.
        """
        name = profile_data.get("name")
        email = profile_data.get("email")
        needs_name = not (isinstance(name, str) and name.strip())
        needs_email = not (isinstance(email, str) and email.strip())

        if not needs_name and not needs_email:
            return

        fallback_name, fallback_email = self._spacy_fallback(raw_text)

        if needs_name:
            if fallback_name:
                profile_data["name"] = fallback_name
            else:
                # Leave the model default ("Unknown Candidate") to apply.
                profile_data.pop("name", None)

        if needs_email and fallback_email:
            profile_data["email"] = fallback_email

    def _spacy_fallback(self, raw_text: str) -> tuple[str | None, str | None]:
        """Recover a candidate name and email from raw text (Requirement 2.6).

        The email is matched with a simple regex; the name is taken from the
        first spaCy PERSON entity. The spaCy pipeline is loaded lazily and may be
        unavailable (model not installed), in which case only the regex email is
        attempted and the name is left unresolved — the method never raises.

        Args:
            raw_text: The resume text to scan.

        Returns:
            A ``(name, email)`` tuple, where either element is ``None`` when it
            could not be recovered.
        """
        email: str | None = None
        match = _EMAIL_RE.search(raw_text)
        if match is not None:
            email = match.group(0)

        name: str | None = None
        nlp = _get_nlp()
        if nlp is not None:
            try:
                document = nlp(raw_text)
                for entity in document.ents:
                    if entity.label_ == "PERSON" and entity.text.strip():
                        name = entity.text.strip()
                        break
            except Exception as exc:  # noqa: BLE001 - fallback must never crash
                logger.warning("spaCy NER failed during fallback: %s", exc)

        return name, email


#: Exact user prompt that instructs the LLM to classify a ``.txt`` job
#: description into a JSON array of requirement objects. The actual JD text is
#: appended to this prompt before the call. The enumerated buckets and
#: dimensions mirror the :class:`~models.job.JobRequirement` ``Literal`` fields
#: so the model targets the same vocabulary the Pydantic model validates against
#: (Requirements 3.1, 3.5, 3.6).
JD_CLASSIFICATION_PROMPT: str = (
    "Parse this job description. Return ONLY a JSON array of requirement "
    "objects. Each object: {text, bucket, dimension}. No preamble. No markdown. "
    "Buckets: must_have | nice_to_have | culture_signal | seniority_marker "
    "Dimensions: technical | soft_skill | domain | experience_level"
)


class JdParser:
    """Convert a job-description file into a validated :class:`JobDescription`.

    The parser dispatches on file extension (case-insensitive). A ``.json`` job
    description is loaded directly without any LLM call; a ``.txt`` job
    description has its text classified into requirement buckets/dimensions by
    the LLM (via :class:`~utils.ollama_client.OllamaClient`).

    Unlike :class:`ResumeParser`, which contains every per-file error and returns
    ``None`` so a directory scan can continue, ``JdParser`` is fail-fast: there
    is exactly one job description per run, so an unsupported extension or a
    corrupt/invalid ``.json`` file raises rather than producing a placeholder
    (Requirements 3.3, 3.4). Only a ``.txt`` classification that yields malformed
    JSON degrades gracefully, by falling back to a still-valid
    :class:`JobDescription` with an empty requirements list (Requirement 3.9).
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """Initialize the parser.

        Args:
            ollama_client: The shared LLM wrapper used to classify ``.txt`` job
                descriptions into requirement buckets and dimensions. It is not
                used for ``.json`` job descriptions.

        Returns:
            None.
        """
        self.ollama_client = ollama_client

    def parse_file(self, path: Path) -> JobDescription:
        """Parse the job-description file, dispatching on its extension.

        Dispatch (case-insensitive):
            * ``.json`` -> load the structured model directly, no LLM
              (Requirement 3.2). A corrupt/invalid file fails immediately and
              does NOT fall back to LLM processing (Requirement 3.4).
            * ``.txt``  -> classify the text into requirements via the LLM
              (Requirement 3.1).
            * other      -> raise, producing no :class:`JobDescription`
              (Requirement 3.3).

        Args:
            path: Filesystem path to the job-description file.

        Returns:
            A validated :class:`JobDescription` (Requirement 3.8).

        Raises:
            ValueError: When the extension is unsupported (Requirement 3.3) or a
                ``.json`` file is corrupt/structurally invalid or fails
                validation (Requirement 3.4).
            OSError: When the file cannot be read from disk.
        """
        extension = path.suffix.lower()

        if extension == ".json":
            return self._parse_json_file(path)

        if extension == ".txt":
            return self._parse_txt_file(path)

        message = (
            f"Unsupported job description extension {extension!r}: {path}. "
            "Only .json and .txt are supported."
        )
        logger.error(message)
        raise ValueError(message)

    # ------------------------------------------------------------------
    # JSON job description loading (no LLM, fail-fast)
    # ------------------------------------------------------------------

    def _parse_json_file(self, path: Path) -> JobDescription:
        """Load a structured ``.json`` job description directly (Requirement 3.2).

        A ``job_id`` is assigned via ``uuid5`` on the file path when absent
        (Requirement 3.7), and the result is validated against the Pydantic v2
        model (Requirement 3.8). A structurally invalid file (a JSON decode
        error, a non-object top level, or a payload that fails validation) is
        reported and raised immediately; the LLM is never invoked as a fallback
        (Requirement 3.4).

        Args:
            path: Path to the ``.json`` job-description file.

        Returns:
            A validated :class:`JobDescription`.

        Raises:
            ValueError: When the file is structurally invalid, has a non-object
                top level, or fails Pydantic validation (Requirement 3.4).
            OSError: When the file cannot be read from disk.
        """
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            message = f"Corrupt or invalid JSON job description {path}: {exc}"
            logger.error(message)
            raise ValueError(message) from exc

        if not isinstance(data, dict):
            message = (
                f"Invalid JSON job description with a non-object top level: {path}"
            )
            logger.error(message)
            raise ValueError(message)

        job_data = dict(data)
        if not job_data.get("job_id"):
            job_data["job_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))

        try:
            return JobDescription(**job_data)
        except ValidationError as exc:
            message = f"JSON job description failed validation {path}: {exc}"
            logger.error(message)
            raise ValueError(message) from exc

    # ------------------------------------------------------------------
    # TXT job description classification (LLM)
    # ------------------------------------------------------------------

    def _parse_txt_file(self, path: Path) -> JobDescription:
        """Classify a ``.txt`` job description into requirements via the LLM.

        Reads the job-description text and calls
        :meth:`OllamaClient.chat_json` with the exact classification prompt and
        an empty-list fallback (Requirement 3.1). Each returned item is turned
        into a :class:`JobRequirement`; an item with an invalid ``bucket`` or
        ``dimension`` (or missing fields) is skipped per-item so one malformed
        entry does not crash the whole parse (Requirements 3.5, 3.6). When the
        classification is malformed (the ``[]`` fallback is returned after the
        client's retries) or yields zero valid requirements, a warning is logged
        and a :class:`JobDescription` with an empty requirements list is produced
        — which still validates (Requirements 3.8, 3.9). A ``job_id`` is assigned
        via ``uuid5`` on the file path (Requirement 3.7).

        Args:
            path: Path to the ``.txt`` job-description file.

        Returns:
            A validated :class:`JobDescription` whose ``raw_text`` is the
            original job-description text.

        Raises:
            OSError: When the file cannot be read from disk.
        """
        raw_text = path.read_text(encoding="utf-8")

        messages = [
            {
                "role": "user",
                "content": f"{JD_CLASSIFICATION_PROMPT}\n\n{raw_text}",
            }
        ]
        classified = self.ollama_client.chat_json(
            messages,
            fallback=[],
            max_tokens=config.MAX_TOKENS_EXTRACTION,
            model=config.get_model("jd_parser"),
        )

        requirements = self._build_requirements(classified)
        if not requirements:
            logger.warning(
                "JD classification produced no valid requirements for %s; "
                "falling back to an empty requirements list",
                path,
            )

        job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))
        try:
            return JobDescription(
                job_id=job_id,
                requirements=requirements,
                raw_text=raw_text,
            )
        except ValidationError as exc:
            # The fallback must always validate; an empty requirements list and a
            # string job_id satisfy the model, so reaching here is unexpected.
            logger.warning(
                "JobDescription failed validation for %s; returning a minimal "
                "fallback model: %s",
                path,
                exc,
            )
            return JobDescription(job_id=job_id, raw_text=raw_text)

    def _build_requirements(self, classified: Any) -> list[JobRequirement]:
        """Build :class:`JobRequirement` objects from classified LLM output.

        Tolerates a malformed response shape: a non-list ``classified`` value
        yields no requirements, and any individual entry that is not a mapping or
        whose ``bucket``/``dimension`` is outside the allowed vocabulary is
        skipped (its :class:`~pydantic.ValidationError` is caught per-item) so a
        single bad item does not discard the rest (Requirements 3.5, 3.6, 3.9).

        Args:
            classified: The parsed classification result returned by
                :meth:`OllamaClient.chat_json`; expected to be a list of
                ``{text, bucket, dimension}`` mappings.

        Returns:
            The valid :class:`JobRequirement` objects, in their original order.
            An empty list when ``classified`` is not a list or contains no valid
            entries.
        """
        if not isinstance(classified, list):
            logger.warning(
                "JD classification returned a non-list JSON value; ignoring"
            )
            return []

        requirements: list[JobRequirement] = []
        for item in classified:
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-object JD requirement entry: %r", item
                )
                continue
            try:
                requirements.append(JobRequirement(**item))
            except ValidationError as exc:
                logger.warning(
                    "Skipping JD requirement that failed validation %r: %s",
                    item,
                    exc,
                )
        return requirements


# ---------------------------------------------------------------------------
# Competition mode: streaming JSONL loader
# ---------------------------------------------------------------------------


def load_candidates_jsonl(
    path: str | Path,
    *,
    max_candidates: int | None = None,
) -> Iterator[CandidateProfile]:
    """Stream candidates from a competition-format JSONL file.

    Each line is a JSON object matching the competition ``candidate_schema.json``
    structure (``profile``, ``career_history``, ``education``, ``skills``,
    ``certifications``, ``languages``, ``redrob_signals``).  This function
    performs *no LLM inference* — the mapping to :class:`CandidateProfile` is
    purely structural.

    Args:
        path: Path to the ``.jsonl`` file.
        max_candidates: Optional cap; stop after yielding this many profiles.
            ``None`` means read until EOF.

    Yields:
        Validated :class:`CandidateProfile` instances.  Lines that fail
        validation are logged and skipped (fail-open, consistent with the
        existing INGEST philosophy).
    """
    from models.candidate import RedrobSignals

    path = Path(path)
    logger.info("Loading candidates from JSONL: %s", path)

    count = 0
    skipped = 0

    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if max_candidates is not None and count >= max_candidates:
                logger.info(
                    "Reached max_candidates=%d — stopping JSONL load",
                    max_candidates,
                )
                break

            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)
                skipped += 1
                continue

            try:
                profile = _map_competition_row(row, line_no)
                count += 1
                yield profile
            except Exception as exc:
                logger.warning(
                    "Skipping invalid candidate on line %d: %s", line_no, exc
                )
                skipped += 1

    logger.info(
        "JSONL load complete: %d candidates loaded, %d skipped", count, skipped
    )


def _map_competition_row(row: dict[str, Any], line_no: int) -> CandidateProfile:
    """Map a single competition JSONL row to a :class:`CandidateProfile`.

    The competition schema stores fields like ``career_history`` (list of
    dicts) and ``redrob_signals`` (dict) which differ from the internal
    ``CandidateProfile`` shape.  This mapper bridges the gap.

    Raises:
        KeyError: If ``candidate_id`` or ``profile`` is missing (caller logs
            and skips).
    """
    from models.candidate import RedrobSignals

    # --- candidate_id (required) ---
    candidate_id = str(row.get("candidate_id") or row.get("id") or "")
    if not candidate_id:
        raise KeyError("Missing candidate_id")

    # --- profile sub-object ---
    profile_data = row.get("profile", row)  # fallback to top-level

    name = profile_data.get("name", "")
    email = profile_data.get("email")
    years_experience = float(profile_data.get("years_of_experience", 0))
    headline = profile_data.get("headline", "")
    summary = profile_data.get("summary", "")
    location = profile_data.get("location", "")

    # --- career_history → roles ---
    roles: list[CandidateRole] = []
    for entry in row.get("career_history", row.get("roles", [])):
        if not isinstance(entry, dict):
            continue
        start = entry.get("start_date", "")
        end = entry.get("end_date")
        duration = int(entry.get("duration_months", 0))
        if not duration and start:
            # compute duration if end is present and start is parseable
            if end:
                try:
                    sy, sm = map(int, start[:7].split("-"))
                    ey, em = map(int, end[:7].split("-"))
                    duration = max(0, (ey - sy) * 12 + (em - sm))
                except Exception:
                    pass

        roles.append(
            CandidateRole(
                title=entry.get("title", entry.get("job_title", "")),
                company=entry.get("company", entry.get("company_name", "")),
                start_date=start,
                end_date=end,
                duration_months=duration,
                company_size_estimate=entry.get("company_size_estimate"),
                scope_keywords=entry.get("scope_keywords", []),
            )
        )

    # --- skills ---
    raw_skills = row.get("skills", [])
    skills_claimed: list[str] = []
    for s in raw_skills:
        if isinstance(s, str):
            skills_claimed.append(s)
        elif isinstance(s, dict):
            skills_claimed.append(s.get("name", ""))

    # --- education ---
    education: list[dict] = []
    for edu in row.get("education", []):
        if isinstance(edu, dict):
            education.append(edu)

    # --- redrob_signals ---
    signals_data = row.get("redrob_signals")
    redrob_signals: RedrobSignals | None = None
    if isinstance(signals_data, dict):
        try:
            redrob_signals = RedrobSignals(**signals_data)
        except Exception:
            # Partial signals — populate what we can
            try:
                redrob_signals = RedrobSignals.model_validate(signals_data)
            except Exception:
                logger.debug(
                    "Line %d: could not parse redrob_signals, skipping field",
                    line_no,
                )

    # --- build raw_text fallback for downstream compat ---
    raw_text = summary or headline or f"{name} {years_experience}yoe"

    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        email=email,
        years_experience=years_experience,
        roles=roles,
        skills_claimed=skills_claimed,
        education=education,
        trajectory_vector=None,
        raw_text=raw_text,
        source_file=f"jsonl:line:{line_no}",
        is_complete=True,
        redrob_signals=redrob_signals,
    )
