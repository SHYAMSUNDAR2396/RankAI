"""Multi-persona candidate scoring for the Candidate Ranking System.

This module owns the SCORE phase. It evaluates a candidate through a panel of
three evaluator personas (``hiring_manager``, ``peer_interviewer``,
``devils_advocate``) using retrieved RAG context and aggregates their scores
into a single clamped composite score plus a panel-variance signal that routes
high-disagreement candidates to human review.

The two aggregation surfaces exposed here -- :meth:`CandidateScoringPipeline.composite_score`
and :meth:`CandidateScoringPipeline.panel_variance` -- are *pure* static
functions of the persona scores. Keeping them free of LLM and vector-store I/O
makes the scoring math deterministically property-testable offline
(Requirements 12.9, 12.10). The full ``score`` orchestration that calls the
Ollama panel and the vector store is layered on top of these helpers.

Scoring constants (the persona weights) are read exclusively from
``config.PERSONA_WEIGHTS`` so this module never redefines them (Requirement 10.5).
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    # Imported only for type checking so that importing ``pipeline.score`` at
    # runtime does not require ``ollama``/``pydantic``/``chromadb`` (pulled in by
    # these modules) to be installed. The ``__init__`` receives already-built
    # instances and ``CandidateProfile`` is used purely as an annotation, which
    # ``from __future__ import annotations`` keeps unevaluated at import time.
    from models.candidate import CandidateProfile
    from pipeline.embed import VectorStoreManager
    from utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

#: Inclusive bounds the composite score is clamped to (Requirement 6.4).
_COMPOSITE_MIN = 0.0
_COMPOSITE_MAX = 10.0

#: Number of decimal places the composite score is rounded to (Requirement 6.3).
_COMPOSITE_ROUNDING = 2

#: The three evaluator personas, in the fixed order they are run sequentially
#: (Requirement 6.1). Each name also names the ``personas/{name}.txt`` prompt
#: file loaded in :meth:`CandidateScoringPipeline.__init__`.
_PERSONAS: tuple[str, str, str] = (
    "hiring_manager",
    "peer_interviewer",
    "devils_advocate",
)

#: Directory (relative to the repo root) holding the persona system prompts.
_PERSONA_DIR = Path(__file__).resolve().parent.parent / "personas"

#: Inclusive bounds each persona score is clamped to (Requirement 6.1).
_PERSONA_SCORE_MIN = 0.0
_PERSONA_SCORE_MAX = 10.0

#: Inclusive bounds a persona confidence value is clamped to.
_CONFIDENCE_MIN = 0.0
_CONFIDENCE_MAX = 1.0

#: The allowed persona verdicts (Requirement 6.11). Anything outside this set is
#: coerced to ``"maybe"``.
_VALID_VERDICTS = frozenset({"strong_yes", "yes", "maybe", "no"})

#: The defined default persona result substituted when a persona response cannot
#: be parsed as JSON (Requirement 6.9). Passed to ``chat_json`` as its fallback.
def _default_persona_result() -> dict:
    """Return a fresh copy of the default persona result.

    A factory (rather than a shared module-level dict) is used so each caller
    receives an independent copy and cannot mutate a shared default's list
    fields.

    Args:
        None.

    Returns:
        dict: The default persona result ``{"score": 5.0, "confidence": 0.0,
        "strengths": [], "concerns": ["Parse error"], "verdict": "maybe"}``.
    """
    return {
        "score": 5.0,
        "confidence": 0.0,
        "strengths": [],
        "concerns": ["Parse error"],
        "verdict": "maybe",
    }


def _clamp_number(value: object, low: float, high: float, default: float) -> float:
    """Coerce ``value`` to a float clamped to ``[low, high]``.

    Used to defensively normalize a persona's ``score`` and ``confidence``: a
    non-numeric, missing, or out-of-range value falls back to ``default`` (then
    itself clamped), so a malformed persona response cannot produce an
    out-of-range score (Requirement 6.9).

    Args:
        value: The raw value from the persona response (any type).
        low: The inclusive lower bound.
        high: The inclusive upper bound.
        default: The value used when ``value`` is not a finite number.

    Returns:
        float: ``value`` as a float clamped to ``[low, high]``, or ``default``
        clamped to ``[low, high]`` when ``value`` is not a usable number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        number = default
    else:
        number = float(value)
    return max(low, min(high, number))


class CandidateScoringPipeline:
    """Scores a candidate via the three-persona panel with RAG context.

    On construction the pipeline loads and caches the three persona system
    prompts from ``personas/*.txt`` (Requirement 6.1) and holds references to
    the :class:`OllamaClient` (the single LLM seam) and the
    :class:`VectorStoreManager` used for RAG retrieval. The :meth:`score`
    orchestration builds a profile query, retrieves JD and calibration context,
    runs the three persona Ollama calls sequentially, aggregates their scores
    via the pure static helpers below, generates a narrative, and returns the
    full result dict (Requirement 6.11).

    The two static methods (:meth:`composite_score`, :meth:`panel_variance`) are
    pure aggregation helpers that depend only on their inputs and
    ``config.PERSONA_WEIGHTS``; they perform no I/O and are safe to call without
    a running Ollama server or vector store.

    Attributes:
        ollama_client: The wrapper through which every LLM call is routed.
        store: The vector store queried for JD and calibration RAG context. May
            be ``None``, in which case retrieval yields empty context.
        persona_prompts: Mapping of persona name to its cached system-prompt
            text, loaded once in :meth:`__init__`.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        store: VectorStoreManager | None,
    ) -> None:
        """Store collaborators and load the three persona system prompts.

        The persona prompts are read once from ``personas/{persona}.txt`` (with
        the path resolved relative to the repo root) and cached on the instance
        so each :meth:`score` call reuses them without re-reading the files
        (Requirement 6.1).

        Args:
            ollama_client: The :class:`OllamaClient` through which all persona
                and narrative LLM calls are routed.
            store: The :class:`VectorStoreManager` queried for JD and
                calibration RAG context. ``None`` is tolerated; retrieval then
                yields empty context.

        Returns:
            None.
        """
        self.ollama_client = ollama_client
        self.store = store
        self.persona_prompts: dict[str, str] = self._load_persona_prompts()

    @staticmethod
    def _load_persona_prompts() -> dict[str, str]:
        """Load and cache the three persona system prompts from disk.

        Reads ``personas/{persona}.txt`` for each persona in ``_PERSONAS``,
        resolving the directory relative to the repo root so the load is robust
        to the process's working directory (Requirement 6.1).

        Args:
            None.

        Returns:
            dict[str, str]: Mapping of persona name to its system-prompt text.
        """
        prompts: dict[str, str] = {}
        for persona in _PERSONAS:
            prompt_path = _PERSONA_DIR / f"{persona}.txt"
            prompts[persona] = prompt_path.read_text(encoding="utf-8").strip()
            logger.debug("Loaded persona prompt: %s", prompt_path)
        return prompts

    def score(self, profile: CandidateProfile) -> dict:
        """Score one candidate through the three-persona panel with RAG context.

        Orchestrates the full SCORE phase for a single candidate:

        1. Builds a free-text query from the profile (name, experience, skills,
           roles, trajectory).
        2. Retrieves up to 5 JD context strings and 3 calibration dicts from the
           vector store, defending against a ``None`` store or empty results
           (Requirement 6.2).
        3. Runs the three persona Ollama calls *sequentially* via ``chat_json``,
           coercing each response defensively to a valid score/verdict and
           substituting the default persona result on an unparseable response
           (Requirements 6.1, 6.8, 6.9).
        4. Aggregates the persona scores into the clamped composite and the
           panel variance, setting ``requires_human_review`` when the variance
           exceeds ``config.HUMAN_REVIEW_VARIANCE_THRESHOLD`` (Requirements
           6.3-6.7).
        5. Generates a three-sentence narrative via one more LLM call, falling
           back to a minimal narrative if that optional call fails
           (Requirement 6.10).
        6. Returns the full schema-complete result dict with de-duplicated
           ``strengths`` and ``concerns`` (Requirement 6.11).

        Args:
            profile: The enriched candidate profile to score.

        Returns:
            dict: The full score result dict described by the design's Score
            Result Dict schema (Requirement 6.11). ``bias_flag`` and
            ``counterfactual_delta`` are populated with their defaults
            (``False`` / ``0.0``); the audit phase fills in the real values.
        """
        logger.info("Scoring candidate %s (%s)", profile.candidate_id, profile.name)

        profile_text = self._build_profile_text(profile)
        jd_context, calibration_context = self._retrieve_context(profile_text)
        shared_context = self._build_shared_context(jd_context, calibration_context)
        candidate_block = self._build_candidate_block(profile)

        persona_results: dict[str, dict] = {}
        for persona in _PERSONAS:
            persona_results[persona] = self._score_persona(
                persona, candidate_block, shared_context
            )

        persona_scores = {
            persona: persona_results[persona]["score"] for persona in _PERSONAS
        }
        composite = self.composite_score(persona_scores)
        variance = self.panel_variance(persona_scores)
        requires_human_review = variance > config.HUMAN_REVIEW_VARIANCE_THRESHOLD

        persona_verdicts = {
            persona: persona_results[persona]["verdict"] for persona in _PERSONAS
        }
        strengths = self._dedupe(
            item
            for persona in _PERSONAS
            for item in persona_results[persona]["strengths"]
        )
        concerns = self._dedupe(
            item
            for persona in _PERSONAS
            for item in persona_results[persona]["concerns"]
        )

        narrative = self._generate_narrative(
            profile_text, composite, persona_verdicts, concerns
        )

        trajectory_vector = profile.trajectory_vector or {}
        result = {
            "candidate_id": profile.candidate_id,
            "name": profile.name,
            "trajectory_score": trajectory_vector.get("seniority_score", 5.0),
            "hiring_manager_score": persona_scores["hiring_manager"],
            "peer_interviewer_score": persona_scores["peer_interviewer"],
            "devils_advocate_score": persona_scores["devils_advocate"],
            "composite_score": composite,
            "panel_variance": variance,
            "requires_human_review": requires_human_review,
            "persona_verdicts": persona_verdicts,
            "strengths": strengths,
            "concerns": concerns,
            "narrative": narrative,
            "bias_flag": False,  # populated by the audit phase later.
            "counterfactual_delta": 0.0,  # populated by the audit phase later.
        }
        logger.info(
            "Scored candidate %s: composite=%.2f variance=%.2f review=%s",
            profile.candidate_id,
            composite,
            variance,
            requires_human_review,
        )
        return result

    @staticmethod
    def _build_profile_text(profile: CandidateProfile) -> str:
        """Build the free-text profile query used for RAG retrieval.

        Args:
            profile: The candidate profile to render.

        Returns:
            str: A one-line summary combining the candidate's name, experience,
            skills, roles, and trajectory vector, used both as the retrieval
            query and inside the narrative prompt.
        """
        skills = ", ".join(profile.skills_claimed)
        roles = "; ".join(f"{role.title} at {role.company}" for role in profile.roles)
        return (
            f"{profile.name}, {profile.years_experience} yrs exp. "
            f"Skills: {skills}. Roles: {roles}. "
            f"Trajectory: {profile.trajectory_vector}"
        )

    def _retrieve_context(self, profile_text: str) -> tuple[list[str], list[dict]]:
        """Retrieve JD and calibration RAG context for a profile query.

        Defends against a ``None`` store by returning empty context, so scoring
        can proceed without a vector store (Requirement 6.2).

        Args:
            profile_text: The profile query to match against the store.

        Returns:
            tuple[list[str], list[dict]]: ``(jd_context, calibration_context)``
            -- up to 5 JD context strings and up to 3 calibration dicts. Either
            list is empty when the store is ``None`` or returns no matches.
        """
        if self.store is None:
            logger.debug("No vector store configured; using empty RAG context")
            return [], []

        jd_context = self.store.query_jd_context(profile_text, n=5) or []
        calibration_context = self.store.query_calibration(profile_text, n=3) or []
        logger.debug(
            "Retrieved %d JD context items and %d calibration items",
            len(jd_context),
            len(calibration_context),
        )
        return jd_context, calibration_context

    @staticmethod
    def _build_shared_context(
        jd_context: list[str], calibration_context: list[dict]
    ) -> str:
        """Assemble the shared RAG context block passed to every persona.

        Args:
            jd_context: The retrieved JD requirement strings.
            calibration_context: The retrieved calibration dicts, each carrying
                ``outcome`` and ``reason``.

        Returns:
            str: A formatted block listing the relevant job requirements and the
            calibration reference examples.
        """
        calibration_lines = "\n".join(
            f"- {entry.get('outcome', '')}: {entry.get('reason', '')}"
            for entry in calibration_context
        )
        return (
            "Relevant job requirements:\n"
            + "\n".join(jd_context)
            + "\n\nCalibration reference examples:\n"
            + calibration_lines
        )

    @staticmethod
    def _build_candidate_block(profile: CandidateProfile) -> str:
        """Render the candidate's fields into the persona prompt block.

        Args:
            profile: The candidate profile to render.

        Returns:
            str: A multi-line block describing the candidate's name, experience,
            skills, roles, education, and trajectory for the persona prompt.
        """
        skills = ", ".join(profile.skills_claimed)
        roles = "\n".join(
            f"  - {role.title} at {role.company}" for role in profile.roles
        )
        education = "; ".join(str(entry) for entry in profile.education)
        return (
            "Candidate profile:\n"
            f"Name: {profile.name}\n"
            f"Years of experience: {profile.years_experience}\n"
            f"Skills: {skills}\n"
            f"Roles:\n{roles}\n"
            f"Education: {education}\n"
            f"Trajectory: {profile.trajectory_vector}"
        )

    def _score_persona(
        self, persona: str, candidate_block: str, shared_context: str
    ) -> dict:
        """Run one persona's scoring call and coerce the response defensively.

        Calls ``chat_json`` (which already retries the LLM call and returns the
        supplied fallback on a JSON-parse failure, satisfying Requirements 6.8
        and 6.9) and then coerces the parsed dict to a valid persona result:
        missing keys are filled from the default, the score is clamped to
        ``[0, 10]``, the confidence to ``[0, 1]``, and an out-of-set verdict is
        coerced to ``"maybe"``. The net effect is that an unparseable or
        malformed persona never raises and contributes the default verdict and
        score while scoring continues (Requirement 6.9).

        Args:
            persona: The persona name; also selects the cached system prompt.
            candidate_block: The rendered candidate block for the user prompt.
            shared_context: The shared RAG context appended to the user prompt.

        Returns:
            dict: A coerced persona result with keys ``score`` (float in
            ``[0, 10]``), ``confidence`` (float in ``[0, 1]``), ``strengths``
            (list), ``concerns`` (list), and ``verdict`` (one of the allowed
            verdicts).
        """
        user_prompt = candidate_block + "\n\n" + shared_context
        messages = [
            {"role": "system", "content": self.persona_prompts[persona]},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.ollama_client.chat_json(
            messages=messages,
            fallback=_default_persona_result(),
            max_tokens=config.MAX_TOKENS_SCORING,
            model=config.get_model(persona),
        )
        result = self._coerce_persona_result(raw)
        logger.debug(
            "Persona %s -> score=%.2f verdict=%s",
            persona,
            result["score"],
            result["verdict"],
        )
        return result

    @staticmethod
    def _coerce_persona_result(raw: object) -> dict:
        """Coerce a raw persona response into a valid persona result dict.

        Guards against a non-dict payload, missing keys, out-of-range scores,
        and invalid verdicts so a malformed-but-parseable response cannot break
        scoring (Requirement 6.9).

        Args:
            raw: The value returned by ``chat_json`` -- expected to be a dict but
                tolerated otherwise.

        Returns:
            dict: A persona result with a clamped ``score`` and ``confidence``,
            list ``strengths`` and ``concerns``, and a valid ``verdict``,
            defaulting any missing or invalid field.
        """
        default = _default_persona_result()
        if not isinstance(raw, dict):
            return default

        score = _clamp_number(
            raw.get("score"), _PERSONA_SCORE_MIN, _PERSONA_SCORE_MAX, default["score"]
        )
        confidence = _clamp_number(
            raw.get("confidence"),
            _CONFIDENCE_MIN,
            _CONFIDENCE_MAX,
            default["confidence"],
        )
        strengths = raw.get("strengths")
        concerns = raw.get("concerns")
        verdict = raw.get("verdict")
        return {
            "score": score,
            "confidence": confidence,
            "strengths": list(strengths) if isinstance(strengths, list) else [],
            "concerns": list(concerns) if isinstance(concerns, list) else [],
            "verdict": verdict if verdict in _VALID_VERDICTS else "maybe",
        }

    def _generate_narrative(
        self,
        profile_text: str,
        composite: float,
        persona_verdicts: dict[str, str],
        concerns: list[str],
    ) -> str:
        """Generate the three-sentence candidate narrative via an LLM call.

        Wraps the optional narrative call in a try/except: per Requirement 6.10
        a narrative is generated via Ollama, but a missing summary should not
        crash an otherwise-complete candidate, so an :class:`OllamaCallError` is
        caught and logged and a minimal fallback narrative is used instead.

        Args:
            profile_text: The candidate summary used in the prompt.
            composite: The candidate's composite score.
            persona_verdicts: Mapping of persona name to verdict for the prompt.
            concerns: The de-duplicated concerns; the first is used as the
                fallback narrative when the LLM call fails.

        Returns:
            str: The generated narrative, or a minimal fallback string when the
            narrative call fails.
        """
        # Imported here (not at module load) so importing ``pipeline.score`` does
        # not require ``ollama`` to be installed for the pure static helpers.
        from utils.ollama_client import OllamaCallError

        system_prompt = (
            "You write concise 3-sentence candidate summaries for hiring managers."
        )
        user_prompt = (
            "Summarise this candidate's ranking in exactly 3 sentences. "
            "Sentence 1: their primary strength. "
            "Sentence 2: the main concern or gap. "
            "Sentence 3: what additional evidence would increase confidence. "
            f"Candidate: {profile_text}. Composite score: {composite}/10. "
            f"Panel verdicts: hiring_manager={persona_verdicts['hiring_manager']}, "
            f"peer={persona_verdicts['peer_interviewer']}, "
            f"devils_advocate={persona_verdicts['devils_advocate']}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.ollama_client.chat(
                messages,
                max_tokens=config.MAX_TOKENS_NARRATIVE,
                model=config.get_model("narrative"),
            )
        except OllamaCallError as exc:
            fallback = concerns[0] if concerns else "Narrative unavailable."
            logger.warning(
                "Narrative generation failed (%s); using fallback narrative", exc
            )
            return fallback

    @staticmethod
    def _dedupe(items: object) -> list[str]:
        """Return the items as a list with duplicates removed, order-preserving.

        Args:
            items: An iterable of string items (e.g. collected persona
                strengths or concerns).

        Returns:
            list[str]: The items in first-seen order with later duplicates
            removed.
        """
        seen: set[str] = set()
        result: list[str] = []
        for item in items:  # type: ignore[union-attr]
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @staticmethod
    def composite_score(persona_scores: dict[str, float]) -> float:
        """Aggregate persona scores into the clamped composite score.

        The composite is a plain weighted sum of the persona scores using
        ``config.PERSONA_WEIGHTS`` (``hiring_manager`` 0.45, ``peer_interviewer``
        0.35, ``devils_advocate`` -0.20). The devil's-advocate weight is already
        negative in config, so a straight weighted sum subtracts that persona's
        contribution. The sum is rounded to two decimal places (Requirement 6.3)
        and then clamped to the inclusive range ``[0.0, 10.0]`` (Requirement 6.4).

        Args:
            persona_scores: Mapping of persona name to that persona's score in
                ``[0, 10]``. Must contain every key in ``config.PERSONA_WEIGHTS``
                (``"hiring_manager"``, ``"peer_interviewer"``,
                ``"devils_advocate"``).

        Returns:
            float: The weighted sum rounded to two decimals and clamped to
            ``[0.0, 10.0]``.
        """
        weighted_sum = sum(
            weight * persona_scores[persona]
            for persona, weight in config.PERSONA_WEIGHTS.items()
        )
        rounded = round(weighted_sum, _COMPOSITE_ROUNDING)
        clamped = max(_COMPOSITE_MIN, min(_COMPOSITE_MAX, rounded))
        return float(clamped)

    @staticmethod
    def panel_variance(persona_scores: dict[str, float]) -> float:
        """Compute the population variance of the three persona scores.

        The panel variance is the population variance of the persona scores --
        the mean of the squared deviations of the three scores from their mean
        (Requirement 6.5). A larger value indicates more disagreement across the
        panel and is later used to decide whether a candidate requires human
        review.

        Args:
            persona_scores: Mapping of persona name to that persona's score in
                ``[0, 10]``. Must contain every key in ``config.PERSONA_WEIGHTS``
                (``"hiring_manager"``, ``"peer_interviewer"``,
                ``"devils_advocate"``).

        Returns:
            float: The population variance of the three persona scores; ``0.0``
            when all three scores are equal.
        """
        scores = [persona_scores[persona] for persona in config.PERSONA_WEIGHTS]
        return float(statistics.pvariance(scores))
