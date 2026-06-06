"""Ranked-output construction for the Candidate Ranking System.

This module owns the OUTPUT phase's pure ranking logic. It exposes two
side-effect-free helpers used to build ``ranked_candidates.csv``:

* :func:`rank_candidates` -- orders scored result dicts by composite score
  (descending, with a deterministic ``candidate_id`` tie-break) and assigns
  each a consecutive integer rank starting at 1 (Requirements 8.1, 8.2).
* :func:`verdict_consensus` -- collapses the three persona verdicts into a
  single consensus verdict: the verdict held by at least two of the three
  personas, falling back to the ``hiring_manager`` persona's verdict when all
  three disagree (Requirements 8.5, 8.6).

Both functions are pure: they read their inputs, return new values, and never
mutate the arguments they are given.

On top of those pure helpers, :func:`write_ranked_csv` performs the OUTPUT
phase's single side effect: it ranks the scored results, serializes them to the
fixed CSV schema (Requirement 8.4), and writes ``ranked_candidates.csv``
*atomically* -- the data is written to a temporary file in the destination
directory and then renamed into place, so a failure mid-write never leaves a
partial ``ranked_candidates.csv`` behind (Requirement 8.7). ``pandas`` is
imported lazily inside that function so importing this module never requires
``pandas`` to be installed.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

#: Output CSV filename written into the operator's output directory.
_RANKED_CSV_FILENAME = "ranked_candidates.csv"

#: Fixed CSV column order for the ranked output (Requirement 8.4).
_CSV_COLUMNS = [
    "rank",
    "candidate_id",
    "name",
    "composite_score",
    "trajectory_score",
    "hiring_manager_score",
    "peer_interviewer_score",
    "devils_advocate_score",
    "panel_variance",
    "requires_human_review",
    "verdict_consensus",
    "strengths",
    "concerns",
    "narrative",
    "bias_flag",
    "counterfactual_delta",
]


def rank_candidates(results: list[dict]) -> list[dict]:
    """Sort scored results and assign each a consecutive rank.

    Results are ordered by ``composite_score`` in descending order, breaking
    ties by ``candidate_id`` in ascending order so the ordering is fully
    deterministic (Requirement 8.1). Each result is then given a ``"rank"`` key
    equal to its 1-based position in the sorted order, incrementing by 1
    (Requirement 8.2).

    The function is pure: it does not mutate the input list or any of the dicts
    it contains. A shallow copy is made of every result dict, and the ``rank``
    key is added to the copy.

    Args:
        results: The scored candidate result dicts to rank. Each dict must
            contain at least a numeric ``composite_score`` and a ``candidate_id``
            used as the tie-breaker.

    Returns:
        list[dict]: A new list of shallow-copied result dicts in ranked order,
        each carrying an added ``"rank"`` key holding its consecutive 1-based
        rank.
    """
    # Sort copies so neither the input list nor its dicts are mutated. The
    # composite_score is negated for a descending primary sort while keeping the
    # candidate_id tie-break ascending in a single stable key (Requirement 8.1).
    ranked = sorted(
        (dict(result) for result in results),
        key=lambda result: (-result["composite_score"], result["candidate_id"]),
    )

    # Consecutive integer ranks starting at 1 in sorted order (Requirement 8.2).
    for position, result in enumerate(ranked, start=1):
        result["rank"] = position

    return ranked


def verdict_consensus(persona_verdicts: dict[str, str]) -> str:
    """Collapse the three persona verdicts into a single consensus verdict.

    The consensus is the verdict held by at least two of the three personas
    (Requirement 8.5). Because there are only three personas, at most one
    verdict can reach a count of two or more, so the majority is unambiguous.
    If all three personas disagree (no verdict reaches a count of two), the
    ``hiring_manager`` persona's verdict is used as the tie-breaking fallback
    (Requirement 8.6).

    Args:
        persona_verdicts: Mapping of persona name to that persona's verdict,
            e.g. ``{"hiring_manager": ..., "peer_interviewer": ...,
            "devils_advocate": ...}``. Verdict values come from
            ``{"strong_yes", "yes", "maybe", "no"}``.

    Returns:
        str: The verdict held by at least two personas, or the
        ``hiring_manager`` verdict when no such majority exists.
    """
    counts = Counter(persona_verdicts.values())
    verdict, count = counts.most_common(1)[0]
    if count >= 2:
        return verdict
    # No verdict is shared by >= 2 personas: fall back to the hiring_manager's
    # verdict (Requirement 8.6).
    return persona_verdicts["hiring_manager"]


def write_ranked_csv(results: list[dict], output_dir: Path) -> Path:
    """Write ranked_candidates.csv atomically (no partial file on failure).

    The scored results are first ordered and ranked with :func:`rank_candidates`
    (descending ``composite_score``, ``candidate_id`` tie-break, consecutive
    1-based ranks). Each ranked result is then projected onto the fixed CSV
    schema in the exact column order required by Requirement 8.4. The
    ``verdict_consensus`` column is derived per row from the result's
    ``persona_verdicts`` via :func:`verdict_consensus`, and the ``strengths`` and
    ``concerns`` list columns are pipe-joined (``"|".join(...)``) so an empty
    list serializes to the empty string ``""``.

    The file is written atomically (Requirement 8.7): the DataFrame is first
    written to a uniquely named temporary file in ``output_dir`` and then
    :func:`os.replace` renames it onto the final ``ranked_candidates.csv`` path
    in a single filesystem operation. If serialization or writing fails, the
    temporary file is removed and the exception is re-raised, so no partial
    ``ranked_candidates.csv`` is ever left behind. ``pandas`` is imported lazily
    here so importing :mod:`output.writer` does not require ``pandas``.

    Args:
        results: The scored candidate result dicts to write. Each dict is
            expected to carry ``candidate_id``, ``name``, ``composite_score``,
            ``trajectory_score``, the three persona scores
            (``hiring_manager_score``, ``peer_interviewer_score``,
            ``devils_advocate_score``), ``panel_variance``,
            ``requires_human_review``, ``persona_verdicts`` (a mapping),
            ``strengths`` and ``concerns`` (lists), ``narrative``, ``bias_flag``,
            and ``counterfactual_delta``.
        output_dir: Directory the ``ranked_candidates.csv`` file is written
            into. The directory (and any missing parents) is created if it does
            not already exist.

    Returns:
        Path: The path to the written ``ranked_candidates.csv`` file
        (``output_dir / "ranked_candidates.csv"``).

    Raises:
        Exception: Any error raised while building the rows, constructing the
            DataFrame, or writing the file is propagated after the temporary
            file is cleaned up, guaranteeing no partial output remains.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / _RANKED_CSV_FILENAME

    # Rank first (pure, non-mutating), then project each ranked result onto the
    # fixed CSV schema (Requirement 8.4).
    ranked = rank_candidates(results)
    rows = [
        {
            "rank": result["rank"],
            "candidate_id": result["candidate_id"],
            "name": result["name"],
            "composite_score": result["composite_score"],
            "trajectory_score": result["trajectory_score"],
            "hiring_manager_score": result["hiring_manager_score"],
            "peer_interviewer_score": result["peer_interviewer_score"],
            "devils_advocate_score": result["devils_advocate_score"],
            "panel_variance": result["panel_variance"],
            "requires_human_review": result["requires_human_review"],
            "verdict_consensus": verdict_consensus(result["persona_verdicts"]),
            # Empty list -> "" via str.join (Requirement 8.4).
            "strengths": "|".join(result["strengths"]),
            "concerns": "|".join(result["concerns"]),
            "narrative": result["narrative"],
            "bias_flag": result["bias_flag"],
            "counterfactual_delta": result["counterfactual_delta"],
        }
        for result in ranked
    ]

    frame = pd.DataFrame(rows, columns=_CSV_COLUMNS)

    # Atomic write: serialize to a temp file in the SAME directory as the target
    # so os.replace is an atomic same-filesystem rename, then swap it into place.
    # On any failure, remove the temp file so no partial CSV is left behind
    # (Requirement 8.7).
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{_RANKED_CSV_FILENAME}.", suffix=".tmp", dir=output_dir
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup of the temp file; never mask the original error.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    logger.info("Wrote ranked output for %d candidates to %s", len(rows), target)
    return target
