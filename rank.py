#!/usr/bin/env python3
"""CLI entry point for the competition-mode ranking pipeline.

Reads ``candidates.jsonl``, scores every candidate with zero LLM calls,
and writes ``submission.csv`` — the 100-row, 4-column file required by the
India Runs data-and-AI challenge.

Usage::

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Compute budget: ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU only, no network.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from src.ranker.io import load_candidates_jsonl
from src.ranker.score import ScoredCandidate, score_all, select_top_n
from src.ranker.reasoning import build_reasoning

logger = logging.getLogger(__name__)

# Submission spec: columns must appear in this exact order.
CSV_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]

# Maximum characters for the reasoning field (spec allows longer, but
# the sample uses ≤400 chars).
MAX_REASONING_CHARS = 380


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic candidate ranking for the India Runs competition.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to the candidates.jsonl input file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./submission.csv"),
        help="Path to the output submission.csv (default: ./submission.csv).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to include (default: 100).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Cap on total candidates to process (default: all).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def write_submission_csv(
    ranked: List[ScoredCandidate],
    cand_map: Dict[str, dict],
    out_path: Path,
) -> None:
    """Write the submission CSV in the format required by the competition.

    Args:
        ranked: Exactly 100 :class:`ScoredCandidate` objects, ordered by
            score descending.
        cand_map: Mapping candidate_id → original normalized candidate dict
            (needed for reasoning generation, which reads career_history
            and signals not carried in ScoredCandidate).
        out_path: Destination file path.
    """
    ranked = sorted(ranked, key=lambda s: (-round(s.score, 4), s.candidate_id))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, scored in enumerate(ranked, start=1):
            cand = cand_map.get(scored.candidate_id, {})
            reasoning = build_reasoning(cand, scored.features)
            if len(reasoning) > MAX_REASONING_CHARS:
                reasoning = reasoning[:MAX_REASONING_CHARS].rstrip()
            writer.writerow({
                "candidate_id": scored.candidate_id,
                "rank": i,
                "score": round(scored.score, 4),
                "reasoning": reasoning,
            })


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.candidates.exists():
        logger.error("Candidates file not found: %s", args.candidates)
        return 1

    t0 = time.time()
    logger.info("Starting competition ranking pipeline")
    logger.info("Candidates file: %s", args.candidates)

    # 1. Stream-load all candidates (normalizes fields for downstream)
    candidates = list(load_candidates_jsonl(args.candidates))
    logger.info("Loaded %d candidates", len(candidates))

    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
        logger.info("Capped to %d candidates", len(candidates))

    # Build lookup map: candidate_id → normalized candidate dict
    # (needed for reasoning generation downstream)
    cand_map: Dict[str, dict] = {c["candidate_id"]: c for c in candidates}

    # 2. Score every candidate (deterministic, no LLM)
    scored = score_all(candidates)
    logger.info("Scored %d candidates", len(scored))

    # 3. Select top N (safe-first strategy)
    top = select_top_n(scored, n=args.top_n)
    logger.info("Selected top %d candidates", len(top))

    # 4. Write submission CSV
    write_submission_csv(top, cand_map, args.out)

    elapsed = time.time() - t0
    honeypots = sum(1 for s in top if s.is_honeypot)
    logger.info(
        "Done — %d candidates ranked, %d honeypots in top %d, %.1fs elapsed",
        len(top), honeypots, args.top_n, elapsed,
    )

    if honeypots / max(1, len(top)) > 0.10:
        logger.warning(
            "Honeypot rate %.1f%% exceeds 10%% — submission may be disqualified",
            100 * honeypots / max(1, len(top)),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
