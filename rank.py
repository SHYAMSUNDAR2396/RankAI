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
import logging
import sys
import time
from pathlib import Path

import config


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.candidates.exists():
        logging.error("Candidates file not found: %s", args.candidates)
        return 1

    t0 = time.time()
    logging.info("Starting competition ranking pipeline")

    from pipeline.competition import CompetitionRanker

    ranker = CompetitionRanker()
    ranked = ranker.rank(
        args.candidates,
        top_n=args.top_n,
        max_candidates=args.max_candidates,
    )

    ranker.write_csv(ranked, args.out)

    elapsed = time.time() - t0
    honeypots = sum(1 for r in ranked if r.get("_is_honeypot"))
    logging.info(
        "Done — %d candidates ranked, %d honeypots in top %d, %.1fs elapsed",
        len(ranked), honeypots, args.top_n, elapsed,
    )

    if honeypots / max(1, len(ranked)) > config.HONEYPOT_HONEYPOT_RATE_DISQUALIFY:
        logging.warning(
            "Honeypot rate %.1f%% exceeds %.1f%% — submission may be disqualified",
            100 * honeypots / max(1, len(ranked)),
            100 * config.HONEYPOT_HONEYPOT_RATE_DISQUALIFY,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
