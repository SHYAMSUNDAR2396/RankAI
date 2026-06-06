"""Output package: artifact writers.

Contains ``writer.py`` which produces the two pipeline artifacts:

- ``ranked_candidates.csv`` -- the ranked candidate report (written atomically).
- the ranked/consensus helpers used to build those rows.

The bias audit JSON (``bias_audit_report.json``) is written by the
``CounterfactualAuditor`` in the ``audit`` package, which shares responsibility
for the audit artifact.

Note: generated run artifacts (the CSV and JSON) are written to the operator's
chosen ``--output-dir`` at runtime; this package holds the writing logic only.
"""
