"""Audit package: counterfactual fairness auditing (phase five).

Contains ``counterfactual.py`` with the ``CounterfactualAuditor``, which builds a
demographically swapped twin of each candidate, re-scores it through the same
scoring pipeline and vector store, and flags scoring deltas that exceed the
configured bias threshold. The auditor also writes ``bias_audit_report.json``.
"""
