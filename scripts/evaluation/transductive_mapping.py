"""Target-graph → leave-out-run mapping for transductive evaluation.

Rule: use the `aids` run (aids held out, everything else in training) for every
target graph; for target == aids itself, fall back to the `wn18rr` run.
Keys are canonical graph names (see scripts/evaluation/naming.py).
Values are the held-out shortname (canonical) of the run to source predictions
from — this shortname is what both the FICE folder name (aids, wn18rr, ...) and
the GNCE folder name prefix (loo_aids_*, loo_wn18rr_*) encode.
"""
from __future__ import annotations

TARGET_TO_HELDOUT: dict[str, str] = {
    "aids": "wn18rr",
    "codex_l": "aids",
    "dbpedia100k": "aids",
    "fb237": "aids",
    "hetionet": "aids",
    "lubm": "aids",
    "swdf": "aids",
    "wikidata": "aids",
    "wn18rr": "aids",
    "yago": "aids",
}
