"""Graph-name canonicalization across methods.

Canonical names follow the GNCE shortname convention so a single lookup works
for both FICE and GNCE outputs. Edit GRAPH_NAME_MAP to add new methods.
"""
from __future__ import annotations

GRAPH_NAME_MAP: dict[str, str] = {
    # FICE native dataset_name values
    "fb237_v4": "fb237",
    "swdf": "swdf",
    "hetionet": "hetionet",
    "lubm": "lubm",
    "wikidata": "wikidata",
    "codex_l": "codex_l",
    "wn18rr_v4": "wn18rr",
    "wn18rr": "wn18rr",
    "yago": "yago",
    "dbpedia": "dbpedia100k",
    "aids": "aids",
    # GNCE shortnames (from loo_<short>_<ts>)
    "fb237": "fb237",
    "dbpedia100k": "dbpedia100k",
}


def canonicalize_graph(name: str) -> str:
    return GRAPH_NAME_MAP.get(name, name)


QUERY_TYPE_MAP: dict[str, str] = {"binary_tree": "tree"}


def canonicalize_query_type(name: str) -> str:
    return QUERY_TYPE_MAP.get(name, name)
