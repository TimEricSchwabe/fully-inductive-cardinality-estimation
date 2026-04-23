#!/usr/bin/env python3
"""Count queries per graph per query type by inspecting data/<graph>_<qtype>_queries/raw/."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

GRAPHS = [
    "swdf",
    "fb237_v4",
    "fb237_v4_ind",
    "wn18rr_v4",
    "wn18rr_v4_ind",
    "dbpedia100k",
    "lubm",
    "wikidata",
    "hetionet",
    "yago",
    "codex_l",
    "aids",
]


UNTYPED_IS_STAR = {"lubm", "wikidata"}


def match_graph(dirname: str) -> Optional[tuple[str, str]]:
    """Return (graph, qtype) by longest-prefix match against GRAPHS, or None if no match.

    For legacy compatibility, the untyped `lubm_queries/` and `wikidata_queries/` directories
    are star queries despite the missing suffix.
    """
    stem = dirname.removesuffix("_queries")
    best = None
    for g in GRAPHS:
        if stem == g or stem.startswith(g + "_"):
            if best is None or len(g) > len(best):
                best = g
    if best is None:
        return None
    qtype = stem[len(best):].lstrip("_")
    if not qtype:
        qtype = "star" if best in UNTYPED_IS_STAR else "(none)"
    return best, qtype


def count_queries(query_dir: Path) -> int:
    """Pick a single canonical raw file and return the number of queries it contains.

    Some dirs contain two duplicate files (e.g. `fb237_v4_star_queries/raw/{59651327,fb237_v4_star_queries}`);
    we always take just the first alphabetically to avoid double counting. For files where `json.load`
    fails (truncated legacy data), fall back to counting top-level entries via a `{"x":` pattern.
    """
    raw_dir = query_dir / "raw"
    if not raw_dir.is_dir():
        return 0
    data_files = sorted(
        f for f in raw_dir.iterdir()
        if f.is_file() and not f.name.endswith(".cache.json")
    )
    if not data_files:
        return 0
    f = data_files[0]
    with open(f, "r", encoding="utf-8") as fh:
        try:
            obj = json.load(fh)
            if isinstance(obj, list):
                return len(obj)
            return 0
        except json.JSONDecodeError:
            pass
    # Fallback: scan for top-level entry markers (each query starts with `{"x":`).
    print(f"[warn] {f} is not valid JSON — falling back to regex count of '{{\"x\":' markers")
    count = 0
    with open(f, "r", encoding="utf-8") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), ""):
            count += chunk.count('{"x":')
    return count


def main():
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    qtypes: set[str] = set()

    for d in sorted(DATA_DIR.glob("*_queries")):
        if not d.is_dir():
            continue
        match = match_graph(d.name)
        if match is None:
            print(f"[skip] unrecognized graph for: {d.name}")
            continue
        graph, qtype = match
        n = count_queries(d)
        counts[graph][qtype] = counts[graph].get(qtype, 0) + n
        qtypes.add(qtype)

    col_order = [
        "star", "path", "cycle", "flower", "diamond",
        "snowflake", "binary_tree", "tree", "path_star_ends", "(none)",
    ]
    cols = [q for q in col_order if q in qtypes] + sorted(qtypes - set(col_order))

    header = f"{'graph':<18}" + "".join(f"{q:>16}" for q in cols) + f"{'total':>12}"
    print(header)
    print("-" * len(header))
    for g in GRAPHS:
        if g not in counts:
            continue
        row = counts[g]
        total = sum(row.values())
        line = f"{g:<18}" + "".join(f"{row.get(q, 0):>16,}" for q in cols) + f"{total:>12,}"
        print(line)


if __name__ == "__main__":
    main()
