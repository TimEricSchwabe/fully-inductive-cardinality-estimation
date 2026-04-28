"""Convert gCARE results.json files into the runtime-CSV schema used by
scripts/evaluation/run_runtime_plot.py.

For each method folder under {gcare-root}/{graph}_{qtype}/{method}/, the
results.json has one entry per query with `triples` and `exec_time_total`.
With the current gCARE wrapper (output/final/gcare-runtime), the graph index
is loaded once per shape and the C++ binary's internal timing wraps only the
`Estimator::Run` call. `exec_time_total` is already the mean-per-estimation
time in seconds (the binary averages across its --iteration reps internally).

Outputs: one CSV per method with columns `n_triple_pattern, runtime` in µs,
matching the FICE/GNCE runtime CSV schema.

Usage:
    uv run python scripts/evaluation/runtime_from_gcare.py \\
        --gcare-root output/final/gcare-runtime/ \\
        --graph-prefix dbpedia100k \\
        --method wj \\
        --output output/runtime_gcare_wj_dbpedia100k.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METHODS = ["wj", "cset", "bsk", "sumrdf"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcare-root", type=Path, required=True)
    parser.add_argument("--graph-prefix", type=str, required=True,
                        help="e.g. dbpedia100k — matches {prefix}_{qtype} folders.")
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root: Path = args.gcare_root.expanduser().resolve()
    rows: list[tuple[int, float]] = []

    folders = sorted(p for p in root.iterdir()
                     if p.is_dir() and (p.name == args.graph_prefix
                                        or p.name.startswith(args.graph_prefix + "_")))
    if not folders:
        raise SystemExit(f"No folders under {root} matching {args.graph_prefix}_*")

    total = 0
    for folder in folders:
        results_path = folder / args.method / "results.json"
        if not results_path.exists():
            logger.warning(f"Skipping {folder.name}: no {args.method}/results.json")
            continue
        with open(results_path) as f:
            entries = json.load(f)
        for e in entries:
            n = len(e["triples"])
            secs = float(e["exec_time_total"])
            rows.append((n, secs * 1e6))
        logger.info(f"{folder.name}: {len(entries)} queries")
        total += len(entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_triple_pattern", "runtime"])
        for n, r in rows:
            writer.writerow([n, f"{r:.3f}"])

    logger.info(f"Wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
