"""Convert LMKG all_predictions.csv into the runtime-CSV schema used by
scripts/evaluation/run_runtime_plot.py.

LMKG logs one row per query with an `est_ms` column (per-query inference
latency in milliseconds). We filter to dataset_name matching the given graph
prefix, pull `query_size` (triple count) and `est_ms`, convert ms → µs.

Only the `star` and `path` shapes are available for dbpedia100k under LMKG.

Usage:
    uv run python scripts/evaluation/runtime_from_lmkg.py \\
        --predictions-csv output/final/lmkg-runtime/all_predictions.csv \\
        --graph-prefix dbpedia100k \\
        --output output/runtime_lmkg_dbpedia100k.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--graph-prefix", type=str, required=True,
                        help="e.g. dbpedia100k — matches dataset_name substring.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv,
                     usecols=["dataset_name", "query_size", "est_ms"])
    mask = df["dataset_name"].astype(str).str.contains(args.graph_prefix)
    sub = df.loc[mask, ["query_size", "est_ms"]].copy()
    if sub.empty:
        raise SystemExit(f"No rows in {args.predictions_csv} match "
                         f"graph-prefix={args.graph_prefix!r}")

    sub["runtime_us"] = sub["est_ms"] * 1000.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_triple_pattern", "runtime"])
        for n, r in zip(sub["query_size"].astype(int), sub["runtime_us"]):
            writer.writerow([int(n), f"{r:.3f}"])

    logger.info(f"Wrote {len(sub)} rows to {args.output}")
    logger.info(f"Per-size medians (µs):")
    med = sub.groupby("query_size")["runtime_us"].median().round(1)
    for n, v in med.items():
        logger.info(f"  n={int(n):2d}: {v}")


if __name__ == "__main__":
    main()
