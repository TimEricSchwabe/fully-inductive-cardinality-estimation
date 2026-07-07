"""Convert LSS loco_<graph>/predictions.csv into the runtime-CSV schema used by
scripts/evaluation/run_runtime_plot.py.

LSS logs one row per query with an `eval_time_s` column (per-query inference
latency in seconds). We pull `query_size` (triple count) and `eval_time_s`,
convert s → µs.

Usage:
    uv run python scripts/evaluation/runtime_from_lss.py \\
        --lss-root output/final/lss \\
        --graph dbpedia100k \\
        --output output/runtime_lss_dbpedia100k.csv
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
    parser.add_argument("--lss-root", type=Path, required=True,
                        help="path to output/final/lss")
    parser.add_argument("--graph", type=str, required=True,
                        help="graph folder suffix, e.g. dbpedia100k → loco_dbpedia100k")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    csv_path = args.lss_root / f"loco_{args.graph}" / "predictions.csv"
    if not csv_path.exists():
        raise SystemExit(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["query_size", "eval_time_s"])
    df["runtime_us"] = df["eval_time_s"] * 1_000_000.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_triple_pattern", "runtime"])
        for n, r in zip(df["query_size"].astype(int), df["runtime_us"]):
            writer.writerow([int(n), f"{r:.3f}"])

    logger.info(f"Wrote {len(df)} rows to {args.output}")
    logger.info(f"Per-size medians (µs):")
    med = df.groupby("query_size")["runtime_us"].median().round(1)
    for n, v in med.items():
        logger.info(f"  n={int(n):2d}: {v}")


if __name__ == "__main__":
    main()
