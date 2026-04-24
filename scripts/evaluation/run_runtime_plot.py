"""Standalone runtime plot: boxplot + median/IQR line plot, per method.

Reads one CSV per method with columns `n_triple_pattern, runtime` (µs).
Produces LNCS-styled PDFs and a per-(method, size) summary CSV.


uv run python scripts/evaluation/run_runtime_plot.py --csv fice=output/runtime_measurement_dbpedia100k.csv --csv
   gnce=/Users/timschwabe/Projects/GNCE/output/runtime_measurement_dbpedia100k.csv --csv wj=output/runtime_gcare_wj_dbpedia100k.csv --csv cset=output/runtime_gcare_cset_dbpedia100k.csv
   --output-dir output/evaluation/runtime/ --tag dbpedia100k 2>&1 | tail -5 && echo '---summary---' && cat output/evaluation/runtime/runtime_summary_dbpedia100k.csv

Example:
    uv run python scripts/evaluation/run_runtime_plot.py \\
        --csv fice=output/runtime_measurement_dbpedia100k.csv \\
        --csv gnce=/Users/timschwabe/Projects/GNCE/output/runtime_measurement_dbpedia100k.csv \\
        --output-dir output/evaluation/runtime/ \\
        --tag dbpedia100k
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # scripts/

from evaluation.plotting import (  # noqa: E402
    boxplot_runtime_by_triples,
    lineplot_runtime_by_triples,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_csv_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--csv expects method=path, got {value!r}"
        )
    method, path = value.split("=", 1)
    return method.strip(), Path(path).expanduser().resolve()


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, n), grp in df.groupby(["method", "n_triple_pattern"]):
        a = grp["runtime"].to_numpy()
        rows.append({
            "method": method,
            "n_triple_pattern": int(n),
            "count": int(a.size),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["method", "n_triple_pattern"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        action="append",
        type=_parse_csv_arg,
        required=True,
        help="Repeatable: method=path to a runtime CSV "
             "(columns: n_triple_pattern, runtime).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True,
                        help="Suffix for output filenames (e.g. dbpedia100k).")
    args = parser.parse_args()

    out_dir: Path = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for method, path in args.csv:
        logger.info(f"Loading {method}: {path}")
        df = pd.read_csv(path)
        if not {"n_triple_pattern", "runtime"}.issubset(df.columns):
            raise ValueError(
                f"{path} missing required columns; has {list(df.columns)}"
            )
        df = df[["n_triple_pattern", "runtime"]].copy()
        df["method"] = method
        frames.append(df)
    unified = pd.concat(frames, ignore_index=True)
    logger.info(f"Unified: {len(unified)} rows, "
                f"methods={sorted(unified['method'].unique())}")

    box_path = out_dir / f"runtime_boxplot_{args.tag}.pdf"
    line_path = out_dir / f"runtime_lineplot_{args.tag}.pdf"
    summary_path = out_dir / f"runtime_summary_{args.tag}.csv"

    boxplot_runtime_by_triples(unified, box_path)
    lineplot_runtime_by_triples(unified, line_path)
    _summary(unified).to_csv(summary_path, index=False)

    logger.info(f"Wrote {box_path}")
    logger.info(f"Wrote {line_path}")
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
