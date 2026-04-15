"""Cross-method evaluation CLI.

Example:
    .venv/bin/python scripts/evaluation/run.py \\
        --fice-root output/final/FICE \\
        --gnce-root output/final/GNCE \\
        --out output/evaluation/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script: put scripts/ on path so `evaluation` is importable as a package.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # scripts/

from evaluation.adapters import ADAPTER_REGISTRY  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    assign_cardinality_bin,
    compute_tables,
    per_query_q_error,
)
from evaluation.plotting import (  # noqa: E402
    bias_histogram,
    boxplot_by_query_type,
    boxplots_overall,
    boxplots_per_graph,
    boxplots_per_query_dataset,
    cdf_overall,
    cdf_per_graph,
    headline_bar_median_q,
    scatter_overall_panel,
    scatter_per_graph,
)

logger = logging.getLogger("evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fice-root", type=Path, default=None, help="path to output/final/FICE")
    p.add_argument("--gnce-root", type=Path, default=None, help="path to output/final/GNCE")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    method_roots = {"fice": args.fice_root, "gnce": args.gnce_root}
    frames: list[pd.DataFrame] = []
    for method, root in method_roots.items():
        if root is None:
            continue
        if not root.exists():
            logger.warning("Root for %s does not exist: %s", method, root)
            continue
        logger.info("Loading %s from %s", method, root)
        df = ADAPTER_REGISTRY[method].load(root)
        logger.info("%s: %d rows", method, len(df))
        frames.append(df)

    if not frames:
        logger.error("No method data loaded. Provide --fice-root and/or --gnce-root.")
        sys.exit(1)

    unified = pd.concat(frames, ignore_index=True)
    unified["q_error"] = per_query_q_error(
        unified["prediction"].to_numpy(), unified["true_value"].to_numpy()
    )
    unified["cardinality_bin"] = assign_cardinality_bin(unified["true_value"])

    out_dir = args.out
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "unified_predictions.csv"
    unified.to_csv(raw_path, index=False)
    logger.info("Wrote %s (%d rows)", raw_path, len(unified))

    tables = compute_tables(unified)
    for name, tbl in tables.items():
        path = tables_dir / f"{name}.csv"
        tbl.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(tbl))

    boxplots_overall(unified, plots_dir)
    boxplots_per_graph(unified, plots_dir / "boxplot_per_graph")
    boxplots_per_query_dataset(unified, plots_dir / "boxplot_per_query_dataset")
    boxplot_by_query_type(unified, plots_dir / "boxplot_by_query_type.png")
    cdf_overall(unified, plots_dir)
    cdf_per_graph(unified, plots_dir / "cdf_per_graph")
    scatter_per_graph(unified, plots_dir / "scatter_per_graph")
    scatter_overall_panel(unified, plots_dir / "scatter_overall_panel.png")
    bias_histogram(unified, plots_dir / "bias_histogram.png")
    headline_bar_median_q(tables["per_method_graph"], plots_dir / "headline_median_q_per_graph.png")
    logger.info("Plots written under %s", plots_dir)


if __name__ == "__main__":
    main()
