"""Transductive cross-method evaluation CLI.

Evaluates each method on graphs it was *trained* on, using a deterministic
target-graph -> leave-out-run mapping (see transductive_mapping.py).

Example:
    .venv/bin/python scripts/evaluation/run_transductive.py \\
        --fice-root output/final/FICE \\
        --gnce-root output/final/GNCE \\
        --gcare-root output/final/gcare
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # scripts/

from evaluation.adapters.fice_transductive import FiceTransductiveAdapter  # noqa: E402
from evaluation.adapters.gcare import GcareAdapter  # noqa: E402
from evaluation.adapters.gnce_transductive import GnceTransductiveAdapter  # noqa: E402
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
    calibration_overall,
    calibration_per_graph,
    ccdf_overall,
    ccdf_per_graph,
    cdf_overall,
    cdf_per_graph,
    headline_bar_median_q,
    relative_improvement_bar,
    scatter_overall_panel,
    scatter_per_graph,
    signed_boxplot_overall,
    signed_boxplot_per_cardinality_bin,
    signed_boxplot_per_graph,
    signed_boxplot_per_query_type,
    signed_violin_overall,
    signed_violin_per_cardinality_bin,
    signed_violin_per_graph,
    signed_violin_per_query_type,
)

logger = logging.getLogger("evaluation.transductive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fice-root", type=Path, default=None, help="path to output/final/FICE")
    p.add_argument("--gnce-root", type=Path, default=None, help="path to output/final/GNCE")
    p.add_argument("--gcare-root", type=Path, default=None, help="path to output/final/gcare (feeds both wj and cset)")
    p.add_argument("--out", type=Path, default=Path("output/evaluation/transductive"), help="output directory")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    method_adapters = [
        ("fice", args.fice_root, FiceTransductiveAdapter()),
        ("gnce", args.gnce_root, GnceTransductiveAdapter()),
        ("wj", args.gcare_root, GcareAdapter("wj")),
        ("cset", args.gcare_root, GcareAdapter("cset")),
    ]
    frames: list[pd.DataFrame] = []
    for method, root, adapter in method_adapters:
        if root is None:
            continue
        if not root.exists():
            logger.warning("Root for %s does not exist: %s", method, root)
            continue
        logger.info("Loading %s from %s", method, root)
        df = adapter.load(root)
        logger.info("%s: %d rows", method, len(df))
        frames.append(df)

    if not frames:
        logger.error("No method data loaded. Provide --fice-root, --gnce-root, and/or --gcare-root.")
        sys.exit(1)

    unified = pd.concat(frames, ignore_index=True)
    excluded_types = {"ind_path", "ind_star"}
    before = len(unified)
    unified = unified[~unified["query_type"].isin(excluded_types)].reset_index(drop=True)
    logger.info("Excluded %d rows with query_type in %s", before - len(unified), sorted(excluded_types))
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
    boxplot_by_query_type(unified, plots_dir / "boxplot_by_query_type.pdf")
    cdf_overall(unified, plots_dir)
    cdf_per_graph(unified, plots_dir / "cdf_per_graph")
    scatter_per_graph(unified, plots_dir / "scatter_per_graph")
    scatter_overall_panel(unified, plots_dir / "scatter_overall_panel.pdf")
    bias_histogram(unified, plots_dir / "bias_histogram.pdf")
    headline_bar_median_q(tables["per_method_graph"], plots_dir / "headline_median_q_per_graph.pdf")
    signed_boxplot_overall(unified, plots_dir / "signed_boxplot_overall.pdf")
    signed_boxplot_per_graph(unified, plots_dir / "signed_boxplot_per_graph.pdf")
    signed_boxplot_per_query_type(unified, plots_dir / "signed_boxplot_per_query_type.pdf")
    signed_boxplot_per_cardinality_bin(unified, plots_dir / "signed_boxplot_per_cardinality_bin.pdf")
    signed_violin_overall(unified, plots_dir / "signed_violin_overall.pdf")
    signed_violin_per_graph(unified, plots_dir / "signed_violin_per_graph.pdf")
    signed_violin_per_query_type(unified, plots_dir / "signed_violin_per_query_type.pdf")
    signed_violin_per_cardinality_bin(unified, plots_dir / "signed_violin_per_cardinality_bin.pdf")
    ccdf_overall(unified, plots_dir)
    ccdf_per_graph(unified, plots_dir / "ccdf_per_graph")
    calibration_overall(unified, plots_dir)
    calibration_per_graph(unified, plots_dir / "calibration_per_graph")
    relative_improvement_bar(tables["per_method_graph"], plots_dir / "relative_improvement.pdf")
    logger.info("Plots written under %s", plots_dir)


if __name__ == "__main__":
    main()
