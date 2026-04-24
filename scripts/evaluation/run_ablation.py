"""FICE ablation evaluation for a single held-out graph (DBpedia100k).

Loads the three runs under <fice-root>:
  - dbpedia                        -> fice              (full FICE)
  - dbpedia-ablation-counts        -> fice_no_occ       (count_features=none)
  - dbpedia-ablation-embeddings    -> fice_no_emb       (disable_embedding_gnn=true)

Tags them as distinct methods, pipes through the same compute_tables() used
in run.py, and renders a LaTeX booktabs table in the style of
make_latex_table.py (one row per graph + Overall, qE and log-Pearson r per
method, best bolded per row).

Example:
    .venv/bin/python scripts/evaluation/run_ablation.py \\
        --fice-root output/final/FICE \\
        --out output/evaluation/ablation/
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

from evaluation.adapters.fice import FiceAdapter, read_held_out  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    assign_cardinality_bin,
    compute_tables,
    per_query_q_error,
)
from evaluation import plotting  # noqa: E402

logger = logging.getLogger("ablation")


ABLATION_RUNS: list[tuple[str, str, str]] = [
    ("dbpedia",                     "fice",        r"FICE"),
    ("dbpedia-ablation-embeddings", "fice_no_emb", r"FICE w/o embeddings"),
    ("dbpedia-ablation-counts",     "fice_no_occ", r"FICE w/o occurrences"),
]


def register_ablation_styles() -> None:
    """Patch plotting module globals so ablation methods get distinct colors/linestyles/labels."""
    plotting.METHOD_ORDER = ("fice", "fice_no_emb", "fice_no_occ")
    plotting.METHOD_COLORS.update({
        "fice":        "#1f77b4",  # blue (matches main paper)
        "fice_no_emb": "#d62728",  # red
        "fice_no_occ": "#2ca02c",  # green
    })
    plotting.METHOD_LINESTYLE.update({
        "fice":        "-",
        "fice_no_emb": "--",
        "fice_no_occ": "-.",
    })
    plotting.DISPLAY_NAME.update({
        "fice":        "FICE",
        "fice_no_emb": "FICE w/o embeddings",
        "fice_no_occ": "FICE w/o occurrences",
    })


def load_run(root: Path, run_dir: str, method: str) -> pd.DataFrame:
    adapter = FiceAdapter()
    csv_path = root / run_dir / "logs" / "predictions.csv"
    cfg_path = root / run_dir / "config.yaml"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    held_out = read_held_out(cfg_path)
    if not held_out:
        raise RuntimeError(f"No type:validation datasets in {cfg_path}")
    df = adapter._load_run(csv_path, held_out)
    df["method"] = method
    return df


def _fmt(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.2f}"


def _row(label: str, q_vals: list[float], r_vals: list[float]) -> str:
    best_q = int(np.nanargmin(q_vals)) if np.any(np.isfinite(q_vals)) else -1
    best_r = int(np.nanargmax(r_vals)) if np.any(np.isfinite(r_vals)) else -1
    cells = [label]
    for i in range(len(q_vals)):
        qv, rv = _fmt(q_vals[i]), _fmt(r_vals[i])
        if i == best_q and np.isfinite(q_vals[i]):
            qv = r"\textbf{" + qv + "}"
        if i == best_r and np.isfinite(r_vals[i]):
            rv = r"\textbf{" + rv + "}"
        cells.extend([qv, rv])
    return " & ".join(cells) + r" \\"


def render_compact(
    per_method: pd.DataFrame,
    method_order: list[str],
    method_labels: list[str],
) -> str:
    """Methods-as-rows table. Clean layout when there is only one held-out graph."""
    overall = per_method.set_index("method")
    q_vals = [float(overall["median_q"].get(m, float("nan"))) for m in method_order]
    r_vals = [float(overall["log_pearson"].get(m, float("nan"))) for m in method_order]
    best_q = int(np.nanargmin(q_vals)) if np.any(np.isfinite(q_vals)) else -1
    best_r = int(np.nanargmax(r_vals)) if np.any(np.isfinite(r_vals)) else -1

    lines: list[str] = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Median $qE$ & log-Pearson $r$ \\",
        r"\midrule",
    ]
    for i, lbl in enumerate(method_labels):
        qv = _fmt(q_vals[i])
        rv = _fmt(r_vals[i])
        if i == best_q and np.isfinite(q_vals[i]):
            qv = r"\textbf{" + qv + "}"
        if i == best_r and np.isfinite(r_vals[i]):
            rv = r"\textbf{" + rv + "}"
        lines.append(f"{lbl} & {qv} & {rv} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_per_shape(
    per_queryset: pd.DataFrame,
    method_order: list[str],
    method_labels: list[str],
) -> str:
    """Per-query-shape breakdown: shapes as rows, methods as column groups (qE, r)."""
    df = per_queryset.copy()
    df["shape"] = df["query_dataset"].str.split("-", n=1).str[1]
    shape_order = sorted(df["shape"].unique())
    pivot_q = df.pivot(index="shape", columns="method", values="median_q")
    pivot_r = df.pivot(index="shape", columns="method", values="log_pearson")

    n = len(method_order)
    col_spec = "l" + "cc" * n
    lines: list[str] = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " " + "".join(f" & \\multicolumn{{2}}{{c}}{{{lbl}}}" for lbl in method_labels) + r" \\",
        " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n)),
        "Query shape" + "".join(" & $qE$ & $r$" for _ in method_order) + r" \\",
        r"\midrule",
    ]
    for shape in shape_order:
        q_vals = [
            float(pivot_q.loc[shape, m]) if m in pivot_q.columns and pd.notna(pivot_q.loc[shape, m]) else float("nan")
            for m in method_order
        ]
        r_vals = [
            float(pivot_r.loc[shape, m]) if m in pivot_r.columns and pd.notna(pivot_r.loc[shape, m]) else float("nan")
            for m in method_order
        ]
        lines.append(_row(shape, q_vals, r_vals))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_latex(
    per_graph: pd.DataFrame,
    per_method: pd.DataFrame,
    method_order: list[str],
    method_labels: list[str],
) -> str:
    pivot_q = per_graph.pivot(index="graph", columns="method", values="median_q")
    pivot_r = per_graph.pivot(index="graph", columns="method", values="log_pearson")
    overall = per_method.set_index("method")

    n = len(method_order)
    col_spec = "l" + "cc" * n
    lines: list[str] = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    lines.append(
        " " + "".join(f" & \\multicolumn{{2}}{{c}}{{{lbl}}}" for lbl in method_labels) + r" \\"
    )
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n)))
    lines.append("Graph" + "".join(" & $qE$ & $r$" for _ in method_order) + r" \\")
    lines.append(r"\midrule")

    for graph in pivot_q.index:
        q_vals = [
            float(pivot_q.loc[graph, m]) if m in pivot_q.columns and pd.notna(pivot_q.loc[graph, m]) else float("nan")
            for m in method_order
        ]
        r_vals = [
            float(pivot_r.loc[graph, m]) if m in pivot_r.columns and pd.notna(pivot_r.loc[graph, m]) else float("nan")
            for m in method_order
        ]
        lines.append(_row(str(graph), q_vals, r_vals))

    lines.append(r"\midrule")
    q_vals = [float(overall["median_q"].get(m, float("nan"))) for m in method_order]
    r_vals = [float(overall["log_pearson"].get(m, float("nan"))) for m in method_order]
    lines.append(_row("Overall", q_vals, r_vals))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fice-root", type=Path, default=Path("output/final/FICE"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    tables_dir = args.out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for run_dir, method, _label in ABLATION_RUNS:
        df = load_run(args.fice_root, run_dir, method)
        logger.info("%s (%s): %d rows", run_dir, method, len(df))
        frames.append(df)

    unified = pd.concat(frames, ignore_index=True)

    excluded = {"ind_path", "ind_star"}
    before = len(unified)
    unified = unified[~unified["query_type"].isin(excluded)].reset_index(drop=True)
    logger.info("Excluded %d rows with query_type in %s", before - len(unified), sorted(excluded))

    unified["q_error"] = per_query_q_error(
        unified["prediction"].to_numpy(), unified["true_value"].to_numpy()
    )
    unified["cardinality_bin"] = assign_cardinality_bin(unified["true_value"])

    unified.to_pickle(args.out / "ablation_unified.pkl")
    unified.to_csv(args.out / "ablation_unified.csv", index=False)
    logger.info("Wrote unified frame: %d rows", len(unified))

    tables = compute_tables(unified)
    for name, tbl in tables.items():
        path = tables_dir / f"{name}.csv"
        tbl.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(tbl))

    method_order = [m for _, m, _ in ABLATION_RUNS]
    method_labels = [lbl for _, _, lbl in ABLATION_RUNS]

    latex = render_latex(
        tables["per_method_graph"],
        tables["per_method"],
        method_order,
        method_labels,
    )
    (tables_dir / "ablation.tex").write_text(latex)

    compact = render_compact(tables["per_method"], method_order, method_labels)
    (tables_dir / "ablation_compact.tex").write_text(compact)

    per_shape = render_per_shape(tables["per_method_graph_queryset"], method_order, method_labels)
    (tables_dir / "ablation_per_shape.tex").write_text(per_shape)

    logger.info("Wrote ablation.tex, ablation_compact.tex, ablation_per_shape.tex under %s", tables_dir)
    print("=== ablation.tex ===")
    print(latex)
    print("=== ablation_compact.tex ===")
    print(compact)
    print("=== ablation_per_shape.tex ===")
    print(per_shape)

    # Plots — register ablation styles and reuse the main plotting helpers.
    plots_dir = args.out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    register_ablation_styles()

    plotting.ccdf_overall(unified, plots_dir)
    plotting.signed_boxplot_overall(unified, plots_dir / "signed_boxplot_overall.pdf")
    plotting.signed_boxplot_per_query_type(unified, plots_dir / "signed_boxplot_per_query_type.pdf")
    plotting.signed_boxplot_per_cardinality_bin(unified, plots_dir / "signed_boxplot_per_cardinality_bin.pdf")
    logger.info("Wrote plots under %s", plots_dir)


if __name__ == "__main__":
    main()
