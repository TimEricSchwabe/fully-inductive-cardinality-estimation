"""Generate a LaTeX booktabs table from per_method_graph.csv.

Rows: graphs (+ Overall).
Columns: one per method, each with two sub-columns (median q-error, log Pearson).
Bolds the best method per row (min median q, max Pearson).

Usage:
    .venv/bin/python scripts/evaluation/make_latex_table.py \\
        output/evaluation/tables/per_method_graph.csv \\
        --overall output/evaluation/tables/per_method.csv \\
        --out output/evaluation/tables/comparison.tex

If --overall is omitted, the Overall row falls back to the mean of per-graph values.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # scripts/

from evaluation.plotting import DISPLAY_NAME, GRAPH_DISPLAY_NAME  # noqa: E402


def fmt_q(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.2f}"


def fmt_r(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.2f}"


def escape(s: str) -> str:
    return s.replace("_", r"\_")


def graph_display(name: str) -> str:
    """Pretty graph label for the table (e.g. 'fb237' -> 'FB15K237')."""
    pretty = GRAPH_DISPLAY_NAME.get(name)
    return pretty if pretty is not None else escape(name)


def method_display(name: str) -> str:
    """Pretty method label for the column header (e.g. 'wj' -> 'WJ')."""
    pretty = DISPLAY_NAME.get(name.lower())
    return pretty if pretty is not None else escape(name)


def render_row(label: str, q_vals: list[float], r_vals: list[float]) -> str:
    best_q = int(np.nanargmin(q_vals)) if np.any(np.isfinite(q_vals)) else -1
    best_r = int(np.nanargmax(r_vals)) if np.any(np.isfinite(r_vals)) else -1
    cells = [label]
    for i in range(len(q_vals)):
        qv = fmt_q(q_vals[i])
        rv = fmt_r(r_vals[i])
        if i == best_q and np.isfinite(q_vals[i]):
            qv = r"\textbf{" + qv + "}"
        if i == best_r and np.isfinite(r_vals[i]):
            rv = r"\textbf{" + rv + "}"
        cells.extend([qv, rv])
    return " & ".join(cells) + r" \\"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph_csv", type=Path, help="per_method_graph.csv")
    p.add_argument(
        "--overall",
        type=Path,
        default=None,
        help="per_method.csv for a true Overall row (falls back to mean across graphs)",
    )
    p.add_argument("--out", type=Path, default=None, help="output .tex (default: stdout)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.graph_csv)

    method_order = ["fice", "gnce", "lmkg", "lss", "wj", "wanderjoin", "cset"]
    present = list(df["method"].unique())
    by_key = {m.lower(): m for m in present}
    methods = [by_key[k] for k in method_order if k in by_key]
    methods += sorted(m for m in present if m.lower() not in set(method_order))
    # Sort graphs by GRAPH_DISPLAY_NAME insertion order (size-ordered),
    # falling back to alphabetical for unknown graphs.
    graph_order_index = {g: i for i, g in enumerate(GRAPH_DISPLAY_NAME.keys())}
    graphs = sorted(
        df["graph"].unique(),
        key=lambda g: (graph_order_index.get(g, len(graph_order_index)), g),
    )

    pivot_q = df.pivot(index="graph", columns="method", values="median_q")
    pivot_r = df.pivot(index="graph", columns="method", values="log_pearson")

    if args.overall is not None:
        overall = pd.read_csv(args.overall).set_index("method")
        overall_q = overall["median_q"]
        overall_r = overall["log_pearson"]
    else:
        overall_q = pivot_q.mean(axis=0)
        overall_r = pivot_r.mean(axis=0)

    n = len(methods)
    col_spec = "l" + "cc" * n

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(
        " " + "".join(f" & \\multicolumn{{2}}{{c}}{{{method_display(m)}}}" for m in methods) + r" \\"
    )
    lines.append(
        " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n))
    )
    lines.append(
        "Graph" + "".join(" &$qE$ &$r$" for _ in methods) + r" \\"
    )
    lines.append(r"\midrule")

    for graph in graphs:
        q_vals = [float(pivot_q.loc[graph, m]) if m in pivot_q.columns and pd.notna(pivot_q.loc[graph, m]) else float("nan") for m in methods]
        r_vals = [float(pivot_r.loc[graph, m]) if m in pivot_r.columns and pd.notna(pivot_r.loc[graph, m]) else float("nan") for m in methods]
        lines.append(render_row(graph_display(graph), q_vals, r_vals))

    lines.append(r"\midrule")
    q_vals = [float(overall_q.get(m, float("nan"))) for m in methods]
    r_vals = [float(overall_r.get(m, float("nan"))) for m in methods]
    lines.append(render_row("Overall", q_vals, r_vals))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
