"""Boxplots (grouped by cardinality bin) and q-error CDFs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .schema import CARDINALITY_BIN_LABELS

# LNCS single-column figure geometry. LNCS text width is ~4.6 in; there is no
# two-column layout, so we cap all figures at COL_W and scale height.
COL_W: float = 4.6
GOLDEN: float = 0.618

METHOD_ORDER: tuple[str, ...] = ("fice", "gnce", "lmkg", "lss", "wj", "cset")
METHOD_COLORS: dict[str, str] = {
    "fice": "#1f77b4",  # blue
    "gnce": "#ff7f0e",  # orange
    "lmkg": "#2ca02c",  # green
    "wj":   "#d62728",  # red
    "cset": "#9467bd",  # purple
    "lss":  "#8c564b",  # brown
}
METHOD_LINESTYLE: dict[str, object] = {
    "fice": "-",
    "gnce": "--",
    "lmkg": "-.",
    "wj":   ":",
    "cset": (0, (3, 1, 1, 1)),
    "lss":  (0, (1, 1)),  # dotted
}
DISPLAY_NAME: dict[str, str] = {
    "fice": "FICE",
    "gnce": "GNCE",
    "lmkg": "LMKG",
    "wj":   "WJ",
    "cset": "CSet",
    "lss":  "LSS",
}
GRAPH_DISPLAY_NAME: dict[str, str] = {
    "wn18rr": "WN18RR",
    "fb237": "FB15K237",
    "swdf": "SWDF",
    "codex_l": "CoDEx-L",
    "dbpedia100k": "DBpedia100k",
    "aids": "AIDS",
    "hetionet": "Hetionet",
    "lubm": "LUBM",
    "wikidata": "WIKIDATA",
    "yago": "YAGO",
}


def _setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.weight": "normal",
        "mathtext.fontset": "stixsans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


_setup_rc()


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique())
    extras = sorted(m for m in present if m not in METHOD_COLORS)
    return [m for m in METHOD_ORDER if m in present] + extras


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#7f7f7f")


def _linestyle(method: str) -> object:
    return METHOD_LINESTYLE.get(method, "-")


def _display(method: str) -> str:
    return DISPLAY_NAME.get(method, method)


def _display_graph(graph: str) -> str:
    return GRAPH_DISPLAY_NAME.get(graph, graph)


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def _signed_q_error(sub: pd.DataFrame, eps: float = 1e-7) -> np.ndarray:
    p = np.maximum(sub["prediction"].to_numpy(), eps)
    t = np.maximum(sub["true_value"].to_numpy(), eps)
    over = p >= t
    out = np.empty_like(p, dtype=float)
    out[over] = p[over] / t[over]
    out[~over] = -(t[~over] / p[~over])
    return out


def _signed_log_q(sub: pd.DataFrame, eps: float = 1e-7) -> np.ndarray:
    p = np.maximum(sub["prediction"].to_numpy(), eps)
    t = np.maximum(sub["true_value"].to_numpy(), eps)
    return np.log10(p / t)


def _signed_log_tick(y: float, _pos: int) -> str:
    # y is log10(pred/true); show absolute Q-error magnitude as 10^k
    # (except the origin, shown as "1"). Sign is encoded by the axis
    # direction via the under ↔ over label.
    if abs(y) < 1e-9:
        return "1"
    av = abs(y)
    if abs(av - round(av)) < 1e-6:
        return rf"$10^{{{int(round(av))}}}$"
    return f"{10 ** av:g}"


def _symlog_signed_tick(y: float, _pos: int) -> str:
    # symlog axis: blank the linear-scale 0 (it overlaps 10^0) and render
    # all magnitudes through mathtext so weights match (plain "1" in the
    # sans font renders heavier than $10^k$ under stixsans mathtext).
    # Sign is conveyed by the y-axis label.
    if abs(y) < 1e-9:
        return ""
    av = abs(y)
    if abs(av - 1.0) < 1e-9:
        return r"$1$"
    k = np.log10(av)
    if abs(k - round(k)) < 1e-6:
        return rf"$10^{{{int(round(k))}}}$"
    return f"{av:g}"


_SIGNED_YLABEL = r"Q-error  (underestimation $\leftrightarrow$ overestimation)"

_SMALL_FONT_RC: dict[str, float] = {
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 6,
}


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _styled_violin(ax, data, positions, color, width):
    parts = ax.violinplot(
        data, positions=positions, widths=width,
        showmedians=True, showextrema=True,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(0.6)
    return parts


def _legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    kwargs.pop("loc", None)
    kwargs.pop("bbox_to_anchor", None)
    kwargs.pop("ncol", None)
    ax.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(handles),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.0,
        **kwargs,
    )


def boxplot_by_cardinality_bin(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    bins = CARDINALITY_BIN_LABELS
    n_methods = len(methods)
    if n_methods == 0:
        return

    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        data = [
            method_df.loc[method_df["cardinality_bin"] == b, "q_error"].to_numpy()
            for b in bins
        ]
        positions = [j + (i - (n_methods - 1) / 2) * width for j in range(len(bins))]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.4},
        )
        color = _color(method)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot([], [], color=color, label=_display(method), linewidth=4)

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("True cardinality")
    ax.set_ylabel("Q-error")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    _legend(ax)
    _save(fig, out_path)


def boxplots_overall(df: pd.DataFrame, out_dir: Path) -> None:
    boxplot_by_cardinality_bin(df, out_dir / "boxplot_overall.pdf")


def boxplots_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    for graph, g_df in df.groupby("graph", observed=True):
        boxplot_by_cardinality_bin(g_df, out_dir / f"{_safe_filename(str(graph))}.pdf")


def boxplots_per_query_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    for (graph, qd), g_df in df.groupby(["graph", "query_dataset"], observed=True):
        boxplot_by_cardinality_bin(
            g_df,
            out_dir / f"{_safe_filename(str(graph))}__{_safe_filename(str(qd))}.pdf",
        )


def cdf_q_error(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    for method in methods:
        q = np.sort(df.loc[df["method"] == method, "q_error"].to_numpy())
        if q.size == 0:
            continue
        y = np.arange(1, q.size + 1) / q.size
        ax.plot(
            q, y,
            label=_display(method),
            color=_color(method),
            linestyle=_linestyle(method),
            linewidth=1.2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Q-error")
    ax.set_ylabel("CDF")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    _legend(ax, loc="lower right")
    _save(fig, out_path)


def cdf_overall(df: pd.DataFrame, out_dir: Path) -> None:
    cdf_q_error(df, out_dir / "cdf_overall.pdf")


def cdf_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    for graph, g_df in df.groupby("graph", observed=True):
        cdf_q_error(g_df, out_dir / f"{_safe_filename(str(graph))}.pdf")


def _scatter_ax(ax, df: pd.DataFrame, show_legend: bool = True) -> None:
    methods = _ordered_methods(df)
    eps = 1e-7
    x_all = np.maximum(df["true_value"].to_numpy(), eps)
    y_all = np.maximum(df["prediction"].to_numpy(), eps)
    lo = min(x_all.min(), y_all.min())
    hi = max(x_all.max(), y_all.max())
    for method in methods:
        sub = df[df["method"] == method]
        x = np.maximum(sub["true_value"].to_numpy(), eps)
        y = np.maximum(sub["prediction"].to_numpy(), eps)
        ax.scatter(
            x, y, s=3, alpha=0.25,
            label=_display(method), color=_color(method), edgecolors="none",
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True cardinality")
    ax.set_ylabel("Predicted cardinality")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(handles),
            frameon=False,
            markerscale=3,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)


def scatter_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for graph, g_df in df.groupby("graph", observed=True):
        fig, ax = plt.subplots(figsize=(COL_W, COL_W))
        _scatter_ax(ax, g_df)
        _save(fig, out_dir / f"{_safe_filename(str(graph))}.pdf")


def scatter_overall_panel(df: pd.DataFrame, out_path: Path) -> None:
    graphs = sorted(df["graph"].dropna().unique())
    n = len(graphs)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    cell = COL_W / cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(COL_W, cell * rows + 0.4),
        squeeze=False, sharex=True, sharey=True,
    )
    for ax, graph in zip(axes.flat, graphs):
        sub = df[df["graph"] == graph]
        _scatter_ax(ax, sub, show_legend=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(
            0.04, 0.96, _display_graph(str(graph)), transform=ax.transAxes,
            va="top", ha="left", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
        )
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.supxlabel("True cardinality", fontsize=8)
    fig.supylabel("Predicted cardinality", fontsize=8)

    methods = _ordered_methods(df)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=_color(m), label=_display(m))
        for m in methods
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(methods),
        frameon=False, bbox_to_anchor=(0.5, -0.02),
        handlelength=1.0, handletextpad=0.4, columnspacing=1.2,
    )
    _save(fig, out_path)


def bias_histogram(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    if not methods:
        return
    eps = 1e-7
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    bins = np.linspace(-6, 6, 61)
    for method in methods:
        sub = df[df["method"] == method]
        p = np.maximum(sub["prediction"].to_numpy(), eps)
        t = np.maximum(sub["true_value"].to_numpy(), eps)
        r = np.log10(p / t)
        ax.hist(
            r, bins=bins,
            histtype="step", linewidth=1.2,
            label=_display(method),
            color=_color(method),
            linestyle=_linestyle(method),
            density=True,
        )
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$\log_{10}(\mathrm{predicted} / \mathrm{true})$")
    ax.set_ylabel("Density")
    _legend(ax)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    _save(fig, out_path)


def boxplot_by_query_type(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    query_types = sorted(df["query_type"].dropna().unique())
    n_methods = len(methods)
    if n_methods == 0 or not query_types:
        return
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        data = [
            method_df.loc[method_df["query_type"] == qt, "q_error"].to_numpy()
            for qt in query_types
        ]
        positions = [j + (i - (n_methods - 1) / 2) * width for j in range(len(query_types))]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.4},
        )
        color = _color(method)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot([], [], color=color, label=_display(method), linewidth=4)
    ax.set_xticks(range(len(query_types)))
    ax.set_xticklabels(query_types, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("Query type")
    ax.set_ylabel("Q-error")
    _legend(ax)
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    _save(fig, out_path)


def _grouped_signed_boxplot(
    df: pd.DataFrame,
    group_col: str,
    group_values: list,
    xlabel: str,
    out_path: Path,
    rotate_xticks: bool = False,
) -> None:
    methods = _ordered_methods(df)
    n_methods = len(methods)
    if n_methods == 0 or not group_values:
        return
    width = 0.8 / n_methods
    with plt.rc_context(_SMALL_FONT_RC):
        fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN * 2 / 3))
        for i, method in enumerate(methods):
            method_df = df[df["method"] == method]
            data = [
                _signed_q_error(method_df[method_df[group_col] == g])
                if (method_df[group_col] == g).any()
                else np.array([np.nan])
                for g in group_values
            ]
            positions = [j + (i - (n_methods - 1) / 2) * width for j in range(len(group_values))]
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"linewidth": 0.6},
                capprops={"linewidth": 0.6},
                boxprops={"linewidth": 0.4},
            )
            color = _color(method)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.plot([], [], color=color, label=_display(method), linewidth=4)
        ax.set_yscale("symlog", linthresh=1, linscale=0.01)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(_symlog_signed_tick))
        ax.set_xticks(range(len(group_values)))
        ax.set_xticklabels(
            [_display_graph(str(g)) if group_col == "graph" else str(g) for g in group_values],
            rotation=30 if rotate_xticks else 0,
            ha="right" if rotate_xticks else "center",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_SIGNED_YLABEL, fontsize=4.5)
        _legend(ax)
        ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
        _save(fig, out_path)


def signed_boxplot_overall(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    if not methods:
        return
    with plt.rc_context(_SMALL_FONT_RC):
        fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN * 2 / 3))
        data = [_signed_q_error(df[df["method"] == m]) for m in methods]
        bp = ax.boxplot(
            data,
            positions=list(range(len(methods))),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.4},
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_color(methods[i]))
            patch.set_alpha(0.7)
        ax.set_yscale("symlog", linthresh=1, linscale=0.01)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(_symlog_signed_tick))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_display(m) for m in methods])
        ax.set_ylabel(_SIGNED_YLABEL, fontsize=4.5)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
        _save(fig, out_path)


def signed_boxplot_per_graph(df: pd.DataFrame, out_path: Path) -> None:
    graphs = sorted(df["graph"].dropna().unique())
    _grouped_signed_boxplot(df, "graph", graphs, "Graph", out_path, rotate_xticks=True)


QUERY_TYPE_ORDER: tuple[str, ...] = (
    "star", "path", "path_star_ends", "tree", "cycle", "diamond", "flower", "snowflake",
)


def signed_boxplot_per_query_type(df: pd.DataFrame, out_path: Path) -> None:
    present = set(df["query_type"].dropna().unique())
    qts = [q for q in QUERY_TYPE_ORDER if q in present]
    qts += sorted(present - set(QUERY_TYPE_ORDER))
    _grouped_signed_boxplot(df, "query_type", qts, "Query type", out_path, rotate_xticks=True)


def signed_boxplot_per_cardinality_bin(df: pd.DataFrame, out_path: Path) -> None:
    _grouped_signed_boxplot(
        df, "cardinality_bin", list(CARDINALITY_BIN_LABELS),
        "True cardinality", out_path, rotate_xticks=True,
    )


def headline_bar_median_q(per_method_graph: pd.DataFrame, out_path: Path) -> None:
    graphs = sorted(per_method_graph["graph"].unique())
    methods = _ordered_methods(per_method_graph)
    n_methods = len(methods)
    if not graphs or n_methods == 0:
        return
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    x = np.arange(len(graphs))
    for i, method in enumerate(methods):
        sub = per_method_graph[per_method_graph["method"] == method].set_index("graph")
        vals = [sub.loc[g, "median_q"] if g in sub.index else np.nan for g in graphs]
        ax.bar(
            x + (i - (n_methods - 1) / 2) * width,
            vals,
            width=width * 0.9,
            label=_display(method),
            color=_color(method),
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_graph(g) for g in graphs], rotation=30, ha="right")
    ax.set_ylabel("Median Q-error")
    _legend(ax)
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    _save(fig, out_path)


def ccdf_q_error(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    if not methods:
        return
    with plt.rc_context(_SMALL_FONT_RC):
        fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN * 0.5))
        for method in methods:
            q = np.sort(df.loc[df["method"] == method, "q_error"].to_numpy())
            if q.size == 0:
                continue
            y = np.arange(q.size, 0, -1) / q.size
            ax.plot(
                q, y,
                label=_display(method),
                color=_color(method),
                linestyle=_linestyle(method),
                linewidth=1.2,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Q-error $x$")
        ax.set_ylabel(r"$P \ (\mathrm{Q\text{-}error} > x)$")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
        _legend(ax, loc="lower left")
        _save(fig, out_path)


def ccdf_overall(df: pd.DataFrame, out_dir: Path) -> None:
    ccdf_q_error(df, out_dir / "ccdf_overall.pdf")


def ccdf_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    for graph, g_df in df.groupby("graph", observed=True):
        ccdf_q_error(g_df, out_dir / f"{_safe_filename(str(graph))}.pdf")


def _calibration_bins(
    p: np.ndarray, t: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p)
    p_sorted, t_sorted = p[order], t[order]
    edges = np.linspace(0, p.size, n_bins + 1).astype(int)
    means_p = np.array([p_sorted[edges[k]:edges[k + 1]].mean() for k in range(n_bins)])
    means_t = np.array([t_sorted[edges[k]:edges[k + 1]].mean() for k in range(n_bins)])
    return means_p, means_t


def calibration(df: pd.DataFrame, out_path: Path, n_bins: int = 10) -> None:
    methods = _ordered_methods(df)
    if not methods:
        return
    eps = 1e-7
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    t_all = np.maximum(df["true_value"].to_numpy(), eps)
    p_all = np.maximum(df["prediction"].to_numpy(), eps)
    lo = min(t_all.min(), p_all.min())
    hi = max(t_all.max(), p_all.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    for method in methods:
        sub = df[df["method"] == method]
        p = np.maximum(sub["prediction"].to_numpy(), eps)
        t = np.maximum(sub["true_value"].to_numpy(), eps)
        if p.size < n_bins:
            continue
        means_p, means_t = _calibration_bins(p, t, n_bins)
        ax.plot(
            means_p, means_t,
            marker="o", markersize=3,
            label=_display(method),
            color=_color(method),
            linestyle=_linestyle(method),
            linewidth=1.0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted cardinality")
    ax.set_ylabel("Mean true cardinality")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    _legend(ax, loc="upper left")
    _save(fig, out_path)


def calibration_overall(df: pd.DataFrame, out_dir: Path) -> None:
    calibration(df, out_dir / "calibration_overall.pdf")


def calibration_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    for graph, g_df in df.groupby("graph", observed=True):
        calibration(g_df, out_dir / f"{_safe_filename(str(graph))}.pdf")


def relative_improvement_bar(
    per_method_graph: pd.DataFrame, out_path: Path, target: str = "fice"
) -> None:
    methods = set(per_method_graph["method"].unique())
    if target not in methods:
        return
    others = [m for m in _ordered_methods(per_method_graph) if m != target]
    if not others:
        return
    piv = per_method_graph.pivot(index="graph", columns="method", values="median_q")
    best_other = piv[others].min(axis=1)
    best_other_name = piv[others].idxmin(axis=1)
    valid = piv[target].notna() & best_other.notna()
    graphs = [g for g in sorted(piv.index) if valid.loc[g]]
    if not graphs:
        return
    ratios = np.array([piv.loc[g, target] / best_other.loc[g] for g in graphs])
    baseline_labels = [_display(str(best_other_name.loc[g])) for g in graphs]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    x = np.arange(len(graphs))
    colors = ["#2a9d8f" if r < 1 else "#e76f51" for r in ratios]
    bars = ax.bar(x, ratios, color=colors, alpha=0.85, linewidth=0)
    for bar, label in zip(bars, baseline_labels):
        ax.annotate(
            f"vs {label}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_graph(g) for g in graphs], rotation=30, ha="right")
    ax.set_ylabel(f"Median Q-error ratio ({_display(target)} / best baseline)")
    ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.4)
    _save(fig, out_path)


def _grouped_signed_violin(
    df: pd.DataFrame,
    group_col: str,
    group_values: list,
    xlabel: str,
    out_path: Path,
    rotate_xticks: bool = False,
) -> None:
    methods = _ordered_methods(df)
    n_methods = len(methods)
    if n_methods == 0 or not group_values:
        return
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        good_positions = []
        good_data = []
        for j, g in enumerate(group_values):
            sub = method_df[method_df[group_col] == g]
            if len(sub) < 2:
                continue
            good_positions.append(j + (i - (n_methods - 1) / 2) * width)
            good_data.append(_signed_log_q(sub))
        if good_data:
            _styled_violin(ax, good_data, good_positions, _color(method), width * 0.9)
        ax.plot([], [], color=_color(method), label=_display(method), linewidth=4)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xticks(range(len(group_values)))
    ax.set_xticklabels(
        [_display_graph(str(g)) if group_col == "graph" else str(g) for g in group_values],
        rotation=30 if rotate_xticks else 0,
        ha="right" if rotate_xticks else "center",
    )
    ax.yaxis.set_major_formatter(FuncFormatter(_signed_log_tick))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_SIGNED_YLABEL)
    _legend(ax)
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    _save(fig, out_path)


def signed_violin_overall(df: pd.DataFrame, out_path: Path) -> None:
    methods = _ordered_methods(df)
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        if len(sub) < 2:
            continue
        _styled_violin(ax, [_signed_log_q(sub)], [i], _color(method), 0.6)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_display(m) for m in methods])
    ax.yaxis.set_major_formatter(FuncFormatter(_signed_log_tick))
    ax.set_ylabel(_SIGNED_YLABEL)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    _save(fig, out_path)


def signed_violin_per_graph(df: pd.DataFrame, out_path: Path) -> None:
    graphs = sorted(df["graph"].dropna().unique())
    _grouped_signed_violin(df, "graph", graphs, "Graph", out_path, rotate_xticks=True)


def signed_violin_per_query_type(df: pd.DataFrame, out_path: Path) -> None:
    qts = sorted(df["query_type"].dropna().unique())
    _grouped_signed_violin(df, "query_type", qts, "Query type", out_path, rotate_xticks=True)


def signed_violin_per_cardinality_bin(df: pd.DataFrame, out_path: Path) -> None:
    _grouped_signed_violin(
        df, "cardinality_bin", list(CARDINALITY_BIN_LABELS),
        "True cardinality", out_path, rotate_xticks=True,
    )


# ---------------------------------------------------------------------------
# Per-query runtime plots (µs) grouped by number of triple patterns.
# Input schema: columns {method, n_triple_pattern, runtime}.
# ---------------------------------------------------------------------------


def boxplot_runtime_by_triples(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped dodge boxplot: runtime (µs, log y) per method, per triple count."""
    methods = _ordered_methods(df)
    sizes = sorted(int(s) for s in df["n_triple_pattern"].dropna().unique())
    n_methods = len(methods)
    if n_methods == 0 or not sizes:
        return

    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN * 2 / 3))

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        # Drop sizes with no data for this method — matplotlib boxplot chokes
        # on empty arrays (np.percentile([], q) raises IndexError).
        kept = [
            (j, method_df.loc[method_df["n_triple_pattern"] == s, "runtime"].to_numpy())
            for j, s in enumerate(sizes)
            if (method_df["n_triple_pattern"] == s).any()
        ]
        color = _color(method)
        if kept:
            data = [a for _, a in kept]
            positions = [j + (i - (n_methods - 1) / 2) * width for j, _ in kept]
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"linewidth": 0.6},
                capprops={"linewidth": 0.6},
                boxprops={"linewidth": 0.4},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.plot([], [], color=color, label=_display(method), linewidth=4)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yscale("log")
    ax.set_xlabel("# triple patterns")
    ax.set_ylabel("Runtime per query (µs)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    _legend(ax)
    _save(fig, out_path)


def lineplot_runtime_by_triples(df: pd.DataFrame, out_path: Path) -> None:
    """Median with IQR error bars per method vs triple count (log y).

    Methods are slightly x-dodged to keep overlapping error bars legible.
    """
    methods = _ordered_methods(df)
    sizes = sorted(int(s) for s in df["n_triple_pattern"].dropna().unique())
    n_methods = len(methods)
    if not methods or not sizes:
        return

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN * 2 / 3))

    dodge = 0.12
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        med = np.empty(len(sizes), dtype=float)
        q25 = np.empty(len(sizes), dtype=float)
        q75 = np.empty(len(sizes), dtype=float)
        for j, s in enumerate(sizes):
            arr = method_df.loc[method_df["n_triple_pattern"] == s, "runtime"].to_numpy()
            if arr.size == 0:
                med[j] = q25[j] = q75[j] = np.nan
            else:
                med[j] = np.median(arr)
                q25[j] = np.percentile(arr, 25)
                q75[j] = np.percentile(arr, 75)
        # Asymmetric error bar extents from median to quartiles.
        err_low = med - q25
        err_high = q75 - med
        x = np.array(sizes, dtype=float) + (i - (n_methods - 1) / 2) * dodge
        color = _color(method)
        ax.errorbar(
            x, med,
            yerr=[err_low, err_high],
            color=color,
            linestyle=_linestyle(method),
            linewidth=1.0,
            marker=".",
            markersize=4,
            capsize=2.5,
            capthick=0.6,
            elinewidth=0.6,
            label=_display(method),
        )

    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel("# triple patterns")
    ax.set_ylabel("Runtime per query (µs)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    _legend(ax)
    _save(fig, out_path)
