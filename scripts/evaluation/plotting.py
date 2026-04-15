"""Boxplots (grouped by cardinality bin) and q-error CDFs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema import CARDINALITY_BIN_LABELS


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def boxplot_by_cardinality_bin(df: pd.DataFrame, title: str, out_png: Path) -> None:
    methods = sorted(df["method"].unique())
    bins = CARDINALITY_BIN_LABELS
    n_methods = len(methods)
    if n_methods == 0:
        return

    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(bins)), 5))
    cmap = plt.get_cmap("tab10")

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
            medianprops={"color": "black"},
        )
        color = cmap(i % 10)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot([], [], color=color, label=method, linewidth=6)

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_yscale("log")
    ax.set_xlabel("true cardinality bin")
    ax.set_ylabel("q-error (log scale)")
    ax.set_title(title)
    ax.legend(title="method")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def boxplots_overall(df: pd.DataFrame, out_dir: Path) -> None:
    boxplot_by_cardinality_bin(df, "q-error by true cardinality (overall)", out_dir / "boxplot_overall.png")


def boxplots_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    for graph, g_df in df.groupby("graph", observed=True):
        boxplot_by_cardinality_bin(
            g_df,
            f"q-error by true cardinality — graph={graph}",
            out_dir / f"{_safe_filename(str(graph))}.png",
        )


def boxplots_per_query_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    for (graph, qd), g_df in df.groupby(["graph", "query_dataset"], observed=True):
        boxplot_by_cardinality_bin(
            g_df,
            f"q-error — {graph} / {qd}",
            out_dir / f"{_safe_filename(str(graph))}__{_safe_filename(str(qd))}.png",
        )


def cdf_q_error(df: pd.DataFrame, title: str, out_png: Path) -> None:
    methods = sorted(df["method"].unique())
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, method in enumerate(methods):
        q = np.sort(df.loc[df["method"] == method, "q_error"].to_numpy())
        if q.size == 0:
            continue
        y = np.arange(1, q.size + 1) / q.size
        ax.plot(q, y, label=method, color=cmap(i % 10))
    ax.set_xscale("log")
    ax.set_xlabel("q-error (log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def cdf_overall(df: pd.DataFrame, out_dir: Path) -> None:
    cdf_q_error(df, "q-error CDF (overall)", out_dir / "cdf_overall.png")


def cdf_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    for graph, g_df in df.groupby("graph", observed=True):
        cdf_q_error(g_df, f"q-error CDF — graph={graph}", out_dir / f"{_safe_filename(str(graph))}.png")


def _scatter_ax(ax, df: pd.DataFrame, title: str) -> None:
    methods = sorted(df["method"].unique())
    cmap = plt.get_cmap("tab10")
    eps = 1e-7
    x_all = np.maximum(df["true_value"].to_numpy(), eps)
    y_all = np.maximum(df["prediction"].to_numpy(), eps)
    lo = min(x_all.min(), y_all.min())
    hi = max(x_all.max(), y_all.max())
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        x = np.maximum(sub["true_value"].to_numpy(), eps)
        y = np.maximum(sub["prediction"].to_numpy(), eps)
        ax.scatter(x, y, s=4, alpha=0.25, label=method, color=cmap(i % 10), edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true cardinality")
    ax.set_ylabel("predicted cardinality")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    leg = ax.legend(markerscale=3, loc="upper left")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)


def scatter_per_graph(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for graph, g_df in df.groupby("graph", observed=True):
        fig, ax = plt.subplots(figsize=(6, 6))
        _scatter_ax(ax, g_df, f"pred vs true — graph={graph}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe_filename(str(graph))}.png", dpi=120)
        plt.close(fig)


def scatter_overall_panel(df: pd.DataFrame, out_png: Path) -> None:
    graphs = sorted(df["graph"].dropna().unique())
    n = len(graphs)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax, graph in zip(axes.flat, graphs):
        _scatter_ax(ax, df[df["graph"] == graph], str(graph))
    for ax in axes.flat[n:]:
        ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def bias_histogram(df: pd.DataFrame, out_png: Path) -> None:
    methods = sorted(df["method"].unique())
    if not methods:
        return
    eps = 1e-7
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    bins = np.linspace(-6, 6, 61)
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        p = np.maximum(sub["prediction"].to_numpy(), eps)
        t = np.maximum(sub["true_value"].to_numpy(), eps)
        r = np.log10(p / t)
        ax.hist(r, bins=bins, alpha=0.5, label=method, color=cmap(i % 10), density=True)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("log10(prediction / true)  — negative = under-estimate, positive = over-estimate")
    ax.set_ylabel("density")
    ax.set_title("estimation bias distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def boxplot_by_query_type(df: pd.DataFrame, out_png: Path) -> None:
    methods = sorted(df["method"].unique())
    query_types = sorted(df["query_type"].dropna().unique())
    n_methods = len(methods)
    if n_methods == 0 or not query_types:
        return
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(query_types)), 5))
    cmap = plt.get_cmap("tab10")
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
            medianprops={"color": "black"},
        )
        color = cmap(i % 10)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot([], [], color=color, label=method, linewidth=6)
    ax.set_xticks(range(len(query_types)))
    ax.set_xticklabels(query_types, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("query type")
    ax.set_ylabel("q-error (log scale)")
    ax.set_title("q-error by query type (all graphs)")
    ax.legend(title="method")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def headline_bar_median_q(per_method_graph: pd.DataFrame, out_png: Path) -> None:
    graphs = sorted(per_method_graph["graph"].unique())
    methods = sorted(per_method_graph["method"].unique())
    n_methods = len(methods)
    if not graphs or n_methods == 0:
        return
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(graphs)), 5))
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(graphs))
    for i, method in enumerate(methods):
        sub = per_method_graph[per_method_graph["method"] == method].set_index("graph")
        vals = [sub.loc[g, "median_q"] if g in sub.index else np.nan for g in graphs]
        ax.bar(
            x + (i - (n_methods - 1) / 2) * width,
            vals,
            width=width * 0.9,
            label=method,
            color=cmap(i % 10),
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=30, ha="right")
    ax.set_ylabel("median q-error (log)")
    ax.set_title("median q-error per graph")
    ax.legend(title="method")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
