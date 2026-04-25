"""Plot p-error distributions from leftdeep_perror per_query.csv.

Produces two figures per run directory:
  1. boxplot_perror.pdf / .png — grouped boxplots of log10(p-error), x=shape,
     one box per estimator (FICE, geom, QLever).
  2. hist_perror.pdf / .png    — overlaid histogram of log10(p-error) for
     FICE vs QLever, pooled across shapes.

Usage (from repo root):
    .venv/bin/python inference/plot_perror.py \\
        --dirs output/evaluation/leftdeep_perror output/evaluation/leftdeep_perror_ranking_v2 \\
        --labels baseline rank_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ESTIMATORS = ("fice", "geom", "qlever")
EST_LABELS = {"fice": "FICE", "geom": "geom(FICE,QLever)", "qlever": "QLever"}
EST_COLORS = {"fice": "#2b7bba", "geom": "#6aa84f", "qlever": "#cc4125"}
SHAPES = ("star", "path", "cycle", "diamond", "flower", "binary_tree", "path_star_ends")


def _plog(x):
    """Return log10(p-error), clipped to avoid -inf for ties at 1."""
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, 1.0, None))


def boxplot_run(per_query: pd.DataFrame, title: str, out_path: Path) -> None:
    shapes = [s for s in SHAPES if s in per_query["shape"].unique()]
    n_est = len(ESTIMATORS)
    width = 0.8 / n_est
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(shapes)), 4.5))

    for j, est in enumerate(ESTIMATORS):
        col = f"p_error_{est}"
        data = [_plog(per_query.loc[per_query["shape"] == s, col].values) for s in shapes]
        positions = np.arange(len(shapes)) + (j - (n_est - 1) / 2) * width
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.35),
            medianprops=dict(color="black", linewidth=1.2),
        )
        color = EST_COLORS[est]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for whisker in bp["whiskers"]:
            whisker.set_color(color)
        for cap in bp["caps"]:
            cap.set_color(color)

    # Guide lines for 1, 10, 100
    for y in (0, 1, 2):
        ax.axhline(y, color="gray", linewidth=0.3, linestyle="--", alpha=0.5)

    ax.set_xticks(np.arange(len(shapes)))
    ax.set_xticklabels(shapes, rotation=15, ha="right")
    ax.set_ylabel(r"$\log_{10}$(p-error)")
    ax.set_title(title)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=EST_COLORS[e], alpha=0.55, label=EST_LABELS[e])
        for e in ESTIMATORS
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def hist_run(per_query: pd.DataFrame, title: str, out_path: Path) -> None:
    """Two-panel: pooled log10(p-error) histogram for FICE vs QLever."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    fice = _plog(per_query["p_error_fice"].values)
    qlever = _plog(per_query["p_error_qlever"].values)

    lo = 0.0
    hi = max(float(np.nanmax(fice)), float(np.nanmax(qlever))) + 0.1
    bins = np.linspace(lo, hi, 41)

    ax.hist(fice, bins=bins, alpha=0.55, color=EST_COLORS["fice"],
            label=f"FICE (geomean={10**np.mean(fice):.2f}, median={10**np.median(fice):.2f})",
            edgecolor="black", linewidth=0.3)
    ax.hist(qlever, bins=bins, alpha=0.55, color=EST_COLORS["qlever"],
            label=f"QLever (geomean={10**np.mean(qlever):.2f}, median={10**np.median(qlever):.2f})",
            edgecolor="black", linewidth=0.3)

    ax.axvline(np.mean(fice), color=EST_COLORS["fice"], linestyle=":", linewidth=1.2)
    ax.axvline(np.mean(qlever), color=EST_COLORS["qlever"], linestyle=":", linewidth=1.2)

    ax.set_xlabel(r"$\log_{10}$(p-error)  (0 = optimal)")
    ax.set_ylabel("# queries")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def hist_compare_runs(runs: list[tuple[str, pd.DataFrame]], est: str,
                      title: str, out_path: Path) -> None:
    """Overlay one estimator's distribution across multiple runs (e.g. FICE baseline vs rank_v2)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    arrays = [(lbl, _plog(df[f"p_error_{est}"].values)) for lbl, df in runs]
    hi = max(float(np.nanmax(a)) for _, a in arrays) + 0.1
    bins = np.linspace(0, hi, 41)

    colors = ["#666666", EST_COLORS["fice"], "#9d47c2"]  # baseline gray, rank_v2 blue, other purple
    for (lbl, arr), c in zip(arrays, colors):
        ax.hist(arr, bins=bins, alpha=0.55, color=c,
                label=f"{lbl} (geomean={10**np.mean(arr):.2f}, median={10**np.median(arr):.2f})",
                edgecolor="black", linewidth=0.3)
        ax.axvline(np.mean(arr), color=c, linestyle=":", linewidth=1.2)

    ax.set_xlabel(r"$\log_{10}$(p-error)  (0 = optimal)")
    ax.set_ylabel("# queries")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="One or more leftdeep_perror output directories containing per_query.csv")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Friendly labels for each dir (defaults to dir basenames)")
    ap.add_argument("--out", type=Path, default=Path("output/evaluation/perror_plots"),
                    help="Output directory for combined cross-run plots")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in args.dirs]
    labels = args.labels or [d.name for d in dirs]
    if len(labels) != len(dirs):
        raise SystemExit("--labels must match --dirs length")

    per_run: list[tuple[str, pd.DataFrame]] = []
    for d, lbl in zip(dirs, labels):
        pq = d / "per_query.csv"
        if not pq.exists():
            raise SystemExit(f"missing {pq}")
        df = pd.read_csv(pq)
        per_run.append((lbl, df))

        # Per-run plots
        boxplot_run(df, f"p-error by shape — {lbl}", d / "boxplot_perror")
        hist_run(df, f"p-error distribution (pooled) — {lbl}", d / "hist_perror")
        print(f"wrote {d}/boxplot_perror.{{pdf,png}} and hist_perror.{{pdf,png}}")

    # Cross-run comparison: FICE across runs vs QLever (same)
    if len(per_run) >= 2:
        hist_compare_runs(per_run, est="fice",
                          title="FICE p-error — cross-run comparison",
                          out_path=args.out / "hist_fice_crossrun")
        hist_compare_runs(per_run, est="qlever",
                          title="QLever p-error — cross-run (should be ~identical)",
                          out_path=args.out / "hist_qlever_crossrun")
        print(f"wrote {args.out}/hist_{{fice,qlever}}_crossrun.{{pdf,png}}")


if __name__ == "__main__":
    main()
