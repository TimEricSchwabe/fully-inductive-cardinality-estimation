#!/usr/bin/env python3
"""Correlate per-KG median q-error with structural divergence + graph size.

Three analyses are produced, for every method (FICE, GNCE, LMKG, WJ, CSet):

  1. Marginal correlation of median q-error with mean JSD (the structural-
     divergence story from structural_divergence.py).
  2. Marginal correlation of median q-error with log graph size |G|
     (the scale-OOD story: does FICE get worse just because graphs are big?).
  3. Partial correlation of log q-error with mean JSD, controlling for log|G|.
     This asks: after removing the linear effect of graph size, does divergence
     still carry information about error?

Additionally runs a leave-one-out robustness check on FICE's headline
correlation -- with n=10, a single KG can move Pearson by 0.1-0.2, so we
report the range of FICE's divergence-correlation across the 10 leave-outs.

Artifacts:

    correlation_table.csv       methods x divergence-metric: marginal rho + p
    size_correlation.csv        per-method: rho(qE, |G|), rho(qE, JSD),
                                and partial rho(qE, JSD | |G|), with p-values
    leave_one_out_fice.csv      FICE's (JSD, qE) Pearson across 10 leave-outs
    scatter_per_method.pdf      divergence scatter, 5 panels
    scatter_overlay.pdf         divergence scatter, all methods overlaid
    scatter_size.pdf            log|G| vs qE, all methods overlaid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as tdist

# Per-KG median q-error, paper Table 2.
QERROR = pd.DataFrame(
    [
        ("aids",        1.70,  4.00,   78.64,    7.00,  26.67,  758.76,    3.00),
        ("codex_l",     8.19,  10.31,  18.66,   23.00,  28.08,  359.00,   13.12),
        ("dbpedia100k", 3.22,  7.56,   6.79,    11.00,  41.39,  255.00,  123.20),
        ("fb237",       5.27,  16.44,  7.75,    15.00,  63.46,  211.96,    7.35),
        ("hetionet",    5.91,  61.86,  304.29, 129.78,  36.30,  346.56,  463.56),
        ("lubm",        2.07,  3.06,   8.41,    31.13,  9.92,   98.83,    2.00),
        ("swdf",        2.45,  4.28,   5.24,    23.00,  30.51,  334.51,   14.00),
        ("wikidata",    16.48, 15.76,  5.99,     8.00,  13.04,  309.00,    9.00),
        ("wn18rr",      3.25,  3.46,   7.15,    17.63,  39.59,  107.00,   16.00),
        ("yago",        4.85,  8.69,   80.19,   14.71,  2.92,   197.56,   21.42),
    ],
    columns=["kg", "FICE", "GNCE", "LMKG", "LSS", "WJ", "CSet", "SumRDF"],
).set_index("kg")

# Per-KG number of triples (from structural_divergence.py parsing output).
TRIPLES = pd.Series({
    "swdf":         242_256,
    "aids":         802_066,
    "lubm":       2_688_849,
    "fb237":         33_916,
    "codex_l":      612_437,
    "dbpedia100k":  697_572,
    "wikidata":  21_354_359,
    "yago":      58_276_870,
    "hetionet":   2_250_197,
    "wn18rr":         9_842,
}, name="triples")

METHODS = ["FICE", "GNCE", "LMKG", "LSS", "WJ", "CSet", "SumRDF"]
DIVERGENCE_COLS = ["deg", "csize", "cset_mult", "pred_freq", "mean"]


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------

def partial_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial Pearson of (x, y) controlling for z, with t-based p-value.

    Uses the closed-form identity
        rho(x,y|z) = (rxy - rxz ryz) / sqrt((1-rxz^2)(1-ryz^2))
    and the standard t statistic  t = r sqrt((n-3)/(1-r^2))  with df = n - 3.
    """
    rxy, _ = pearsonr(x, y)
    rxz, _ = pearsonr(x, z)
    ryz, _ = pearsonr(y, z)
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0:
        return float("nan"), float("nan")
    r = (rxy - rxz * ryz) / denom
    n = len(x)
    df = n - 3
    if df <= 0 or abs(r) >= 1:
        return float(r), float("nan")
    tstat = r * np.sqrt(df / (1 - r**2))
    p = 2.0 * (1.0 - tdist.cdf(abs(tstat), df))
    return float(r), float(p)


def residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return x - OLS_fit(x ~ z) residuals."""
    slope, intercept = np.polyfit(z, x, 1)
    return x - (slope * z + intercept)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def marginal_divergence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        qe_full = df[method].to_numpy()
        mask = ~np.isnan(qe_full)
        qe = qe_full[mask]
        log_qe = np.log(qe)
        for col in DIVERGENCE_COLS:
            d = df[col].to_numpy()[mask]
            pr, pp = pearsonr(log_qe, d)
            sr, sp = spearmanr(qe, d)
            rows.append({
                "method": method, "divergence_metric": col, "n": int(mask.sum()),
                "pearson_logqE": float(pr), "pearson_p": float(pp),
                "spearman_qE": float(sr),  "spearman_p": float(sp),
            })
    return pd.DataFrame(rows)


def size_and_partial_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each method, report:
        rho(log qE, mean JSD)            -- marginal structural signal
        rho(log qE, log |G|)             -- marginal size signal
        rho(log qE, mean JSD | log |G|)  -- structural signal after removing size
        rho(log qE, log |G| | mean JSD)  -- size signal after removing structure

    Sanity: Recipe A (residualize-then-correlate) should match Recipe B
    (closed form) to numerical precision; both reported.
    """
    log_G_full = np.log(df["triples"].to_numpy())
    jsd_full = df["mean"].to_numpy()
    rows = []
    for method in METHODS:
        qe_full = df[method].to_numpy()
        mask = ~np.isnan(qe_full)
        log_qe = np.log(qe_full[mask])
        log_G = log_G_full[mask]
        jsd = jsd_full[mask]
        # Marginals
        r_jsd, p_jsd = pearsonr(log_qe, jsd)
        r_size, p_size = pearsonr(log_qe, log_G)
        # Partials (closed form)
        r_jsd_part, p_jsd_part = partial_pearson(jsd, log_qe, log_G)
        r_size_part, p_size_part = partial_pearson(log_G, log_qe, jsd)
        # Partials (residualize-then-correlate, sanity check)
        rx = residualize(jsd, log_G)
        ry = residualize(log_qe, log_G)
        r_jsd_part_resid, _ = pearsonr(rx, ry)
        rows.append({
            "method": method, "n": int(mask.sum()),
            "pearson_logqE_JSD": r_jsd,        "p_JSD": p_jsd,
            "pearson_logqE_logG": r_size,       "p_logG": p_size,
            "partial_JSD_given_logG": r_jsd_part,
            "partial_JSD_given_logG_p": p_jsd_part,
            "partial_logG_given_JSD": r_size_part,
            "partial_logG_given_JSD_p": p_size_part,
            "partial_JSD_given_logG_resid": r_jsd_part_resid,
        })
    return pd.DataFrame(rows).set_index("method")


def leave_one_out_fice(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute FICE's (log qE, mean JSD) Pearson leaving each KG out."""
    kgs = df.index.to_list()
    log_qe = np.log(df["FICE"].to_numpy())
    jsd = df["mean"].to_numpy()
    full_r, full_p = pearsonr(log_qe, jsd)
    rows = [{"left_out": "<none>", "n": len(kgs), "pearson_logqE_JSD": float(full_r),
             "p": float(full_p)}]
    for i, kg in enumerate(kgs):
        mask = np.arange(len(kgs)) != i
        r, p = pearsonr(log_qe[mask], jsd[mask])
        rows.append({"left_out": kg, "n": int(mask.sum()),
                     "pearson_logqE_JSD": float(r), "p": float(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def scatter_per_method(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.5 * len(METHODS), 4.5), sharex=True)
    x_full = df["mean"].to_numpy()
    xs = np.linspace(x_full.min(), x_full.max(), 50)
    for ax, method in zip(axes, METHODS):
        y_full = df[method].to_numpy()
        mask = ~np.isnan(y_full)
        x = x_full[mask]
        y = y_full[mask]
        log_y = np.log(y)
        pr, pp = pearsonr(log_y, x)
        sr, sp = spearmanr(y, x)
        ax.scatter(x, y, s=42, zorder=3)
        for kg, row in df[mask].iterrows():
            ax.annotate(kg, (row["mean"], row[method]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        a, b = np.polyfit(x, log_y, 1)
        ax.plot(xs, np.exp(a * xs + b), "--", color="grey", alpha=0.6, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("mean JSD to train mixture")
        ax.set_title(
            f"{method}\nPearson(log qE)={pr:.2f} (p={pp:.2f})   "
            f"Spearman(qE)={sr:.2f} (p={sp:.2f})", fontsize=9,
        )
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("median q-error")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def scatter_overlay(df: pd.DataFrame, xcol: str, xlabel: str, out_path: Path,
                    log_x: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x_full = df[xcol].to_numpy()
    x_plot_full = np.log(x_full) if log_x else x_full
    xs = np.linspace(x_plot_full.min(), x_plot_full.max(), 50)
    colors = plt.cm.tab10.colors[: len(METHODS)]
    for method, c in zip(METHODS, colors):
        y_full = df[method].to_numpy()
        mask = ~np.isnan(y_full)
        x_plot = x_plot_full[mask]
        y = y_full[mask]
        ax.scatter(x_plot, y, s=42, color=c, label=method, zorder=3)
        a, b = np.polyfit(x_plot, np.log(y), 1)
        ax.plot(xs, np.exp(a * xs + b), "--", color=c, alpha=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median q-error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def load_histograms(
    path: Path, kgs: list[str]
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Inverse of the savez in structural_divergence.py."""
    z = np.load(path)
    hists: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for metric in DIVERGENCE_COLS[:-1]:  # skip "mean"
        edges = z[f"edges__{metric}"]
        kg_hists = {kg: z[f"hist__{metric}__{kg}"] for kg in kgs}
        hists[metric] = (edges, kg_hists)
    return hists


def plot_raw_distributions(
    hists: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    kgs: list[str],
    out_path: Path,
) -> None:
    """Plot 1: 4 panels (one per metric), all KGs overlaid as line histograms."""
    metrics = [m for m in DIVERGENCE_COLS if m != "mean"]
    fig, axes = plt.subplots(1, 4, figsize=(4 * 3.0, 3.0))
    cmap = plt.cm.tab10.colors  # 10 distinct colors for 10 KGs
    for ax, metric in zip(axes, metrics):
        edges, kg_hists = hists[metric]
        centers = 0.5 * (edges[:-1] + edges[1:])
        for kg, color in zip(kgs, cmap):
            ax.plot(centers, kg_hists[kg], color=color, label=kg, linewidth=1.0)
        ax.set_title(metric)
        ax.set_xlabel("bin (log2 scale where applicable)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("density")
    axes[-1].legend(fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_train_mixtures(
    hists: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    kgs: list[str],
    out_path: Path,
) -> None:
    """Plot 2: 4 panels (one per metric), each showing the equal-weight mixture
    histogram across all KGs."""
    metrics = [m for m in DIVERGENCE_COLS if m != "mean"]
    fig, axes = plt.subplots(1, 4, figsize=(4 * 3.0, 3.0))
    for ax, metric in zip(axes, metrics):
        edges, kg_hists = hists[metric]
        centers = 0.5 * (edges[:-1] + edges[1:])
        mix = np.mean(np.stack([kg_hists[k] for k in kgs]), axis=0)
        mix /= mix.sum()
        ax.fill_between(centers, mix, alpha=0.4, color="#444")
        ax.plot(centers, mix, color="black", linewidth=1.0)
        ax.set_title(f"{metric}: equal-weight mixture (all {len(kgs)} KGs)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_loo_pairs(
    hists: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    kgs: list[str],
    out_dir: Path,
) -> None:
    """Plot 3: per-metric file. Top row: global train-mix. Below: 10 panels,
    one per held-out KG, each overlaying the leave-one-out train-mix on the
    held-out KG's histogram."""
    metrics = [m for m in DIVERGENCE_COLS if m != "mean"]
    n_kgs = len(kgs)
    n_cols = 5
    n_rows_below = (n_kgs + n_cols - 1) // n_cols
    for metric in metrics:
        edges, kg_hists = hists[metric]
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig = plt.figure(figsize=(n_cols * 2.4, 2.4 * (1 + n_rows_below)))
        gs = fig.add_gridspec(1 + n_rows_below, n_cols, hspace=0.5, wspace=0.3)
        # Top row: global train mixture (mean of all 10).
        ax_top = fig.add_subplot(gs[0, :])
        mix_all = np.mean(np.stack([kg_hists[k] for k in kgs]), axis=0)
        mix_all /= mix_all.sum()
        ax_top.fill_between(centers, mix_all, alpha=0.4, color="#444")
        ax_top.plot(centers, mix_all, color="black", linewidth=1.0)
        ax_top.set_title(f"{metric}: train mixture (mean of all {n_kgs} KGs)")
        ax_top.grid(True, alpha=0.25)
        # Below: one panel per held-out KG.
        for idx, kg in enumerate(kgs):
            r, c = 1 + idx // n_cols, idx % n_cols
            ax = fig.add_subplot(gs[r, c])
            mix_loo = np.mean(
                np.stack([kg_hists[k] for k in kgs if k != kg]), axis=0
            )
            mix_loo /= mix_loo.sum()
            ax.plot(centers, mix_loo, color="#888", linewidth=1.0,
                    linestyle="--", label="LOO train-mix")
            ax.plot(centers, kg_hists[kg], color="C0", linewidth=1.2, label=kg)
            ax.set_title(kg, fontsize=8)
            ax.grid(True, alpha=0.25)
            if idx == 0:
                ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"loo_pairs_{metric}.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divergence-csv",
                    default="output/structural_divergence/divergence_table.csv")
    ap.add_argument("--out", default="output/structural_divergence")
    args = ap.parse_args()

    div = pd.read_csv(args.divergence_csv, index_col="kg")
    df = div.join(QERROR, how="inner").join(TRIPLES, how="inner")
    missing = set(QERROR.index) - set(df.index)
    if missing:
        print(f"[warn] KGs in q-error but not in divergence CSV: {missing}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Marginal divergence correlations (as before).
    corr = marginal_divergence_table(df)
    corr.to_csv(out_dir / "correlation_table.csv", index=False)
    print("\nMarginal correlation (divergence_metric = mean):")
    print(corr[corr["divergence_metric"] == "mean"]
          .set_index("method")[["pearson_logqE", "pearson_p",
                                "spearman_qE", "spearman_p"]]
          .round(3).to_string())

    # 2. Size vs. divergence, with partial correlation controlling for size.
    size = size_and_partial_table(df)
    size.to_csv(out_dir / "size_correlation.csv")
    print("\nSize vs divergence (log qE):")
    print(size[["pearson_logqE_JSD", "pearson_logqE_logG",
                "partial_JSD_given_logG", "partial_JSD_given_logG_p",
                "partial_logG_given_JSD", "partial_logG_given_JSD_p"]]
          .round(3).to_string())

    # 3. Leave-one-out robustness for FICE.
    loo = leave_one_out_fice(df)
    loo.to_csv(out_dir / "leave_one_out_fice.csv", index=False)
    print("\nFICE Pearson(log qE, mean JSD) -- leave-one-out:")
    print(loo.round(3).to_string(index=False))

    # 4. Plots.
    scatter_per_method(df, out_dir / "scatter_per_method.pdf")
    scatter_overlay(df, "mean", "mean JSD to train mixture",
                    out_dir / "scatter_overlay.pdf")
    scatter_overlay(df, "triples", "log |G| (triples)",
                    out_dir / "scatter_size.pdf", log_x=True)

    # 5. Distribution plots (require histograms.npz from structural_divergence.py).
    hist_path = Path(args.divergence_csv).with_name("histograms.npz")
    if hist_path.exists():
        kgs = list(df.index)
        hists = load_histograms(hist_path, kgs)
        plot_raw_distributions(hists, kgs, out_dir / "raw_distributions.pdf")
        plot_train_mixtures(hists, kgs, out_dir / "train_mixtures.pdf")
        plot_loo_pairs(hists, kgs, out_dir)
        print("Wrote raw_distributions.pdf, train_mixtures.pdf, loo_pairs_*.pdf")
    else:
        print(f"[warn] {hist_path} not found; "
              "re-run structural_divergence.py to generate it.")

    print(f"\nAll artifacts in {out_dir}")


if __name__ == "__main__":
    main()
