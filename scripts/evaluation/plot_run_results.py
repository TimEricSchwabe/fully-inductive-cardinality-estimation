"""Plot training curves and test plots for one or more training runs.

For every run directory (as produced by scripts/train.py) a `plots/` folder is
created containing:
  - curves_loss.png        train vs validation loss per epoch
  - curves_qerror.png      train vs validation mean/median q-error per epoch
  - test_scatter.png       predicted vs true cardinality (final test, log-log)
  - test_qerror_dist.png   q-error distribution of the final test predictions

If multiple run directories are given, an additional comparison figure and a
summary table are written next to them (comparison_curves.png, summary.csv).

Usage:
  python scripts/evaluation/plot_run_results.py <run_dir> [<run_dir> ...] [--out <dir>]
  # <run_dir> may also be the parent output dir containing a single run_*/ folder
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_run_dir(path: Path) -> Path:
    """Accept either a run_* directory or a parent containing run_* dirs (picks newest)."""
    if (path / "logs").exists():
        return path
    runs = sorted(path.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directory found under {path}")
    return runs[-1]


def load_run(run_dir: Path):
    logs = run_dir / "logs"
    train = pd.read_csv(logs / "train_training_metrics.csv")
    val = pd.read_csv(logs / "test_training_metrics.csv")
    preds = pd.read_csv(logs / "predictions.csv")
    # final test predictions; fall back to last validation epoch if absent
    test = preds[preds["stage"] == "final_test"]
    if test.empty:
        last = preds[preds["stage"] == "test"]["epoch"].max()
        test = preds[(preds["stage"] == "test") & (preds["epoch"] == last)]
    return train, val, test


def qerror(pred, true):
    pred = np.clip(pred, 1e-9, None)
    true = np.clip(true, 1e-9, None)
    return np.maximum(pred / true, true / pred)


def plot_run(run_dir: Path, label: str):
    train, val, test = load_run(run_dir)
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)

    # 1) loss curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train["epoch"], train["loss"], label="train")
    ax.plot(val["epoch"], val["loss"], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("smooth L1 loss (log-cardinality)")
    ax.set_title(f"{label}: loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "curves_loss.png", dpi=150); plt.close(fig)

    # 2) q-error curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, col, name in zip(axes, ["q_error", "median_q_error"], ["mean q-error", "median q-error"]):
        ax.plot(train["epoch"], train[col], label="train")
        ax.plot(val["epoch"], val[col], label="validation")
        ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel(name)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"{label}: q-error")
    fig.tight_layout(); fig.savefig(out / "curves_qerror.png", dpi=150); plt.close(fig)

    # 3) test scatter (pred vs true)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for qs, g in test.groupby("query_set_name"):
        ax.scatter(g["true_value"].clip(lower=1e-1), g["prediction"].clip(lower=1e-1),
                   s=8, alpha=0.4, label=qs)
    lims = [1e-1, max(test["true_value"].max(), test["prediction"].max()) * 2]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true cardinality"); ax.set_ylabel("predicted cardinality")
    ax.set_title(f"{label}: final test predictions")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "test_scatter.png", dpi=150); plt.close(fig)

    # 4) q-error distribution on final test
    qe = qerror(test["prediction"].values, test["true_value"].values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(qe), bins=50)
    for q, c in [(50, "tab:green"), (90, "tab:orange"), (99, "tab:red")]:
        v = np.percentile(qe, q)
        ax.axvline(np.log10(v), color=c, ls="--", label=f"p{q} = {v:.1f}")
    ax.set_xlabel("log10(q-error)"); ax.set_ylabel("#queries")
    ax.set_title(f"{label}: final test q-error distribution")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "test_qerror_dist.png", dpi=150); plt.close(fig)

    stats = {
        "run": label,
        "final_train_loss": float(train["loss"].iloc[-1]),
        "final_val_loss": float(val["loss"].iloc[-1]),
        "test_qerror_mean": float(qe.mean()),
        "test_qerror_p50": float(np.percentile(qe, 50)),
        "test_qerror_p90": float(np.percentile(qe, 90)),
        "test_qerror_p99": float(np.percentile(qe, 99)),
        "n_test_queries": int(len(qe)),
    }
    return train, val, qe, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for the cross-run comparison outputs")
    args = ap.parse_args()

    curves, all_stats = {}, []
    for rd in args.run_dirs:
        run_dir = resolve_run_dir(rd)
        label = rd.name if rd.name.startswith("run_") is False else run_dir.name
        train, val, qe, stats = plot_run(run_dir, label)
        curves[label] = (train, val, qe)
        all_stats.append(stats)
        print(f"plots written: {run_dir / 'plots'}")

    if len(curves) > 1:
        out = args.out or args.run_dirs[0].parent
        out.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for label, (train, val, qe) in curves.items():
            axes[0].plot(val["epoch"], val["loss"], label=label)
            axes[1].plot(val["epoch"], val["median_q_error"], label=label)
            xs = np.sort(qe)
            axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=label)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation loss"); axes[0].set_title("Validation loss")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("median q-error"); axes[1].set_yscale("log"); axes[1].set_title("Validation median q-error")
        axes[2].set_xscale("log"); axes[2].set_xlabel("q-error"); axes[2].set_ylabel("fraction of test queries"); axes[2].set_title("Final test q-error CDF")
        for ax in axes:
            ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "comparison_curves.png", dpi=150); plt.close(fig)

        df = pd.DataFrame(all_stats)
        df.to_csv(out / "summary.csv", index=False)
        print(f"comparison written: {out / 'comparison_curves.png'}, {out / 'summary.csv'}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
