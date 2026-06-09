#!/usr/bin/env python3
"""Structural divergence between each KG and the equal-weight mixture of the others.

Four vocabulary-blind fingerprints per KG — computed by streaming the raw .nt files:

    deg         entity total degree               (one sample per entity)
    csize       |char-set(e)| for e as subject    (one sample per subject-entity)
    cset_mult   #entities sharing each char-set   (one sample per distinct char-set)
    pred_freq   log10(1 + count(p)/|G|)           (one sample per predicate)

For every metric we build a histogram over bin edges computed from the union of
all KGs, smooth with epsilon, and compute JSD (base 2, squared scipy output so
values live in [0,1]) between each held-out KG and the equal-weight mixture of
the other nine KGs' histograms.

Outputs (under --out):
    divergence_table.csv       rows = KGs, cols = the four JSDs + mean
    pairwise_heatmaps.pdf      one 10x10 pairwise-JSD heatmap per metric
    divergence_vs_qerror.pdf   scatter of mean JSD vs per-KG median q-error
                               (only if --qerror-csv is provided)
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

EPS = 1e-6
N_BINS = 30

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Paper-display name -> directory under data/
KG_DIRS: dict[str, str] = {
    "swdf": "swdf",
    "aids": "aids",
    "lubm": "lubm",
    "fb237": "fb237_v4",
    "codex_l": "codex_l",
    "dbpedia100k": "dbpedia100k",
    "wikidata": "wikidata",
    "yago": "yago",
    "hetionet": "hetionet",
    "wn18rr": "wn18rr_v4",
}

# metric_name -> log-bin the values before histogramming?
METRICS: dict[str, bool] = {
    "deg": True,
    "csize": True,
    "cset_mult": True,
    "pred_freq": False,  # already log10-transformed at source
}


def find_raw_files(raw_dir: Path) -> list[Path]:
    nt_files = sorted(raw_dir.glob("*.nt"))
    if nt_files:
        return nt_files
    return sorted(f for f in raw_dir.iterdir() if f.is_file() and not f.suffix)


def kg_primitives(raw_files: list[Path]) -> tuple[dict[str, np.ndarray], int]:
    """One streaming pass over the .nt file(s); returns per-metric value arrays."""
    entity_id: dict[str, int] = {}
    pred_id: dict[str, int] = {}
    out_cs: dict[int, set[int]] = {}
    deg: Counter[int] = Counter()
    pred_freq: Counter[int] = Counter()
    n_triples = 0

    def intern(tbl: dict[str, int], key: str) -> int:
        i = tbl.get(key)
        if i is None:
            i = len(tbl)
            tbl[key] = i
        return i

    for path in raw_files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n\r")
                if not line or line.startswith("#"):
                    continue
                body = line.rsplit(" .", 1)[0]
                tokens = body.split(None, 2)
                if len(tokens) != 3:
                    continue
                s, p, o = tokens
                si = intern(entity_id, s)
                pi = intern(pred_id, p)
                oi = intern(entity_id, o)
                out_cs.setdefault(si, set()).add(pi)
                deg[si] += 1
                deg[oi] += 1
                pred_freq[pi] += 1
                n_triples += 1

    degrees = np.fromiter(deg.values(), dtype=np.int64)
    csizes = np.fromiter((len(cs) for cs in out_cs.values()), dtype=np.int64)
    cset_groups = Counter(frozenset(cs) for cs in out_cs.values())
    cset_mults = np.fromiter(cset_groups.values(), dtype=np.int64)
    denom = max(n_triples, 1)
    pred_freq_log = np.log10(
        1.0 + np.fromiter(pred_freq.values(), dtype=np.float64) / denom
    )

    return {
        "deg": degrees,
        "csize": csizes,
        "cset_mult": cset_mults,
        "pred_freq": pred_freq_log,
    }, n_triples


def union_edges(arrays: list[np.ndarray], log: bool) -> np.ndarray:
    allv = np.concatenate(arrays).astype(float)
    if log:
        allv = np.log2(1.0 + allv)
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, N_BINS + 1)


def histogram(values: np.ndarray, edges: np.ndarray, log: bool) -> np.ndarray:
    v = np.log2(1.0 + values.astype(float)) if log else values.astype(float)
    h, _ = np.histogram(v, bins=edges)
    h = h.astype(float) + EPS
    return h / h.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(jensenshannon(p, q, base=2) ** 2)


def equal_weight_mixture(hists: list[np.ndarray]) -> np.ndarray:
    m = np.mean(np.stack(hists, axis=0), axis=0)
    return m / m.sum()


def build_divergence_table(
    prims: dict[str, dict[str, np.ndarray]],
    kgs: list[str],
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]]:
    rows: list[dict] = []
    hists_by_metric: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for metric, log in METRICS.items():
        edges = union_edges([prims[k][metric] for k in kgs], log=log)
        kg_hists = {k: histogram(prims[k][metric], edges, log=log) for k in kgs}
        hists_by_metric[metric] = (edges, kg_hists)
        for holdout in kgs:
            mix = equal_weight_mixture([kg_hists[k] for k in kgs if k != holdout])
            rows.append({
                "kg": holdout,
                "metric": metric,
                "jsd": jsd(kg_hists[holdout], mix),
            })
    df = (
        pd.DataFrame(rows)
        .pivot(index="kg", columns="metric", values="jsd")[list(METRICS.keys())]
        .loc[kgs]
    )
    df["mean"] = df.mean(axis=1)
    return df, hists_by_metric


def plot_heatmaps(
    hists_by_metric: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    kgs: list[str],
    out_path: Path,
) -> None:
    metrics = list(METRICS.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for ax, metric in zip(axes, metrics):
        _, kg_hists = hists_by_metric[metric]
        n = len(kgs)
        M = np.zeros((n, n))
        for i, a in enumerate(kgs):
            for j, b in enumerate(kgs):
                if j > i:
                    M[i, j] = jsd(kg_hists[a], kg_hists[b])
        M = M + M.T  # symmetric; diagonal stays zero
        im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=max(M.max(), 1e-3))
        ax.set_title(f"JSD({metric})")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(kgs, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(kgs, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_divergence_vs_qerror(
    df: pd.DataFrame,
    qerror_csv: Path,
    out_path: Path,
) -> None:
    qe = pd.read_csv(qerror_csv)
    if not {"kg", "q_error"}.issubset(qe.columns):
        raise ValueError("qerror CSV must have columns 'kg' and 'q_error'")
    per_kg = qe.groupby("kg")["q_error"].median().rename("median_qerror")
    merged = df[["mean"]].join(per_kg, how="inner")
    if merged.empty:
        print("[warn] no KG overlap between divergence table and q-error CSV; skipping scatter")
        return
    x = merged["mean"].to_numpy()
    y = merged["median_qerror"].to_numpy()
    pearson = float(np.corrcoef(x, np.log(y))[0, 1])
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, s=40)
    for kg, row in merged.iterrows():
        ax.annotate(kg, (row["mean"], row["median_qerror"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean JSD to training mixture")
    ax.set_ylabel("median q-error")
    ax.set_yscale("log")
    ax.set_title(
        f"Pearson(log q-err, JSD) = {pearson:.2f}   Spearman = {spearman:.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/structural_divergence",
                    help="output directory for CSV + plots")
    ap.add_argument("--kgs", nargs="+", default=list(KG_DIRS.keys()),
                    help="paper names of KGs to include")
    ap.add_argument("--qerror-csv", default=None,
                    help="optional CSV with columns {kg, q_error} for the scatter plot")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prims: dict[str, dict[str, np.ndarray]] = {}
    sizes: dict[str, int] = {}
    for k in args.kgs:
        if k not in KG_DIRS:
            raise KeyError(f"unknown KG name '{k}'; known: {list(KG_DIRS)}")
        raw_dir = DATA_DIR / KG_DIRS[k] / "raw"
        raw_files = find_raw_files(raw_dir)
        if not raw_files:
            raise FileNotFoundError(f"no raw files under {raw_dir}")
        print(f"[{k:12s}] parsing {len(raw_files)} file(s) under {raw_dir}")
        prims[k], sizes[k] = kg_primitives(raw_files)
        print(
            f"[{k:12s}] triples={sizes[k]:>10,}  "
            f"entities≈{len(prims[k]['deg']):>9,}  "
            f"subject-entities={len(prims[k]['csize']):>9,}  "
            f"distinct-csets={len(prims[k]['cset_mult']):>7,}  "
            f"predicates={len(prims[k]['pred_freq']):>5,}"
        )

    df, hists_by_metric = build_divergence_table(prims, args.kgs)
    csv_path = out_dir / "divergence_table.csv"
    df.to_csv(csv_path)
    print("\nJSD (held-out KG vs equal-weight mixture of the other nine):")
    print(df.round(3).to_string())
    print(f"\nWrote {csv_path}")

    hist_path = out_dir / "histograms.npz"
    arrays: dict[str, np.ndarray] = {}
    for metric, (edges, kg_hists) in hists_by_metric.items():
        arrays[f"edges__{metric}"] = edges
        for kg, h in kg_hists.items():
            arrays[f"hist__{metric}__{kg}"] = h
    np.savez(hist_path, **arrays)
    print(f"Wrote {hist_path}")

    heatmap_path = out_dir / "pairwise_heatmaps.pdf"
    plot_heatmaps(hists_by_metric, args.kgs, heatmap_path)
    print(f"Wrote {heatmap_path}")

    if args.qerror_csv:
        scatter_path = out_dir / "divergence_vs_qerror.pdf"
        plot_divergence_vs_qerror(df, Path(args.qerror_csv), scatter_path)
        print(f"Wrote {scatter_path}")


if __name__ == "__main__":
    main()
