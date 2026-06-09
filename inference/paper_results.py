"""Paper-ready aggregate tables + plots for a FICE eval run vs the SPARQL endpoint.

Inputs (per run):
  - leftdeep_perror per_query.csv        (output/evaluation/leftdeep_perror_<run>/per_query.csv)
  - dp_subquery_eval plans.tsv per shape (output/<prefix>_<shape>/plans.tsv)
  - SPARQL endpoint                      (live; only re-queried if endpoint cache missing)

Outputs (under --out, default output/evaluation/paper_<run>/):
  - paper_table.txt                       printed table dump (same as stdout)
  - paper_table_pooled.csv                pooled row, paper-ready
  - paper_table_per_shape.csv             per-shape rows
  - perror_ccdf.{pdf,png}                 CCDF of p-error: FICE vs endpoint
  - cout_scatter.{pdf,png}                per-query scatter, FICE vs endpoint c-out (log-log)
  - endpoint_cout_skipsort.csv            re-queried endpoint c-out cache (skip Sort + IndexScan)

Why c-out skips Sort: the endpoint's Sort operator's `result_rows` always equals
its single child's, so summing it double-counts the intermediate size and
inflates the endpoint's c-out relative to FICE-DP's c-out (which sums one value
per non-singleton subset). Skipping Sort makes the metric directly comparable.

Usage (from repo root, with the SPARQL endpoint running on localhost:7020):

    .venv/bin/python inference/paper_results.py \\
        --prefix dbp_ranking_model_ranknet \\
        --label ranknet \\
        --leftdeep-dir output/evaluation/leftdeep_perror_ranking_model_ranknet \\
        --out output/evaluation/paper_ranking_model_ranknet
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import urllib.parse
import urllib.request
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

SHAPES = [
    "star", "path", "cycle", "diamond",
    "flower", "binary_tree", "path_star_ends",
]

# Same query files used by dp_subquery_eval; see configs/coma_cluster.md.
QFILES = {
    "star":           "data/dbpedia100k_star_queries/raw/59687708",
    "path":           "data/dbpedia100k_path_queries/raw/59898644",
    "cycle":          "data/dbpedia100k_cycle_queries/raw/queries.json",
    "diamond":        "data/dbpedia100k_diamond_queries/raw/queries.json",
    "flower":         "data/dbpedia100k_flower_queries/raw/queries.json",
    "binary_tree":    "data/dbpedia100k_binary_tree_queries/raw/queries.json",
    "path_star_ends": "data/dbpedia100k_path_star_ends_queries/raw/queries.json",
}

DEFAULT_ENDPOINT = "http://localhost:7020/"
DEFAULT_ENDPOINT_ACTION = "qlever_json_export"
DEFAULT_MAX_QUERIES = 100
DEFAULT_MIN_N = 3
DEFAULT_MAX_N = 7

# ---------------------------------------------------------------------------
# SPARQL endpoint helpers (re-query for c-out skip-Sort cache)
# ---------------------------------------------------------------------------

def _fmt_term(x):
    if not isinstance(x, str):
        return str(x)
    if x.startswith("?"):
        return x
    if x.startswith("<") and x.endswith(">"):
        return x
    return f"<{x}>"


def _query_tree(triples, endpoint_url, action, timeout=30):
    bgp = "\n  ".join(
        f"{_fmt_term(s)} {_fmt_term(p)} {_fmt_term(o)} ." for s, p, o in triples
    )
    sparql = f"SELECT * WHERE {{\n  {bgp}\n}}"
    url = endpoint_url.rstrip("/") + "/?" + urllib.parse.urlencode(
        {"query": sparql, "action": action, "send": "0"}
    )
    with urllib.request.urlopen(url, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload.get("runtimeInformation", {}).get("query_execution_tree")


def _walk_cout(tree, skip_sort=True):
    """Sum result_rows over join-style nodes; skip IndexScan (and Sort if requested)."""
    if tree is None:
        return float("nan"), 0
    skip_prefixes = ("IndexScan",) + (("Sort",) if skip_sort else ())
    total = 0.0
    n_ops = 0

    def walk(node):
        nonlocal total, n_ops
        desc = node.get("description", "")
        if not any(desc.startswith(p) for p in skip_prefixes):
            total += float(node.get("result_rows", 0))
            n_ops += 1
        for c in node.get("children") or []:
            walk(c)

    walk(tree)
    return total, n_ops


def _val_queries_for_shape(qfile_path, max_queries, min_n, max_n):
    """Replicate dp_subquery_eval's 80/20 train/val split with seed 42, then take
    the first `max_queries` val queries with min_n <= n_triples <= max_n.

    Returns list of (qi_in_val_list, file_idx, triples). qi_in_val_list is what
    plans.tsv records as `query_idx`.
    """
    with open(qfile_path) as f:
        raw = json.load(f)
    n_total = len(raw)
    random.seed(42)
    idx = list(range(n_total))
    random.shuffle(idx)
    split = int(n_total * 0.8)
    val = idx[split:][:max_queries]
    out = []
    for qi, fi in enumerate(val):
        q = raw[fi]
        triples = q["triples"] if isinstance(q, dict) and "triples" in q else q
        triples = [tuple(t) for t in triples]
        if min_n <= len(triples) <= max_n:
            out.append((qi, fi, triples))
    return out


def _build_endpoint_cache(out_csv, endpoint_url, action, max_queries, min_n, max_n):
    """Re-query the SPARQL endpoint for the held-out val split and cache c-out (skip Sort)."""
    rows = []
    for shape, rel_path in QFILES.items():
        path = REPO_ROOT / rel_path
        items = _val_queries_for_shape(path, max_queries, min_n, max_n)
        n_err = 0
        logger.info(f"  {shape}: {len(items)} queries")
        for qi, fi, triples in items:
            try:
                tree = _query_tree(triples, endpoint_url, action)
            except Exception as e:
                n_err += 1
                continue
            if tree is None:
                n_err += 1
                continue
            ws, _ = _walk_cout(tree, skip_sort=False)
            ss, n_ops = _walk_cout(tree, skip_sort=True)
            rows.append({
                "shape": shape, "query_idx": qi, "file_idx": fi,
                "cout_with_sort": ws, "cout_skip_sort": ss, "n_ops": n_ops,
            })
        if n_err:
            logger.warning(f"    {n_err} queries errored / had no tree")
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved endpoint c-out cache to {out_csv}  N={len(df)}")
    return df


# ---------------------------------------------------------------------------
# Statistics: bootstrap geomean CI, Wilson binomial CI
# ---------------------------------------------------------------------------

def boot_geomean(x, n_boot=2000, seed=0):
    """Geometric mean and a non-parametric bootstrap 95 % CI."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    gms = np.exp(np.mean(np.log(samples), axis=1))
    return (float(np.exp(np.mean(np.log(x)))),
            float(np.percentile(gms, 2.5)),
            float(np.percentile(gms, 97.5)))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def percentile(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else float("nan")


def fmt_ci(g, lo, hi, dp=2):
    if not np.isfinite(g):
        return "—"
    return f"{g:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]"


# ---------------------------------------------------------------------------
# Build merged per-query frame: leftdeep_perror p-error + plans.tsv c-out
# ---------------------------------------------------------------------------

def build_per_query(prefix, leftdeep_dir, ep_cache):
    """Merge p-error per_query.csv with skip-Sort endpoint c-out and FICE plan c-out."""
    pq = pd.read_csv(REPO_ROOT / leftdeep_dir / "per_query.csv")
    plans_rows = []
    for shape in SHAPES:
        path = REPO_ROOT / "output" / f"{prefix}_{shape}" / "plans.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        df = df[(df["n_triples"] >= DEFAULT_MIN_N) & (df["n_triples"] <= DEFAULT_MAX_N)]
        df = df[["query_idx", "n_triples", "cout_true_same_plan", "endpoint_cout_true"]].copy()
        df["shape"] = shape
        plans_rows.append(df)
    plans = pd.concat(plans_rows, ignore_index=True)

    ep = ep_cache.rename(columns={"shape": "shape_name"})[
        ["shape_name", "query_idx", "cout_with_sort", "cout_skip_sort"]
    ]
    plans = plans.merge(
        ep, left_on=["shape", "query_idx"], right_on=["shape_name", "query_idx"], how="left",
    )
    plans = plans.drop(columns=["shape_name"])

    merged = pq.merge(
        plans[["shape", "query_idx", "cout_true_same_plan", "cout_skip_sort"]],
        left_on=["shape", "qi"], right_on=["shape", "query_idx"], how="left",
    )
    merged["cout_ratio"] = (merged["cout_true_same_plan"].clip(lower=1)
                            / merged["cout_skip_sort"].clip(lower=1))
    return merged


# ---------------------------------------------------------------------------
# DP p-error: same plan space as the c-out comparison (bushy)
# ---------------------------------------------------------------------------
#
# Today's `p_error_fice` / `p_error_endpoint` from leftdeep_perror.py are
# computed by exhaustive enumeration over *left-deep* plans. The c-out
# comparison reported above is in the *bushy* plan space (FICE-DP plan vs
# endpoint's actually-emitted plan). The function below closes that gap by
# computing p-error in the bushy plan space:
#
#     oracle_min(q) = dp_best_plan(masks_q, true_card_q).cost[full_mask]
#     p_error_fice_dp(q)     = cout_true_same_plan(q) / oracle_min(q)
#     p_error_endpoint_dp(q) = cout_skip_sort(q)      / oracle_min(q)
#
# All inputs are already on disk after dp_subquery_eval has run — no endpoint
# round-trips are needed here.

def build_per_query_dp(prefix, ep_cache):
    """Per-query bushy-DP p-error, dual to ``build_per_query`` (left-deep).

    For each shape under ``output/<prefix>_<shape>/``:
      - Read ``subqueries.tsv`` → run ``dp_best_plan(masks, true_card)`` to get
        the bushy oracle minimum c-out.
      - Read ``plans.tsv`` → ``cout_true_same_plan`` is the true c-out of the
        FICE-DP plan.
      - Look up endpoint emitted-plan true c-out from the skip-Sort cache.

    Returns one row per query with both p-errors plus the raw c-outs.
    """
    # Local import — dp_subquery_eval imports torch and mutates sys.path; we do
    # not want either of those at module load time.
    from dp_subquery_eval import dp_best_plan

    rows = []
    for shape in SHAPES:
        sd = REPO_ROOT / "output" / f"{prefix}_{shape}"
        sub_path = sd / "subqueries.tsv"
        plans_path = sd / "plans.tsv"
        if not sub_path.exists() or not plans_path.exists():
            continue

        sub = pd.read_csv(sub_path, sep="\t")
        sub = sub[sub["true_card"] >= 0]   # drop subsets where the endpoint failed
        if sub.empty:
            continue

        plans = pd.read_csv(plans_path, sep="\t")
        plans = plans[(plans["n_triples"] >= DEFAULT_MIN_N)
                      & (plans["n_triples"] <= DEFAULT_MAX_N)
                      & np.isfinite(plans["cout_true_same_plan"])]
        if plans.empty:
            continue
        plans_by_qi = plans.set_index("query_idx")

        ep_shape = ep_cache[ep_cache["shape"] == shape].set_index("query_idx") \
            if "shape" in ep_cache.columns else pd.DataFrame()

        for qi, grp in sub.groupby("query_idx"):
            if qi not in plans_by_qi.index:
                continue
            masks = grp["mask"].astype(int).tolist()
            tcard = {int(r.mask): float(r.true_card) for r in grp.itertuples()}
            cost, _, full_mask = dp_best_plan(masks, tcard)
            oracle = cost.get(full_mask, float("nan"))
            if not np.isfinite(oracle) or oracle <= 0:
                continue

            fice_true = float(plans_by_qi.loc[qi, "cout_true_same_plan"])
            ep_true = float("nan")
            if not ep_shape.empty and qi in ep_shape.index:
                v = ep_shape.loc[qi, "cout_skip_sort"]
                if np.isfinite(v):
                    ep_true = float(v)

            p_error_fice = fice_true / oracle
            p_error_ep = (ep_true / oracle) if np.isfinite(ep_true) else float("nan")

            # Sanity: the FICE-DP plan is in the same search space as the oracle,
            # so its true c-out must be ≥ the oracle minimum (modulo FP noise).
            assert p_error_fice >= 1.0 - 1e-6, (
                f"FICE-DP p_error < 1 at shape={shape} qi={qi}: "
                f"{p_error_fice!r} (fice_true={fice_true}, oracle={oracle})"
            )
            # The endpoint's plan should also be in that search space (BGP-only,
            # no cross products). If a value dips below 1 it's almost certainly
            # FP noise from QLever's `result_rows` vs our enumerated true_card;
            # warn rather than crash so the run still produces a table.
            if np.isfinite(p_error_ep) and p_error_ep < 1.0 - 1e-3:
                logger.warning(
                    f"endpoint p_error < 1 at shape={shape} qi={qi}: "
                    f"{p_error_ep:.6f} (ep_true={ep_true}, oracle={oracle})"
                )

            rows.append({
                "shape": shape,
                "query_idx": int(qi),
                "n_triples": int(grp["n_triples"].iloc[0]),
                "oracle_min_cout": oracle,
                "fice_dp_cout_true": fice_true,
                "endpoint_cout_true": ep_true,
                "p_error_fice_dp": p_error_fice,
                "p_error_endpoint_dp": p_error_ep,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_summary(prefix):
    """Return (per_shape_df, pooled_dict) with mean/std of log10(pred/true)."""
    rows = []
    all_f, all_e = [], []
    for shape in SHAPES:
        f = REPO_ROOT / "output" / f"{prefix}_{shape}" / "subqueries.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        df = df[(df["true_card"] > 0) & (df["pred_card"] > 0) & (df["endpoint_est_card"] > 0)]
        df = df[df["n_triples"] >= 2]
        f_lr = np.log10(df["pred_card"].values / df["true_card"].values)
        e_lr = np.log10(df["endpoint_est_card"].values / df["true_card"].values)
        rows.append({
            "shape": shape, "n_sub": len(df),
            "fice_log_mean": float(f_lr.mean()), "fice_log_std": float(f_lr.std()),
            "endpoint_log_mean": float(e_lr.mean()), "endpoint_log_std": float(e_lr.std()),
        })
        all_f.append(f_lr); all_e.append(e_lr)
    if not rows:
        return pd.DataFrame(rows), {}
    all_f = np.concatenate(all_f); all_e = np.concatenate(all_e)
    pooled = {
        "n_sub": int(len(all_f)),
        "fice_log_mean": float(all_f.mean()), "fice_log_std": float(all_f.std()),
        "endpoint_log_mean": float(all_e.mean()), "endpoint_log_std": float(all_e.std()),
    }
    return pd.DataFrame(rows), pooled


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ccdf(merged, out_dir, label):
    """Complementary CDF of p-error: FICE vs endpoint (log-log).

    P(p-error > x) — heavy-tail-friendly. Lower curve = better at every quantile.
    Dimensions and font sizes mirror scripts/evaluation/plotting.py:ccdf_q_error
    (COL_W = 4.6, GOLDEN = 0.618, height = COL_W * GOLDEN * 0.5).
    """
    import matplotlib.pyplot as plt

    COL_W, GOLDEN = 4.6, 0.618
    SMALL_FONT_RC = {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
    }

    series = [
        ("FICE", merged["p_error_fice"].dropna().values, "C0", "-"),
        ("endpoint", merged["p_error_endpoint"].dropna().values, "C3", "-."),
    ]

    with plt.rc_context(SMALL_FONT_RC):
        fig, ax = plt.subplots(figsize=(COL_W, COL_W * GOLDEN * 0.5))
        for name, x, color, ls in series:
            x = np.sort(x)
            ccdf = 1.0 - np.arange(1, len(x) + 1) / len(x)
            ax.step(x, ccdf, where="post", label=name, color=color, ls=ls, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"p-error $x$")
        ax.set_ylabel(r"$P \ (\mathrm{p\text{-}error} > x)$")
        ax.legend(frameon=False, loc="lower left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"perror_ccdf.{ext}", dpi=200)
        plt.close(fig)


def plot_cout_scatter(merged, out_dir, label):
    """Per-query scatter: FICE c-out (y) vs endpoint c-out (x), log-log.

    Below identity = FICE plan cheaper. Color by shape.
    """
    import matplotlib.pyplot as plt

    df = merged.dropna(subset=["cout_true_same_plan", "cout_skip_sort"]).copy()
    fice = df["cout_true_same_plan"].clip(lower=1.0).values
    endp = df["cout_skip_sort"].clip(lower=1.0).values

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, shape in enumerate(SHAPES):
        m = df["shape"] == shape
        if not m.any():
            continue
        ax.scatter(endp[m], fice[m], s=14, alpha=0.7, color=cmap(i), label=shape)

    lo = max(min(endp.min(), fice.min()), 1.0) * 0.5
    hi = max(endp.max(), fice.max()) * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="identity (tie)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("endpoint plan true c-out")
    ax.set_ylabel("FICE plan true c-out")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cout_scatter.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables (emitted to stdout + CSV)
# ---------------------------------------------------------------------------

def emit_tables(merged, label, calib_per_shape, calib_pooled, lines):
    def out(s=""):
        print(s)
        lines.append(s)

    out(f"\n========== Paper-ready aggregate — run: {label} ==========\n")

    # --- Pooled row ---
    pe_f = merged["p_error_fice"].values
    pe_e = merged["p_error_endpoint"].values
    pe_g = merged["p_error_geom"].values
    cr   = merged["cout_ratio"].dropna().values

    pooled = {
        "label": label,
        "N_pq": int(len(merged)),
        "N_cout": int(len(cr)),
    }
    pooled["pe_fice_geom"], pooled["pe_fice_lo"], pooled["pe_fice_hi"] = boot_geomean(pe_f)
    pooled["pe_endp_geom"], pooled["pe_endp_lo"], pooled["pe_endp_hi"] = boot_geomean(pe_e)
    pooled["pe_geom_geom"], pooled["pe_geom_lo"], pooled["pe_geom_hi"] = boot_geomean(pe_g)
    pooled["cr_geom"], pooled["cr_lo"], pooled["cr_hi"] = boot_geomean(cr)

    pooled["pe_fice_p90"]  = percentile(pe_f, 90)
    pooled["pe_fice_p99"]  = percentile(pe_f, 99)
    pooled["pe_fice_max"]  = float(np.nanmax(pe_f))
    pooled["pe_endp_p90"]  = percentile(pe_e, 90)
    pooled["pe_endp_p99"]  = percentile(pe_e, 99)
    pooled["pe_endp_max"]  = float(np.nanmax(pe_e))
    pooled["cr_p90"]       = percentile(cr, 90)
    pooled["cr_p99"]       = percentile(cr, 99)
    pooled["cr_max"]       = float(np.nanmax(cr)) if len(cr) else float("nan")

    n_w = int((cr < 1).sum()); n_t = int(np.isclose(cr, 1.0).sum())
    p, lo, hi = wilson_ci(n_w, len(cr))
    pooled.update({"win_strict": n_w, "win_tie": n_t, "win_total": int(len(cr)),
                   "win_pct": p, "win_lo": lo, "win_hi": hi,
                   "cal_fice_mean": calib_pooled.get("fice_log_mean"),
                   "cal_fice_std":  calib_pooled.get("fice_log_std"),
                   "cal_endp_mean": calib_pooled.get("endpoint_log_mean"),
                   "cal_endp_std":  calib_pooled.get("endpoint_log_std")})

    out("Pooled (held-out DBpedia100k):")
    out(f"  p-error geomean (95% CI)   FICE: {fmt_ci(pooled['pe_fice_geom'], pooled['pe_fice_lo'], pooled['pe_fice_hi'])}    "
        f"endpoint: {fmt_ci(pooled['pe_endp_geom'], pooled['pe_endp_lo'], pooled['pe_endp_hi'])}    "
        f"geom-blend: {fmt_ci(pooled['pe_geom_geom'], pooled['pe_geom_lo'], pooled['pe_geom_hi'])}")
    out(f"  p-error p90 / p99 / max    FICE: {pooled['pe_fice_p90']:.2f} / {pooled['pe_fice_p99']:.2f} / {pooled['pe_fice_max']:.1f}    "
        f"endpoint: {pooled['pe_endp_p90']:.2f} / {pooled['pe_endp_p99']:.2f} / {pooled['pe_endp_max']:.1f}")
    out(f"  c-out ratio (FICE/endpoint, skip-Sort)   geomean: {fmt_ci(pooled['cr_geom'], pooled['cr_lo'], pooled['cr_hi'])}    "
        f"p90={pooled['cr_p90']:.2f}  p99={pooled['cr_p99']:.2f}  max={pooled['cr_max']:.1f}    (N={pooled['N_cout']})")
    out(f"  FICE strict-win rate: {n_w}/{pooled['N_cout']} = {p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]   "
        f"tie={n_t} ({n_t/max(pooled['N_cout'],1)*100:.1f}%)")
    out(f"  log10(pred/true) calibration  FICE: mean={pooled['cal_fice_mean']:+.2f} std={pooled['cal_fice_std']:.2f}    "
        f"endpoint: mean={pooled['cal_endp_mean']:+.2f} std={pooled['cal_endp_std']:.2f}")

    # --- Per-shape table ---
    out("\nPer-shape:")
    hdr = (f"  {'shape':<14} | {'N':>3} | {'cout_ratio (CI)':<26} | {'p-err FICE (CI)':<26} | "
           f"{'p-err endpoint (CI)':<26} | {'F.p90':>6} | {'F.p99':>7} | {'F.max':>8} | {'win%':>14}")
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))

    per_shape_records = []
    for shape, g in merged.groupby("shape", sort=False):
        n = len(g)
        pe_f = g["p_error_fice"].values
        pe_e = g["p_error_endpoint"].values
        cr   = g["cout_ratio"].dropna().values
        gf = boot_geomean(pe_f); ge = boot_geomean(pe_e); gc = boot_geomean(cr)
        n_w = int((cr < 1).sum()); n_t = int(np.isclose(cr, 1.0).sum())
        wp, wlo, whi = wilson_ci(n_w, len(cr)) if len(cr) else (float("nan"),)*3
        out(f"  {shape:<14} | {n:>3} | {fmt_ci(*gc):<26} | {fmt_ci(*gf):<26} | {fmt_ci(*ge):<26} | "
            f"{percentile(pe_f, 90):>6.2f} | {percentile(pe_f, 99):>7.2f} | {np.nanmax(pe_f):>8.1f} | "
            f"{wp*100:>5.1f}%[{wlo*100:>4.1f},{whi*100:>4.1f}]")
        per_shape_records.append({
            "shape": shape, "N": n,
            "pe_fice_geom": gf[0], "pe_fice_lo": gf[1], "pe_fice_hi": gf[2],
            "pe_endp_geom": ge[0], "pe_endp_lo": ge[1], "pe_endp_hi": ge[2],
            "cr_geom": gc[0], "cr_lo": gc[1], "cr_hi": gc[2],
            "pe_fice_p90": percentile(pe_f, 90),
            "pe_fice_p99": percentile(pe_f, 99),
            "pe_fice_max": float(np.nanmax(pe_f)) if n else float("nan"),
            "pe_endp_p90": percentile(pe_e, 90),
            "pe_endp_p99": percentile(pe_e, 99),
            "pe_endp_max": float(np.nanmax(pe_e)) if n else float("nan"),
            "win_strict": n_w, "win_tie": n_t, "win_total": int(len(cr)),
            "win_pct": wp, "win_lo": wlo, "win_hi": whi,
        })

    # --- Calibration per-shape ---
    out("\nCalibration: log10(pred/true)  (multi-triple sub-BGPs only)")
    out(f"  {'shape':<14} | {'N_sub':>5} | {'FICE mean / std':<18} | {'endpoint mean / std':<18}")
    for r in calib_per_shape.to_dict("records"):
        out(f"  {r['shape']:<14} | {r['n_sub']:>5} | "
            f"{r['fice_log_mean']:+.2f} / {r['fice_log_std']:.2f}        | "
            f"{r['endpoint_log_mean']:+.2f} / {r['endpoint_log_std']:.2f}")

    return pooled, pd.DataFrame(per_shape_records)


def emit_dp_perror_tables(merged_dp, label, lines):
    """Pooled + per-shape tables for bushy-DP p-error (dual to ``emit_tables``).

    Returns ``(pooled_dict, per_shape_df, winrates_df)``. ``winrates_df`` has
    one row per shape plus a ``POOLED`` row, ready to dump as a CSV.
    """
    def out(s=""):
        print(s)
        lines.append(s)

    if merged_dp.empty:
        out("\n========== DP p-error (bushy plan space) — no data, skipping ==========\n")
        return None, None, None

    out(f"\n========== DP p-error (bushy plan space) — run: {label} ==========\n")
    out("Plan space: bushy DP. Oracle = min c-out over bushy plans on TRUE cards.")
    out("FICE numerator = cout_true_same_plan (FICE-DP plan).  "
        "Endpoint numerator = endpoint emitted-plan c-out (skip-Sort).")

    pe_f = merged_dp["p_error_fice_dp"].values
    pe_e = merged_dp["p_error_endpoint_dp"].dropna().values
    n_pq = int(len(merged_dp))
    n_ep = int(np.isfinite(merged_dp["p_error_endpoint_dp"]).sum())

    pf_g = boot_geomean(pe_f)
    pe_g = boot_geomean(pe_e)

    out(f"\n  N_FICE = {n_pq}    N_endpoint = {n_ep}")
    out(f"  p-error geomean (95% CI)   FICE: {fmt_ci(*pf_g)}    "
        f"endpoint: {fmt_ci(*pe_g)}")
    out(f"  p-error p90 / p99 / max    "
        f"FICE: {percentile(pe_f, 90):.2f} / {percentile(pe_f, 99):.2f} / "
        f"{(np.nanmax(pe_f) if n_pq else float('nan')):.1f}    "
        f"endpoint: {percentile(pe_e, 90):.2f} / {percentile(pe_e, 99):.2f} / "
        f"{(np.nanmax(pe_e) if n_ep else float('nan')):.1f}")

    # Win/tie/loss on per-query (p_error_fice_dp < / == / > p_error_endpoint_dp).
    both = merged_dp.dropna(subset=["p_error_fice_dp", "p_error_endpoint_dp"])
    n_total = int(len(both))
    n_w = int((both["p_error_fice_dp"] < both["p_error_endpoint_dp"]).sum())
    n_t = int(np.isclose(both["p_error_fice_dp"].values,
                         both["p_error_endpoint_dp"].values).sum())
    n_l = n_total - n_w - n_t
    p, lo, hi = wilson_ci(n_w, n_total)
    out(f"  FICE strict-win on p-error: {n_w}/{n_total} = {p*100:.1f}% "
        f"[{lo*100:.1f}, {hi*100:.1f}]   tie={n_t}  loss={n_l}")

    pooled = {
        "label": label,
        "N_pq_fice": n_pq,
        "N_pq_endp": n_ep,
        "pe_fice_geom": pf_g[0], "pe_fice_lo": pf_g[1], "pe_fice_hi": pf_g[2],
        "pe_endp_geom": pe_g[0], "pe_endp_lo": pe_g[1], "pe_endp_hi": pe_g[2],
        "pe_fice_p90": percentile(pe_f, 90),
        "pe_fice_p99": percentile(pe_f, 99),
        "pe_fice_max": float(np.nanmax(pe_f)) if n_pq else float("nan"),
        "pe_endp_p90": percentile(pe_e, 90),
        "pe_endp_p99": percentile(pe_e, 99),
        "pe_endp_max": float(np.nanmax(pe_e)) if n_ep else float("nan"),
        "win_strict": n_w, "win_tie": n_t, "win_loss": n_l,
        "win_total": n_total, "win_pct": p, "win_lo": lo, "win_hi": hi,
    }

    # --- Per-shape ---
    out("\nPer-shape:")
    hdr = (f"  {'shape':<14} | {'N':>3} | {'p-err FICE (CI)':<26} | "
           f"{'p-err endpoint (CI)':<26} | {'F.p99':>7} | {'E.p99':>7} | {'win%':>14}")
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))

    per_shape_records = []
    winrate_rows = []
    for shape, g in merged_dp.groupby("shape", sort=False):
        n = int(len(g))
        pf = g["p_error_fice_dp"].values
        pe = g["p_error_endpoint_dp"].dropna().values
        gf = boot_geomean(pf)
        ge = boot_geomean(pe)
        bb = g.dropna(subset=["p_error_fice_dp", "p_error_endpoint_dp"])
        nw = int((bb["p_error_fice_dp"] < bb["p_error_endpoint_dp"]).sum())
        nt = int(np.isclose(bb["p_error_fice_dp"].values,
                            bb["p_error_endpoint_dp"].values).sum())
        nl = int(len(bb)) - nw - nt
        wp, wlo, whi = wilson_ci(nw, len(bb)) if len(bb) else (float("nan"),) * 3
        out(f"  {shape:<14} | {n:>3} | {fmt_ci(*gf):<26} | {fmt_ci(*ge):<26} | "
            f"{percentile(pf, 99):>7.2f} | {percentile(pe, 99):>7.2f} | "
            f"{wp*100:>5.1f}%[{wlo*100:>4.1f},{whi*100:>4.1f}]")
        per_shape_records.append({
            "shape": shape, "N": n,
            "pe_fice_geom": gf[0], "pe_fice_lo": gf[1], "pe_fice_hi": gf[2],
            "pe_endp_geom": ge[0], "pe_endp_lo": ge[1], "pe_endp_hi": ge[2],
            "pe_fice_p90": percentile(pf, 90),
            "pe_fice_p99": percentile(pf, 99),
            "pe_fice_max": float(np.nanmax(pf)) if n else float("nan"),
            "pe_endp_p90": percentile(pe, 90),
            "pe_endp_p99": percentile(pe, 99),
            "pe_endp_max": float(np.nanmax(pe)) if len(pe) else float("nan"),
            "win_strict": nw, "win_tie": nt, "win_loss": nl,
            "win_total": int(len(bb)),
            "win_pct": wp, "win_lo": wlo, "win_hi": whi,
        })
        winrate_rows.append({
            "label": label, "shape": shape,
            "win_strict": nw, "win_tie": nt, "win_loss": nl,
            "win_total": int(len(bb)),
            "win_pct": wp, "win_lo": wlo, "win_hi": whi,
        })

    winrate_rows.append({
        "label": label, "shape": "POOLED",
        "win_strict": n_w, "win_tie": n_t, "win_loss": n_l,
        "win_total": n_total,
        "win_pct": p, "win_lo": lo, "win_hi": hi,
    })

    return pooled, pd.DataFrame(per_shape_records), pd.DataFrame(winrate_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="dbp_ranking_model_ranknet",
                    help="Per-shape eval dir prefix (looked up under output/<prefix>_<shape>/)")
    ap.add_argument("--label", default="ranknet",
                    help="Short label used in plot legends and headlines")
    ap.add_argument("--leftdeep-dir", default="output/evaluation/leftdeep_perror_ranking_model_ranknet",
                    help="Directory containing per_query.csv from leftdeep_perror.py")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: output/evaluation/paper_<label>/)")
    ap.add_argument("--endpoint-cache",
                    default="output/evaluation/endpoint_cout_skipsort_v2.csv",
                    help="Endpoint c-out cache path (created if missing)")
    ap.add_argument("--regenerate-cache", action="store_true",
                    help="Re-query the SPARQL endpoint even if cache exists")
    ap.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    ap.add_argument("--endpoint-action", default=DEFAULT_ENDPOINT_ACTION)
    ap.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    ap.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    ap.add_argument("--dp-perror", dest="dp_perror", action="store_true", default=True,
                    help="Compute p-error in the bushy-DP plan space (default: on)")
    ap.add_argument("--no-dp-perror", dest="dp_perror", action="store_false",
                    help="Skip the bushy-DP p-error tables")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / f"output/evaluation/paper_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = REPO_ROOT / args.endpoint_cache

    # Step 1: endpoint c-out cache (skip-Sort)
    if cache_path.exists() and not args.regenerate_cache:
        ep_cache = pd.read_csv(cache_path)
        logger.info(f"Loaded endpoint c-out cache  N={len(ep_cache)}  ({cache_path})")
    else:
        logger.info(f"Building endpoint c-out cache (re-querying {args.endpoint_url}) ...")
        ep_cache = _build_endpoint_cache(
            cache_path, args.endpoint_url, args.endpoint_action,
            args.max_queries, args.min_n, args.max_n,
        )

    # Step 2: merge per-query frame
    merged = build_per_query(args.prefix, args.leftdeep_dir, ep_cache)
    merged.to_csv(out_dir / "per_query_merged.csv", index=False)

    # Step 3: calibration
    calib_per_shape, calib_pooled = calibration_summary(args.prefix)

    # Step 4: tables
    lines = []
    pooled, per_shape = emit_tables(merged, args.label, calib_per_shape, calib_pooled, lines)
    pd.DataFrame([pooled]).to_csv(out_dir / "paper_table_pooled.csv", index=False)
    per_shape.to_csv(out_dir / "paper_table_per_shape.csv", index=False)
    calib_per_shape.to_csv(out_dir / "calibration_per_shape.csv", index=False)

    # Step 4b: DP p-error tables (bushy plan space — same plan space as the
    # c-out comparison above). Same query universe as plans.tsv; nothing is
    # re-queried from the endpoint here.
    n_dp_fice = n_dp_endp = 0
    if args.dp_perror:
        merged_dp = build_per_query_dp(args.prefix, ep_cache)
        merged_dp.to_csv(out_dir / "per_query_dp_perror.csv", index=False)
        result = emit_dp_perror_tables(merged_dp, args.label, lines)
        if result is not None and result[0] is not None:
            pooled_dp, per_shape_dp, winrates_dp = result
            pd.DataFrame([pooled_dp]).to_csv(out_dir / "paper_dp_perror_pooled.csv", index=False)
            per_shape_dp.to_csv(out_dir / "paper_dp_perror_per_shape.csv", index=False)
            winrates_dp.to_csv(out_dir / "paper_dp_perror_winrates.csv", index=False)
            n_dp_fice = pooled_dp["N_pq_fice"]
            n_dp_endp = pooled_dp["N_pq_endp"]

    # Sanity: same query universe as left-deep p-error (per shape).
    if args.dp_perror and n_dp_fice:
        n_ld = int(merged["p_error_fice"].notna().sum())
        lines.append("")
        lines.append(f"[universe check] left-deep p-error N = {n_ld}    "
                     f"DP p-error N (FICE / endpoint) = {n_dp_fice} / {n_dp_endp}")

    (out_dir / "paper_table.txt").write_text("\n".join(lines))
    logger.info(f"Wrote paper_table.txt, paper_table_pooled.csv, paper_table_per_shape.csv to {out_dir}")
    if args.dp_perror:
        logger.info(f"Wrote paper_dp_perror_*.csv (bushy-DP plan space) to {out_dir}")

    # Step 5: plots
    try:
        plot_ccdf(merged, out_dir, args.label)
        plot_cout_scatter(merged, out_dir, args.label)
        logger.info(f"Wrote perror_ccdf.{{pdf,png}}, cout_scatter.{{pdf,png}} to {out_dir}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
