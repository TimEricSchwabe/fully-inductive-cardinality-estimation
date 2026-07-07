"""Exhaustive left-deep p-error and rank correlation over existing
`subqueries.tsv` artifacts from `dp_subquery_eval.py` runs.

For each BGP of size n in [min_n, max_n] we enumerate **every** non-cross-product
left-deep join order (i.e. every permutation of the n triples whose every
prefix is connected in the join graph). For each ordering we compute c-out
under four estimators and under true cardinalities:

  - fice        : raw FICE prediction
  - fice_shift  : FICE * 10**beta  (beta read from plans.tsv)
  - geom        : sqrt(fice * endpoint_est)
  - endpoint    : the endpoint's per-subquery estimated_size
  - true        : measured true cardinality

Per query we report, for each estimator e:
  - p-error(e) = cout_true(argmin_e) / min_P cout_true(P)
  - Spearman rho between est-ranked and true-ranked plan list
  - top-1 hit rate (did argmin_e equal argmin_true?)

Aggregation is per shape and pooled overall; a LaTeX booktabs table is
emitted for the paper appendix.

Usage (from repo root):
    uv run python inference/leftdeep_perror.py \
        --root output \
        --out output/evaluation/leftdeep_perror/ \
        --shapes star,path,cycle,diamond,flower,snowflake,binary_tree,path_star_ends \
        --min-n 3 --max-n 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerate non-cross-product left-deep orderings.
#
# A permutation (sigma_1, ..., sigma_n) is admissible iff every prefix mask
# {sigma_1 | ... | sigma_k} for k >= 2 is connected in the join graph. We do
# not have the join graph explicitly here, but `subqueries.tsv` enumerated
# *only* connected subsets, so "prefix is connected" <=> "prefix mask is in
# the enumerated set".
# ---------------------------------------------------------------------------

def enumerate_left_deep(n: int, connected_masks: set[int]) -> list[tuple[int, ...]]:
    full = (1 << n) - 1
    if full not in connected_masks:
        return []
    results: list[tuple[int, ...]] = []
    perm: list[int] = []

    def recurse(chosen: int) -> None:
        k = bin(chosen).count("1")
        if k == n:
            results.append(tuple(perm))
            return
        for i in range(n):
            bit = 1 << i
            if chosen & bit:
                continue
            new_mask = chosen | bit
            if k + 1 >= 2 and new_mask not in connected_masks:
                continue
            perm.append(i)
            recurse(new_mask)
            perm.pop()

    recurse(0)
    return results


def plan_cout(perm: tuple[int, ...], card: dict[int, float]) -> float:
    """Sum of intermediate-join cardinalities (prefixes of length >= 2)."""
    total = 0.0
    mask = 0
    for k, i in enumerate(perm):
        mask |= 1 << i
        if k >= 1:
            c = card.get(mask, float("nan"))
            if not np.isfinite(c):
                return float("nan")
            total += float(c)
    return total


# ---------------------------------------------------------------------------
# Load subqueries.tsv for one shape, build per-query card tables.
# ---------------------------------------------------------------------------

def load_shape(
    shape_dir: Path,
    min_n: int,
    max_n: int,
) -> tuple[list[dict], float]:
    sub = pd.read_csv(shape_dir / "subqueries.tsv", sep="\t")
    plans = pd.read_csv(shape_dir / "plans.tsv", sep="\t", usecols=["beta_used"])
    beta = float(plans["beta_used"].iloc[0])

    sub = sub[(sub["true_card"] >= 0) & np.isfinite(sub["endpoint_est_card"])]
    shift = 10.0 ** beta

    queries: list[dict] = []
    for qi, grp in sub.groupby("query_idx"):
        masks = grp["mask"].astype(int).tolist()
        mask_set = set(masks)
        n = int(max(bin(m).count("1") for m in masks))
        if not (min_n <= n <= max_n):
            continue
        if ((1 << n) - 1) not in mask_set:
            continue

        pred = dict(zip(masks, grp["pred_card"].astype(float).tolist()))
        true = dict(zip(masks, grp["true_card"].astype(float).tolist()))
        qle = dict(zip(masks, grp["endpoint_est_card"].astype(float).tolist()))

        fice = {m: max(pred[m], 1.0) for m in masks}
        fice_shift = {m: fice[m] * shift for m in masks}
        endpoint = {m: max(qle[m], 1.0) for m in masks}
        geom = {m: float(np.sqrt(fice[m] * endpoint[m])) for m in masks}
        true_c = {m: max(true[m], 1.0) for m in masks}

        queries.append({
            "qi": int(qi),
            "n": n,
            "mask_set": mask_set,
            "cards": {
                "fice": fice,
                "fice_shift": fice_shift,
                "geom": geom,
                "endpoint": endpoint,
                "true": true_c,
            },
        })
    return queries, beta


# ---------------------------------------------------------------------------
# Per-query evaluation.
# ---------------------------------------------------------------------------

ESTIMATORS = ("fice", "fice_shift", "geom", "endpoint")


def eval_query(q: dict) -> dict | None:
    perms = enumerate_left_deep(q["n"], q["mask_set"])
    if len(perms) < 2:
        return None

    true_costs = np.array(
        [plan_cout(p, q["cards"]["true"]) for p in perms], dtype=float
    )
    if not np.all(np.isfinite(true_costs)):
        return None
    t_min = float(true_costs.min())
    if t_min <= 0:
        return None

    out: dict = {
        "qi": q["qi"],
        "n_triples": q["n"],
        "n_plans": len(perms),
        "true_opt_cout": t_min,
    }
    for est in ESTIMATORS:
        est_costs = np.array(
            [plan_cout(p, q["cards"][est]) for p in perms], dtype=float
        )
        if not np.all(np.isfinite(est_costs)):
            out[f"p_error_{est}"] = float("nan")
            out[f"rho_{est}"] = float("nan")
            out[f"top1_{est}"] = float("nan")
            continue
        idx = int(np.argmin(est_costs))
        out[f"p_error_{est}"] = float(true_costs[idx] / t_min)
        if np.unique(true_costs).size < 2 or np.unique(est_costs).size < 2:
            rho = float("nan")
        else:
            rho_val, _ = spearmanr(est_costs, true_costs)
            rho = float(rho_val) if np.isfinite(rho_val) else float("nan")
        out[f"rho_{est}"] = rho
        out[f"top1_{est}"] = 1.0 if np.isclose(true_costs[idx], t_min) else 0.0
    return out


# ---------------------------------------------------------------------------
# Aggregation and LaTeX.
# ---------------------------------------------------------------------------

def _geomean(x: np.ndarray) -> float:
    """Geometric mean of p-error values; p-error is always >= 1 so log is safe."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    x = np.clip(x, 1.0, None)
    return float(np.exp(np.mean(np.log(x))))


def summarize(df: pd.DataFrame) -> dict:
    row: dict = {"N": len(df)}
    for est in ESTIMATORS:
        pe = df[f"p_error_{est}"].values
        rho = df[f"rho_{est}"].values
        top1 = df[f"top1_{est}"].values
        row[f"geomean_p_{est}"] = _geomean(pe)
        row[f"median_p_{est}"] = float(np.nanmedian(pe))
        row[f"p90_p_{est}"] = float(np.nanpercentile(pe, 90))
        row[f"p99_p_{est}"] = float(np.nanpercentile(pe, 99))
        row[f"max_p_{est}"] = float(np.nanmax(pe))
        row[f"median_rho_{est}"] = float(np.nanmedian(rho))
        row[f"top1_rate_{est}"] = float(np.nanmean(top1))
    return row


def _fmt(x: float, ndigits: int = 2) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{ndigits}f}"


LATEX_LABELS = {
    "fice": "FICE",
    "fice_shift": r"FICE$\cdot 10^{\beta}$",
    "geom": r"geom(FICE,endpoint)",
    "endpoint": "endpoint",
}


def render_latex(per_shape: pd.DataFrame, overall: dict) -> str:
    """One row per shape + Overall pooled; columns: median p-error and median
    rho per estimator. Best estimator per row bolded (lowest p-error, highest rho).
    """
    ests = list(ESTIMATORS)
    n = len(ests)
    col_spec = "l" + "cc" * n
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " " + "".join(f" & \\multicolumn{{2}}{{c}}{{{LATEX_LABELS[e]}}}" for e in ests) + r" \\",
        " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n)),
        "Shape (N)" + "".join(r" & $p$-error & $\rho$" for _ in ests) + r" \\",
        r"\midrule",
    ]

    def _row(label: str, N: int, row_src: dict) -> str:
        p_vals = [row_src[f"median_p_{e}"] for e in ests]
        r_vals = [row_src[f"median_rho_{e}"] for e in ests]
        best_p = int(np.nanargmin(p_vals)) if np.any(np.isfinite(p_vals)) else -1
        best_r = int(np.nanargmax(r_vals)) if np.any(np.isfinite(r_vals)) else -1
        cells = [f"{label} ({N})"]
        for i, (p, r) in enumerate(zip(p_vals, r_vals)):
            p_s, r_s = _fmt(p), _fmt(r)
            if i == best_p and np.isfinite(p):
                p_s = r"\textbf{" + p_s + "}"
            if i == best_r and np.isfinite(r):
                r_s = r"\textbf{" + r_s + "}"
            cells.extend([p_s, r_s])
        return " & ".join(cells) + r" \\"

    for _, r in per_shape.iterrows():
        lines.append(_row(str(r["shape"]), int(r["N"]), r.to_dict()))
    lines.append(r"\midrule")
    lines.append(_row("Overall", int(overall["N"]), overall))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SHAPES = (
    "star", "path", "cycle", "diamond",
    "flower", "snowflake", "binary_tree", "path_star_ends",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("output"),
                    help="Root containing dbp_v3_<shape>/ directories")
    ap.add_argument("--prefix", default="dbp_v3_",
                    help="Directory prefix per shape (default: dbp_v3_)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for csv + latex")
    ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]

    per_query_rows: list[dict] = []
    per_shape_rows: list[dict] = []

    for shape in shapes:
        shape_dir = args.root / f"{args.prefix}{shape}"
        if not (shape_dir / "subqueries.tsv").exists():
            logger.warning("skip %s: no subqueries.tsv", shape_dir)
            continue
        queries, beta = load_shape(shape_dir, args.min_n, args.max_n)
        rows: list[dict] = []
        for q in queries:
            r = eval_query(q)
            if r is None:
                continue
            r["shape"] = shape
            rows.append(r)
        if not rows:
            logger.warning("%s: 0 usable queries in [%d,%d]",
                           shape, args.min_n, args.max_n)
            continue
        df_shape = pd.DataFrame(rows)
        per_query_rows.extend(rows)
        summ = summarize(df_shape)
        summ["shape"] = shape
        summ["beta"] = beta
        per_shape_rows.append(summ)
        logger.info(
            "%-14s N=%3d  n-range=[%d,%d]  median plans/query=%d  beta=%+.2f",
            shape, len(df_shape), int(df_shape["n_triples"].min()),
            int(df_shape["n_triples"].max()),
            int(df_shape["n_plans"].median()), beta,
        )

    if not per_query_rows:
        logger.error("no queries evaluated; aborting")
        return

    per_query = pd.DataFrame(per_query_rows)
    per_query.to_csv(args.out / "per_query.csv", index=False)

    per_shape = pd.DataFrame(per_shape_rows)
    front = ["shape", "N", "beta"]
    per_shape = per_shape[front + [c for c in per_shape.columns if c not in front]]
    per_shape.to_csv(args.out / "per_shape.csv", index=False)

    overall = summarize(per_query)
    pd.DataFrame([overall]).to_csv(args.out / "overall.csv", index=False)

    latex = render_latex(per_shape, overall)
    (args.out / "leftdeep_perror.tex").write_text(latex)

    # Console summary
    print("\n===== LEFT-DEEP P-ERROR + RANK CORRELATION =====")
    print(f"{'shape':<16} {'N':>4} | " +
          " | ".join(f"{e:>10}" for e in ESTIMATORS))
    print(f"{'':<16} {'':>4} | " +
          " | ".join(f"{'med p / ρ':>10}" for _ in ESTIMATORS))
    for _, r in per_shape.iterrows():
        cells = []
        for e in ESTIMATORS:
            cells.append(f"{_fmt(r[f'median_p_{e}'])}/{_fmt(r[f'median_rho_{e}'])}")
        print(f"{r['shape']:<16} {int(r['N']):>4} | " +
              " | ".join(f"{c:>10}" for c in cells))
    print(f"{'Overall':<16} {int(overall['N']):>4} | " +
          " | ".join(
              f"{_fmt(overall[f'median_p_{e}'])}/{_fmt(overall[f'median_rho_{e}'])}"
              for e in ESTIMATORS))

    print("\n===== TAIL P-ERROR (pooled across shapes) =====")
    print(f"{'est':<12} {'geomean':>8} {'median':>8} {'p90':>8} {'p99':>8} {'max':>10} {'top-1':>7}")
    for e in ESTIMATORS:
        print(f"{e:<12} "
              f"{_fmt(overall[f'geomean_p_{e}']):>8} "
              f"{_fmt(overall[f'median_p_{e}']):>8} "
              f"{_fmt(overall[f'p90_p_{e}']):>8} "
              f"{_fmt(overall[f'p99_p_{e}']):>8} "
              f"{_fmt(overall[f'max_p_{e}'], 1):>10} "
              f"{_fmt(overall[f'top1_rate_{e}']):>7}")

    print(f"\nWrote: {args.out / 'per_query.csv'}")
    print(f"Wrote: {args.out / 'per_shape.csv'}")
    print(f"Wrote: {args.out / 'overall.csv'}")
    print(f"Wrote: {args.out / 'leftdeep_perror.tex'}")


if __name__ == "__main__":
    main()
