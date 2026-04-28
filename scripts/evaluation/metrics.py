"""Q-error computation and aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CARDINALITY_BIN_EDGES, CARDINALITY_BIN_LABELS, EPSILON


def per_query_q_error(
    pred: np.ndarray, true: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    p = np.maximum(np.asarray(pred, dtype=np.float64), epsilon)
    t = np.maximum(np.asarray(true, dtype=np.float64), epsilon)
    return np.maximum(p / t, t / p)


def _log_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    lp = np.log1p(np.maximum(pred, 0.0))
    lt = np.log1p(np.maximum(true, 0.0))
    if lp.size < 2 or lp.std() == 0.0 or lt.std() == 0.0:
        return np.nan
    return float(np.corrcoef(lp, lt)[0, 1])


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = df.groupby(by, observed=True)
    q_agg = grouped["q_error"].agg(
        mean_q="mean",
        median_q="median",
        p90_q=lambda s: np.percentile(s, 90),
        p95_q=lambda s: np.percentile(s, 95),
        p99_q=lambda s: np.percentile(s, 99),
        max_q="max",
        count="count",
    )
    zero_rate = grouped["prediction"].agg(
        zero_pred_rate=lambda s: float((s < 1).mean())
    )
    corr = grouped.apply(
        lambda g: _log_pearson(g["prediction"].to_numpy(), g["true_value"].to_numpy()),
        include_groups=False,
    ).rename("log_pearson")
    return q_agg.join(zero_rate).join(corr).reset_index()


def compute_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "per_method_graph_queryset": aggregate(df, ["method", "graph", "query_dataset"]),
        "per_method_graph": aggregate(df, ["method", "graph"]),
        "per_method": aggregate(df, ["method"]),
    }


def assign_cardinality_bin(true_values: pd.Series) -> pd.Series:
    return pd.cut(
        true_values,
        bins=CARDINALITY_BIN_EDGES,
        labels=CARDINALITY_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
