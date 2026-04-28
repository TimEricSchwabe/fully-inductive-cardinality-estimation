"""LMKG adapter: reads all_predictions.csv, splits dataset_name='mega_no_{graph}_{qtype}'."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..naming import canonicalize_graph, canonicalize_query_type
from ..schema import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)

_PREFIX = "mega_no_"
_ALLOWED_QTYPES = {"path", "star"}


def _split_dataset_name(name: str) -> tuple[str, str] | None:
    if not name.startswith(_PREFIX):
        return None
    stripped = name[len(_PREFIX) :]
    graph, sep, qtype = stripped.rpartition("_")
    if not sep or qtype not in _ALLOWED_QTYPES:
        return None
    return graph, qtype


class LmkgAdapter:
    name = "lmkg"

    def load(self, root: Path) -> pd.DataFrame:
        csv_path = root / "all_predictions.csv"
        if not csv_path.exists():
            logger.warning("Skipping LMKG (no %s)", csv_path)
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        df = pd.read_csv(
            csv_path, usecols=["dataset_name", "prediction", "true_value"]
        )
        split = df["dataset_name"].astype(str).map(_split_dataset_name)
        valid = split.notna()
        if not valid.all():
            bad = df.loc[~valid, "dataset_name"].unique()
            logger.warning("LMKG: dropping %d rows with unrecognized dataset_name(s): %s", (~valid).sum(), bad[:5])
            df = df.loc[valid].copy()
            split = split.loc[valid]
        df["graph"] = split.map(lambda t: canonicalize_graph(t[0]))
        df["query_type"] = split.map(lambda t: canonicalize_query_type(t[1]))
        df["method"] = self.name
        df["query_dataset"] = df["graph"] + "-" + df["query_type"]
        return df[CANONICAL_COLUMNS].reset_index(drop=True)
