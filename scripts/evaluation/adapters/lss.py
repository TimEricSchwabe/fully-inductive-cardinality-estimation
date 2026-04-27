"""LSS adapter: reads loco_<graph>/predictions.csv (one folder per held-out KG)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..naming import canonicalize_graph, canonicalize_query_type
from ..schema import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)


class LssAdapter:
    name = "lss"

    def load(self, root: Path) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("loco_")):
            csv_path = run_dir / "predictions.csv"
            if not csv_path.exists():
                logger.warning("Skipping %s (no predictions.csv)", run_dir)
                continue
            df = pd.read_csv(
                csv_path,
                usecols=["dataset", "query_type", "prediction", "ground_truth"],
            )
            df = df.rename(columns={"ground_truth": "true_value"})
            df["method"] = self.name
            df["graph"] = df["dataset"].astype(str).map(canonicalize_graph)
            df["query_type"] = df["query_type"].astype(str).map(canonicalize_query_type)
            df["query_dataset"] = df["graph"] + "-" + df["query_type"]
            frames.append(df[CANONICAL_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)
