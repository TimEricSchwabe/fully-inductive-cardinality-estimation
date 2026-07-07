"""GCARE adapter: reads {graph}_{qtype}/{method}/{preds,gts}.npy per folder."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..naming import GRAPH_NAME_MAP, canonicalize_graph, canonicalize_query_type
from ..schema import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)

_GRAPH_PREFIXES = sorted(GRAPH_NAME_MAP.keys(), key=len, reverse=True)


def _split_graph_qtype(folder_name: str) -> tuple[str, str] | None:
    for prefix in _GRAPH_PREFIXES:
        if folder_name == prefix or folder_name.startswith(prefix + "_"):
            return prefix, folder_name[len(prefix) + 1 :]
    return None


class GcareAdapter:
    def __init__(self, method: str) -> None:
        self.name = method

    def load(self, root: Path) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            split = _split_graph_qtype(folder.name)
            if split is None:
                logger.warning("Skipping GCARE folder %s (unknown graph prefix)", folder.name)
                continue
            method_dir = folder / self.name
            preds_path = method_dir / "preds.npy"
            gts_path = method_dir / "gts.npy"
            if not preds_path.exists() or not gts_path.exists():
                logger.warning("Skipping %s (missing %s preds/gts)", folder.name, self.name)
                continue
            graph_raw, qtype_raw = split
            preds = np.load(preds_path)
            gts = np.load(gts_path)
            assert len(preds) == len(gts), (
                f"Length mismatch in {method_dir}: preds={len(preds)}, gts={len(gts)}"
            )
            df = pd.DataFrame(
                {
                    "method": self.name,
                    "graph": canonicalize_graph(graph_raw),
                    "query_type": canonicalize_query_type(qtype_raw),
                    "prediction": preds,
                    "true_value": gts,
                }
            )
            df["query_dataset"] = df["graph"] + "-" + df["query_type"]
            frames.append(df[CANONICAL_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)
