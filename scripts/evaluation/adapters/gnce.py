"""GNCE adapter: reads test_predictions/{preds,gts,sizes}.npy + dataset_names.json."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..naming import canonicalize_graph
from ..schema import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"^loo_(?P<short>.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


class GnceAdapter:
    name = "gnce"

    def load(self, root: Path) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            m = _RUN_RE.match(run_dir.name)
            if not m:
                logger.warning("Skipping GNCE folder %s (unrecognized name)", run_dir.name)
                continue
            pred_dir = run_dir / "test_predictions"
            if not (pred_dir / "preds.npy").exists():
                logger.warning("Skipping %s (no test_predictions/preds.npy)", run_dir)
                continue
            frames.append(self._load_run(pred_dir, m.group("short")))
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _load_run(self, pred_dir: Path, shortname: str) -> pd.DataFrame:
        preds = np.load(pred_dir / "preds.npy")
        gts = np.load(pred_dir / "gts.npy")
        with (pred_dir / "dataset_names.json").open() as fh:
            names = json.load(fh)
        assert len(preds) == len(gts) == len(names), (
            f"Length mismatch in {pred_dir}: preds={len(preds)}, gts={len(gts)}, names={len(names)}"
        )
        graph = canonicalize_graph(shortname)
        query_type = [n.rsplit("-", 1)[-1] for n in names]
        df = pd.DataFrame(
            {
                "method": self.name,
                "graph": graph,
                "query_type": query_type,
                "prediction": preds,
                "true_value": gts,
            }
        )
        df["query_dataset"] = df["graph"] + "-" + df["query_type"]
        return df[CANONICAL_COLUMNS]
