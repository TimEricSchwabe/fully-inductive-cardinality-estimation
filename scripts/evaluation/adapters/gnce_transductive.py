"""GNCE transductive adapter.

For each (target, held-out run) in TARGET_TO_HELDOUT:
  1. Locate loo_{held-out-shortname}_* run folder.
  2. Load val_predictions/epoch_<max>/{preds,gts}.npy + dataset_names.json.
  3. Filter entries whose graph canonicalizes to target.
  4. Emit canonical rows.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..naming import canonicalize_graph, canonicalize_query_type
from ..schema import CANONICAL_COLUMNS
from ..transductive_mapping import TARGET_TO_HELDOUT

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"^loo_(?P<short>.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_EPOCH_RE = re.compile(r"^epoch_(\d+)$")
_KNOWN_SHORTNAMES = {
    "aids", "codex_l", "dbpedia100k", "fb237", "fb237_v4",
    "hetionet", "lubm", "swdf", "wikidata", "wn18rr", "wn18rr_v4", "yago",
}


class GnceTransductiveAdapter:
    name = "gnce"

    def load(self, root: Path) -> pd.DataFrame:
        heldout_to_run: dict[str, Path] = {}
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            m = _RUN_RE.match(run_dir.name)
            if not m:
                continue
            short = m.group("short")
            if short not in _KNOWN_SHORTNAMES:
                logger.info("Skipping non-singleton GNCE run %s (short=%s)", run_dir.name, short)
                continue
            heldout_to_run[canonicalize_graph(short)] = run_dir

        frames: list[pd.DataFrame] = []
        for target, heldout in TARGET_TO_HELDOUT.items():
            run_dir = heldout_to_run.get(heldout)
            if run_dir is None:
                logger.warning("No GNCE run with held-out=%s for target=%s", heldout, target)
                continue
            val_dir = run_dir / "val_predictions"
            if not val_dir.is_dir():
                logger.warning("No val_predictions/ in %s", run_dir)
                continue
            epoch_dirs = [d for d in val_dir.iterdir() if d.is_dir() and _EPOCH_RE.match(d.name)]
            if not epoch_dirs:
                logger.warning("No epoch_* dirs in %s", val_dir)
                continue
            last_epoch_dir = max(epoch_dirs, key=lambda d: int(_EPOCH_RE.match(d.name).group(1)))

            preds = np.load(last_epoch_dir / "preds.npy")
            gts = np.load(last_epoch_dir / "gts.npy")
            with (last_epoch_dir / "dataset_names.json").open() as fh:
                names = json.load(fh)
            assert len(preds) == len(gts) == len(names)

            graph_raw = np.array([n.rsplit("-", 1)[0] for n in names])
            graph_can = np.array([canonicalize_graph(g) for g in graph_raw])
            mask = graph_can == target
            if not mask.any():
                logger.warning("GNCE transductive: no rows for target=%s in %s", target, last_epoch_dir)
                continue

            qtype = [canonicalize_query_type(n.rsplit("-", 1)[-1]) for n, keep in zip(names, mask) if keep]
            df = pd.DataFrame(
                {
                    "method": self.name,
                    "graph": target,
                    "query_type": qtype,
                    "prediction": preds[mask],
                    "true_value": gts[mask],
                }
            )
            df["query_dataset"] = df["graph"] + "-" + df["query_type"]
            logger.info(
                "GNCE transductive: target=%s run=%s epoch=%s rows=%d",
                target, run_dir.name, last_epoch_dir.name, len(df),
            )
            frames.append(df[CANONICAL_COLUMNS])

        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)
