"""FICE transductive adapter.

For each (target graph, held-out run) pair in TARGET_TO_HELDOUT:
  1. Locate the run whose config's type=validation graphs canonicalize to `held_out`.
  2. Stream logs/predictions.csv filtered to
        stage=="test" & dataset_name (canonicalized) == target.
  3. Keep only rows at max(epoch).
  4. Emit canonical rows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..naming import GRAPH_NAME_MAP, canonicalize_graph, canonicalize_query_type
from ..schema import CANONICAL_COLUMNS
from ..transductive_mapping import TARGET_TO_HELDOUT
from .fice import CHUNK_SIZE, _CAT_DTYPES, _PASS2_COLS, read_held_out

logger = logging.getLogger(__name__)


def _canonical_to_raws(canonical: str) -> set[str]:
    return {raw for raw, can in GRAPH_NAME_MAP.items() if can == canonical}


class FiceTransductiveAdapter:
    name = "fice"

    def load(self, root: Path) -> pd.DataFrame:
        heldout_to_run: dict[str, Path] = {}
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cfg_path = run_dir / "config.yaml"
            csv_path = run_dir / "logs" / "predictions.csv"
            if not cfg_path.exists() or not csv_path.exists():
                continue
            held_out_raw = read_held_out(cfg_path)
            if not held_out_raw:
                continue
            canonicals = {canonicalize_graph(h) for h in held_out_raw}
            if len(canonicals) != 1:
                logger.info("Skipping %s for transductive (held-out=%s, not singleton)", run_dir.name, sorted(canonicals))
                continue
            heldout_to_run[next(iter(canonicals))] = run_dir

        frames: list[pd.DataFrame] = []
        for target, heldout in TARGET_TO_HELDOUT.items():
            run_dir = heldout_to_run.get(heldout)
            if run_dir is None:
                logger.warning("No FICE run with held-out=%s for target=%s", heldout, target)
                continue
            target_raws = _canonical_to_raws(target)
            csv_path = run_dir / "logs" / "predictions.csv"
            df_run = self._load_target(csv_path, target_raws)
            if df_run.empty:
                logger.warning("FICE transductive: no rows for target=%s in run=%s", target, run_dir.name)
                continue
            max_epoch = int(df_run["epoch"].max())
            df_run = df_run.loc[df_run["epoch"] == max_epoch].copy()
            logger.info(
                "FICE transductive: target=%s run=%s max_epoch=%d rows=%d",
                target, run_dir.name, max_epoch, len(df_run),
            )
            df_run["method"] = self.name
            df_run["graph"] = df_run["dataset_name"].astype(str).str.lower().map(canonicalize_graph)
            df_run["query_type"] = df_run["query_type"].astype(str).map(canonicalize_query_type)
            df_run["query_dataset"] = df_run["graph"] + "-" + df_run["query_type"]
            frames.append(df_run[CANONICAL_COLUMNS].copy())

        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _load_target(csv_path: Path, target_raws: set[str]) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=_PASS2_COLS,
            dtype={**_CAT_DTYPES, "epoch": "int32"},
            chunksize=CHUNK_SIZE,
        ):
            ds_lower = chunk["dataset_name"].astype(str).str.lower()
            mask = (chunk["stage"] == "test") & ds_lower.isin(target_raws)
            if mask.any():
                parts.append(chunk.loc[mask].copy())
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_PASS2_COLS)
