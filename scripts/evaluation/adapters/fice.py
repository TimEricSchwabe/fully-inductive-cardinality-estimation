"""FICE adapter.

Per run folder `<held_out>/`:
  1. Parse config.yaml → lowercase set of datasets marked `type: validation`
     (these are the held-out graphs for this leave-one-out run).
  2. Pass 1 (chunked): stream stage+epoch+prediction+true_value rows, keep
     only `stage=="test"` rows whose `dataset_name` is in the held-out set,
     and group their q-errors by epoch.
  3. Pick `best_epoch = argmin(median_q_error)`.
  4. Pass 2 (chunked): stream full columns, keep `stage=="test" & epoch==best_epoch & dataset_name∈held_out`.
  5. Canonicalize graph names and emit canonical rows.

Train-val datasets show up with stage=="test" rows too (their 20% val split)
but we exclude them: the evaluation target is the held-out graph(s) only.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..metrics import per_query_q_error
from ..naming import canonicalize_graph, canonicalize_query_type
from ..schema import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1_000_000

_PASS1_COLS = ["stage", "epoch", "dataset_name", "prediction", "true_value"]
_PASS2_COLS = [
    "stage",
    "epoch",
    "dataset_name",
    "query_set_name",
    "query_type",
    "prediction",
    "true_value",
]
_CAT_DTYPES = {
    "stage": "category",
    "dataset_name": "category",
    "query_set_name": "category",
    "query_type": "category",
}


def read_held_out(cfg_path: Path) -> set[str]:
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    return {
        str(entry["dataset"]).lower()
        for entry in cfg.get("data", [])
        if entry.get("type") == "validation"
    }


class FiceAdapter:
    name = "fice"

    def load(self, root: Path) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if "ablation" in run_dir.name:
                logger.info("Skipping FICE ablation run %s", run_dir.name)
                continue
            csv_path = run_dir / "logs" / "predictions.csv"
            cfg_path = run_dir / "config.yaml"
            if not csv_path.exists():
                logger.warning("Skipping %s (no predictions.csv)", run_dir)
                continue
            if not cfg_path.exists():
                logger.warning("Skipping %s (no config.yaml)", run_dir)
                continue
            held_out = read_held_out(cfg_path)
            if not held_out:
                logger.warning("Skipping %s (no type:validation datasets in config)", run_dir)
                continue
            df_run = self._load_run(csv_path, held_out)
            frames.append(df_run)
            del df_run
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _load_run(self, csv_path: Path, held_out: set[str]) -> pd.DataFrame:
        run_name = csv_path.parent.parent.name
        best_epoch, best_median = self._find_best_epoch(csv_path, held_out)
        logger.info(
            "FICE run=%s held_out=%s best_epoch=%d median_q=%.4f",
            run_name,
            sorted(held_out),
            best_epoch,
            best_median,
        )

        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=_PASS2_COLS,
            dtype={**_CAT_DTYPES, "epoch": "int32"},
            chunksize=CHUNK_SIZE,
        ):
            ds_lower = chunk["dataset_name"].astype(str).str.lower()
            mask = (
                (chunk["stage"] == "test")
                & (chunk["epoch"] == best_epoch)
                & ds_lower.isin(held_out)
            )
            if mask.any():
                parts.append(chunk.loc[mask].copy())
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        df["method"] = self.name
        df["graph"] = df["dataset_name"].astype(str).str.lower().map(canonicalize_graph)
        df["query_type"] = df["query_type"].astype(str).map(canonicalize_query_type)
        df["query_dataset"] = df["graph"] + "-" + df["query_type"]
        return df[CANONICAL_COLUMNS].copy()

    def _find_best_epoch(self, csv_path: Path, held_out: set[str]) -> tuple[int, float]:
        per_epoch: dict[int, list[np.ndarray]] = {}
        for chunk in pd.read_csv(
            csv_path,
            usecols=_PASS1_COLS,
            dtype={"stage": "category", "dataset_name": "category", "epoch": "int32"},
            chunksize=CHUNK_SIZE,
        ):
            ds_lower = chunk["dataset_name"].astype(str).str.lower()
            mask = (chunk["stage"] == "test") & ds_lower.isin(held_out)
            if not mask.any():
                continue
            rows = chunk.loc[mask]
            q = per_query_q_error(rows["prediction"].to_numpy(), rows["true_value"].to_numpy())
            for epoch, grp_idx in rows.groupby("epoch", observed=True).indices.items():
                per_epoch.setdefault(int(epoch), []).append(q[grp_idx])

        if not per_epoch:
            raise RuntimeError(
                f"No stage=='test' rows for held-out {held_out} in {csv_path}"
            )

        medians = {ep: float(np.median(np.concatenate(parts))) for ep, parts in per_epoch.items()}
        best_epoch = min(medians, key=medians.__getitem__)
        return best_epoch, medians[best_epoch]
