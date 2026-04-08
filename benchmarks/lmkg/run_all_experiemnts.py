#!/usr/bin/env python3
"""
LOCO (Leave-One-Graph-Out) benchmark pipeline for LMKG-S.

For each dataset in DATASETS, trains on the combined mega-graph of all other
N-1 datasets (with globally distinct entity/relation IDs) and evaluates on
the held-out dataset.

Steps:
  1. Convert FICE data → LMKG-S per-dataset query files + stats
  2. Build LOCO mega-graph folds (merge N-1 datasets with global ID offsets)
  3. Train two LMKG-S models per fold: one for star eval, one for path eval
  4. Evaluate each model on the held-out dataset (star and path separately)
"""

import sys
from collections import defaultdict
from pathlib import Path

import yaml

BENCHMARKS_LMKG = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_LMKG.parent.parent
sys.path.insert(0, str(BENCHMARKS_LMKG))

import prepare_mega_graph as mega_graph
from helper import (
    convert_all_datasets,
    eval_model_path_only,
    eval_model_star_only,
    print_summary,
    train_model_path_only,
    train_model_star_only,
)

_CONFIG_PATH = PROJECT_ROOT / "benchmarks/experiments_config.yaml"
with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

# All datasets in stable order (for global ID assignment)
DATASETS = [d["name"] for d in _cfg["datasets"]]
# LOCO experiment folds to run
EXPERIMENTS = _cfg["experiments"]

def _to_list(value):
    """Normalize a string or list config value to a list."""
    return value if isinstance(value, list) else [value]


def _get_eval_datasets(experiment):
    """Datasets to evaluate on — must be explicitly set via the 'eval' key.

    Independent of 'left_out': a dataset listed here is NOT excluded from the
    training pool unless it also appears in 'left_out'.
    Returns an empty list when 'eval' is absent or null (no evaluation runs).
    """
    raw = experiment.get("eval")
    if not raw:
        return []
    return _to_list(raw)


SKIP_CONVERT = False
SKIP_PREPARE = False
EPOCHS = 100
BATCH_SIZE = 512
LAYERS = 2
NEURONS = 256
SEED = 42


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def step2_prepare():
    """Build LOCO mega-graph folds with globally distinct IDs.

    Uses all datasets for offset computation (stable global IDs), but only
    includes pool datasets in each fold's training data.
    """
    entity_offsets, relation_offsets, mega_d, mega_b = mega_graph.compute_offsets(DATASETS)
    print(f"mega_d={mega_d}, mega_b={mega_b}")
    for ds in DATASETS:
        print(f"  {ds}: entity_offset={entity_offsets[ds]}, relation_offset={relation_offsets[ds]}")

    print("\nRemapping query files...")
    all_train = {}
    all_eval = {}
    for ds in DATASETS:
        print(f"  {ds}...")
        all_train[ds] = mega_graph.remap_dataset_queries(ds, entity_offsets[ds], relation_offsets[ds])
        all_eval[ds] = mega_graph.read_eval_queries(ds, entity_offsets[ds], relation_offsets[ds])

    for exp in EXPERIMENTS:
        left_out_set = set(_to_list(exp["left_out"]))
        eval_list = _get_eval_datasets(exp)
        pool = exp["pool"]
        if pool == "*":
            pool_datasets = [ds for ds in DATASETS if ds not in left_out_set]
        else:
            pool_datasets = [ds for ds in pool if ds not in left_out_set]

        merged_train = {"star": defaultdict(list), "path": defaultdict(list)}
        for ds in pool_datasets:
            for qtype in ("star", "path"):
                for size, lines in all_train[ds][qtype].items():
                    merged_train[qtype][size].extend(lines)

        for eval_ds in eval_list:
            print(f"\n=== Fold: mega_no_{eval_ds} (pool={pool_datasets}) ===")
            mega_graph.write_star_path_folds(
                eval_ds=eval_ds,
                train_queries=merged_train,
                eval_queries=all_eval[eval_ds],
                mega_d=mega_d,
                mega_b=mega_b,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    errors = []

    # ------------------------------------------------------------------
    # Step 1: Convert FICE data to LMKG-S format
    # ------------------------------------------------------------------
    if not SKIP_CONVERT:
        print("\n" + "=" * 60)
        print("STEP 1: Convert FICE data → LMKG-S format")
        print("=" * 60)
        convert_all_datasets(SEED)
    else:
        print("Skipping step 1")

    # ------------------------------------------------------------------
    # Step 2: Prepare LOCO mega-graph datasets
    # ------------------------------------------------------------------
    if not SKIP_PREPARE:
        print("\n" + "=" * 60)
        print("STEP 2: Prepare LOCO mega-graph datasets")
        print("=" * 60)
        step2_prepare()
    else:
        print("Skipping step 2")

    # ------------------------------------------------------------------
    # Step 3: Train (one model per fold, all types + sizes)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Train models (star + path variants per fold, alltypessizes)")
    print("=" * 60)

    for exp in EXPERIMENTS:
        for eval_ds in _get_eval_datasets(exp):
            fold_name = f"mega_no_{eval_ds}"
            print(f"\n--- Training fold: {fold_name}_star (alltypessizes) ---")
            train_model_star_only(fold_name, errors,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, layers=LAYERS, neurons=NEURONS)
            print(f"\n--- Training fold: {fold_name}_path (alltypessizes) ---")
            train_model_path_only(fold_name, errors,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, layers=LAYERS, neurons=NEURONS)

    # ------------------------------------------------------------------
    # Step 4: Evaluate on held-out datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Evaluate models on held-out datasets")
    print("=" * 60)

    for exp in EXPERIMENTS:
        for eval_ds in _get_eval_datasets(exp):
            fold_name = f"mega_no_{eval_ds}"
            print(f"\n--- Evaluating fold: {fold_name}_star (held-out: {eval_ds}, star queries) ---")
            eval_model_star_only(fold_name, errors, layers=LAYERS, neurons=NEURONS)
            print(f"\n--- Evaluating fold: {fold_name}_path (held-out: {eval_ds}, path queries) ---")
            eval_model_path_only(fold_name, errors, layers=LAYERS, neurons=NEURONS)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(errors)


if __name__ == "__main__":
    main()
