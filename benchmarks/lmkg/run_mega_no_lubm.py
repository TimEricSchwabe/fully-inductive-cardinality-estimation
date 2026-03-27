#!/usr/bin/env python3
"""
Run LMKG-S for the mega_no_lubm fold only.

Assumes step 1 (fice_to_lmkg) and step 2 (prepare_mega_graph) have already
been run (or set SKIP_CONVERT/SKIP_PREPARE = False to re-run them).

Steps:
  1. (Optional) Convert FICE data → LMKG-S format
  2. (Optional) Build mega_no_lubm_star and mega_no_lubm_path folds
  3. Train two models (alltypessizes): one for star eval, one for path eval
  4. Evaluate each model on held-out lubm (star and path separately)
"""

import sys
from collections import defaultdict
from pathlib import Path

BENCHMARKS_LMKG = Path(__file__).resolve().parent
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

ALL_DATASETS = [
    "lubm",
    "swdf",
    "wikidata",
    "dbpedia100k",
    "fb237_v4",
    "fb237_v4_ind",
    "wn18rr_v4",
    "wn18rr_v4_ind",
]

EVAL_DS = "lubm"
FOLD_NAME = f"mega_no_{EVAL_DS}"

SKIP_CONVERT = False
SKIP_PREPARE = False
EPOCHS = 100
BATCH_SIZE = 512
LAYERS = 2
NEURONS = 256
SEED = 42


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
    # Step 2: Prepare mega_no_lubm fold
    # ------------------------------------------------------------------
    if not SKIP_PREPARE:
        print("\n" + "=" * 60)
        print(f"STEP 2: Prepare {FOLD_NAME} fold")
        print("=" * 60)
        entity_offsets, relation_offsets, mega_d, mega_b = mega_graph.compute_offsets(ALL_DATASETS)
        print(f"mega_d={mega_d}, mega_b={mega_b}")

        print("\nRemapping query files...")
        all_train = {}
        all_eval = {}
        for ds in ALL_DATASETS:
            print(f"  {ds}...")
            all_train[ds] = mega_graph.remap_dataset_queries(ds, entity_offsets[ds], relation_offsets[ds])
            all_eval[ds] = mega_graph.read_eval_queries(ds, entity_offsets[ds], relation_offsets[ds])

        print(f"\n=== Building fold: {FOLD_NAME} ===")
        merged_train = {"star": defaultdict(list), "path": defaultdict(list)}
        for ds in ALL_DATASETS:
            if ds == EVAL_DS:
                continue
            for qtype in ("star", "path"):
                for size, lines in all_train[ds][qtype].items():
                    merged_train[qtype][size].extend(lines)

        mega_graph.write_star_path_folds(
            eval_ds=EVAL_DS,
            train_queries=merged_train,
            eval_queries=all_eval[EVAL_DS],
            mega_d=mega_d,
            mega_b=mega_b,
        )
    else:
        print("Skipping step 2")

    # ------------------------------------------------------------------
    # Step 3: Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3: Train models for {FOLD_NAME} (star + path variants, alltypessizes)")
    print("=" * 60)

    print(f"\n--- Training {FOLD_NAME}_star ---")
    train_model_star_only(FOLD_NAME, errors,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, layers=LAYERS, neurons=NEURONS)
    print(f"\n--- Training {FOLD_NAME}_path ---")
    train_model_path_only(FOLD_NAME, errors,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, layers=LAYERS, neurons=NEURONS)

    # ------------------------------------------------------------------
    # Step 4: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 4: Evaluate {FOLD_NAME} on held-out {EVAL_DS} (star and path separately)")
    print("=" * 60)

    print(f"\n--- Evaluating {FOLD_NAME}_star ---")
    eval_model_star_only(FOLD_NAME, errors, layers=LAYERS, neurons=NEURONS)
    print(f"\n--- Evaluating {FOLD_NAME}_path ---")
    eval_model_path_only(FOLD_NAME, errors, layers=LAYERS, neurons=NEURONS)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(errors)


if __name__ == "__main__":
    main()
