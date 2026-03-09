#!/usr/bin/env python3
"""
Run LMKG-S on swdf as a standalone dataset (no mega-graph).

Steps:
  1. Convert FICE swdf data → LMKG-S query files + stats
  2. Train one model (alltypessizes)
  3. Evaluate on swdf
"""

import sys
from pathlib import Path

BENCHMARKS_LMKG = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_LMKG))

from helper import convert_all_datasets, eval_model_alltypessizes, print_summary, train_model_alltypessizes

DATASET = "swdf"
SKIP_CONVERT = False
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
        print(f"STEP 1: Convert FICE {DATASET} data → LMKG-S format")
        print("=" * 60)
        stats = convert_all_datasets(SEED)
        if DATASET not in stats:
            print(f"ERROR: No experiments found for dataset '{DATASET}' in config.")
            sys.exit(1)
    else:
        print("Skipping step 1")

    # ------------------------------------------------------------------
    # Step 2: Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 2: Train model for {DATASET} (alltypessizes)")
    print("=" * 60)

    train_model_alltypessizes(DATASET, errors,
                              epochs=EPOCHS, batch_size=BATCH_SIZE, layers=LAYERS, neurons=NEURONS)

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3: Evaluate model for {DATASET}")
    print("=" * 60)

    eval_model_alltypessizes(DATASET, errors, layers=LAYERS, neurons=NEURONS)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(errors)


if __name__ == "__main__":
    main()
