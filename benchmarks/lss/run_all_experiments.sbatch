#!/bin/bash
#SBATCH -J lss_experiments
#SBATCH -t 24:00:00

set -euo pipefail

# Run from project root so relative imports/paths work.
PROJECT_DIR="/storage/home/lukasketzer/fully-inductive-cardinality-estimation"
cd "$PROJECT_DIR"

# Optional: activate your environment if needed.
# source .venv/bin/activate

python3 benchmarks/lss/run_all_experiments.py
