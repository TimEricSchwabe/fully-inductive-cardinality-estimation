#!/usr/bin/env bash
#SBATCH -J lss_benchmark
#!SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1        # request 1 GPU
#SBATCH -t 24:00:00                # walltime (hh:mm:ss)
#SBATCH -o ~/fully-inductive-cardinality-estimation/benchmarks/lss/results/lss_stdout_%j.log
#SBATCH -e ~/fully-inductive-cardinality-estimation/benchmarks/lss/results/lss_stderr_%j.log

set -euo pipefail

SCRIPT_DIR=~/fully-inductive-cardinality-estimation/benchmarks/lss
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
RUNNER="$SCRIPT_DIR/run_all_experiments.py"

if [[ ! -x "$VENV_PYTHON" ]]; then
	echo "Python executable not found in venv: $VENV_PYTHON"
	exit 1
fi

cd "$SCRIPT_DIR"
srun "$VENV_PYTHON" "$RUNNER" --epochs 10

