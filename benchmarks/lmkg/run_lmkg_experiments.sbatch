#!/bin/bash
#SBATCH -J lmkg_training
#SBATCH --gres=gpu:nvidia:1         # request 1 GPU
#SBATCH -t 24:00:00                 # walltime (hh:mm:ss)


# It is often better to use absolute paths instead of ~/ 
VENV_PATH="$HOME/fully-inductive-cardinality-estimation/benchmarks/lmkg/.venv/bin/python"
SCRIPT_PATH="$HOME/fully-inductive-cardinality-estimation/benchmarks/lmkg/run_all_experiemnts.py"

srun $VENV_PATH $SCRIPT_PATH