#!/bin/bash
#SBATCH -J fice_training
#SBATCH --gres=gpu:nvidia:1          # request 1 GPU
#SBATCH -t 12:00:00           # walltime

module load cuda/12.6.3
# source /storage/home/lukasketzer/fully-inductive-cardinality-estimation/venv/bin/activate
srun python3 /storage/home/lukasketzer/fully-inductive-cardinality-estimation/scripts/train.py --config /storage/home/lukasketzer/fully-inductive-cardinality-estimation/configs/train_cluster.json