#!/bin/bash
#SBATCH -J fice_training
#SBATCH --gres=gpu:nvidia:1          # request 1 GPU
#SBATCH -t 12:00:00           # walltime

module load cuda/12.6.3

srun python3 train.py