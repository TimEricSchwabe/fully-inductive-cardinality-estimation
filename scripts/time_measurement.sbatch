#!/bin/bash
#SBATCH -J fice_time_measurement
#SBATCH -w gpu-nvidia-h100          # specify the node
#SBATCH --gres=gpu:nvidia:1         # request 1 GPU
#SBATCH -t 24:00:00                 # walltime (hh:mm:ss)


# -------------------------------
# Variables
# -------------------------------
PROJECT_DIR="/storage/home/lukasketzer/fully-inductive-cardinality-estimation"
CONFIG_FILE="$config_file"
SCRIPT="$PROJECT_DIR/scripts/time_measurement.py"
CONTAINER="/storage/home/lukasketzer/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif"
VENV_PATH="$PROJECT_DIR/venv_apptainer/bin/activate"
# -------------------------------
# Environment setup
# -------------------------------
module load cuda/12.6.3

# -------------------------------
# Run inside Apptainer with venv
# -------------------------------
echo "Starting time measurement"
srun apptainer exec --nv --cleanenv "$CONTAINER" \
    bash -c "source $VENV_PATH && python3 $SCRIPT"

