#!/bin/bash
#SBATCH -J fice_training
#SBATCH --gres=gpu:nvidia:1        # request 1 GPU
#SBATCH -t 24:00:00                # walltime (hh:mm:ss)

# -------------------------------
# Variables
# -------------------------------
PROJECT_DIR="/home/to/user/fully-inductive-cardinality-estimation"
CONFIG_FILE="$PROJECT_DIR/configs/train_cluster.yaml"
SCRIPT="$PROJECT_DIR/scripts/train.py"
CONTAINER="/home/to/user/pytorch_2.8.0-cuda12.6-cudnn9-devel.sif"
VENV_PATH="$PROJECT_DIR/venv_apptainer/bin/activate"

# -------------------------------
# Parse arguments
# -------------------------------
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# -------------------------------
# Environment setup
# -------------------------------
module load cuda/12.6.3

# -------------------------------
# Run inside Apptainer with venv
# -------------------------------
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    srun apptainer exec --nv --cleanenv "$CONTAINER" \
        bash -c "source $VENV_PATH && python3 $SCRIPT --config $CONFIG_FILE --checkpoint $CHECKPOINT"
else
    echo "Starting training from scratch"
    srun apptainer exec --nv --cleanenv "$CONTAINER" \
        bash -c "source $VENV_PATH && python3 $SCRIPT --config $CONFIG_FILE"
fi

