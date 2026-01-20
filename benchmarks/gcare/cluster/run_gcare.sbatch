#!/bin/bash
#SBATCH -J gcare_bench
#SBATCH -w gpu-intel-pvc
#SBATCH -t 12:00:00                 # 12 hour walltime

# -------------------------------
# Variables
# -------------------------------
PROJECT_DIR="/storage/home/lukasketzer/fully-inductive-cardinality-estimation"
CONTAINER="/storage/home/lukasketzer/gcare.sif"
SCRIPT="$PROJECT_DIR/benchmarks/gcare/run_all_experiments_no_docker.py"
CONFIG="$PROJECT_DIR/benchmarks/gcare/experiments_config.yaml"

# -------------------------------
# Setup
# -------------------------------
echo "================================"
echo "G-Care Cardinality Estimation"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Config: $CONFIG"
echo "================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/benchmarks/gcare/logs"

# -------------------------------
# Load modules (if needed)
# -------------------------------
# module load cmake/3.20.0
# module load boost/1.76.0

# -------------------------------
# Run Experiment
# -------------------------------
echo "Starting experiments at $(date)"
echo ""

srun apptainer exec --cleanenv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    python3 "$SCRIPT" \
        --config "$CONFIG" \
        --workspace-root "$PROJECT_DIR"

EXIT_CODE=$?

echo ""
echo "Finished at $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment completed successfully"
else
    echo "✗ Experiment failed"
fi

exit $EXIT_CODE
