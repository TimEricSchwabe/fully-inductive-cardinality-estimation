#!/bin/bash
#SBATCH -J gcare_bench
#SBATCH --array=0-1                 # 2 experiments (wikidata_path, wikidata_star)
#SBATCH --cpus-per-task=8           # 8 CPU cores per job
#SBATCH --mem=32G                   # 32GB memory per job
#SBATCH -t 12:00:00                 # 12 hour walltime
#SBATCH -o logs/gcare_%A_%a.out     # Output log with array job ID and task ID
#SBATCH -e logs/gcare_%A_%a.err     # Error log with array job ID and task ID

# -------------------------------
# Variables
# -------------------------------
PROJECT_DIR="/storage/home/lukasketzer/fully-inductive-cardinality-estimation"
CONTAINER="/storage/home/lukasketzer/gcare.sif"
SCRIPT="$PROJECT_DIR/benchmarks/gcare/run_single_experiment.py"
CONFIG="$PROJECT_DIR/benchmarks/gcare/experiments_config.yaml"
RESULTS_BASE="$PROJECT_DIR/benchmarks/gcare/results"

# Experiment names (must match experiments_config.yaml)
EXPERIMENTS=(
    "wikidata_path"
    "wikidata_star"
)

# Get experiment name for this array task
EXPERIMENT_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="$RESULTS_BASE/${EXPERIMENT_NAME}_${TIMESTAMP}"

# -------------------------------
# Setup
# -------------------------------
echo "================================"
echo "G-Care Cardinality Estimation"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "Results: $RESULTS_DIR"
echo "================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/benchmarks/gcare/logs"

# Create results directory
mkdir -p "$RESULTS_DIR"

# -------------------------------
# Load modules (if needed)
# -------------------------------
# module load cmake/3.20.0
# module load boost/1.76.0

# -------------------------------
# Run Experiment
# -------------------------------
echo "Starting experiment at $(date)"
echo ""

srun apptainer exec --cleanenv \
    --bind "$PROJECT_DIR/data:$PROJECT_DIR/data:ro" \
    --bind "$PROJECT_DIR/benchmarks/gcare/gcare/data/queryset:/usr/src/app/gcare/data/queryset" \
    --bind "$PROJECT_DIR/benchmarks/gcare/gcare/data/dataset:/usr/src/app/gcare/data/dataset" \
    --bind "$RESULTS_DIR:/usr/src/app/results" \
    "$CONTAINER" \
    python3 "$SCRIPT" \
        --experiment "$EXPERIMENT_NAME" \
        --config "$CONFIG" \
        --results-dir "$RESULTS_DIR" \
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
