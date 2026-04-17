#!/bin/bash
#SBATCH -J gcare_benchmark
#SBATCH -t 23:00:00                # walltime (hh:mm:ss) - increase if needed for full experiments
#SBATCH -c 16                      # request 16 CPU cores for parallel processing
#SBATCH --mem=64G                  # request 64GB RAM (adjust based on dataset size)

set -euo pipefail

# ---- Paths ------------------------------------------------------------------
# Use absolute paths to avoid SLURM temp directory issues
PROJECT_ROOT="/storage/home/lukasketzer/fully-inductive-cardinality-estimation"
SCRIPT_DIR="${PROJECT_ROOT}/benchmarks/gcare"
SIF="${SCRIPT_DIR}/gcare.sif"

# ---- Sanity checks ----------------------------------------------------------
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container image not found: ${SIF}"
    echo "Build it first with:  apptainer build ${SIF} ${SCRIPT_DIR}/gcare.def"
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs"

# ---- Optional: rebuild G-Care binaries inside the container ----------------
# Uncomment if the binaries are missing or you want a clean rebuild.
# apptainer exec "${SIF}" bash -c "
#     cd '${SCRIPT_DIR}/gcare' && rm -rf build && mkdir build && cd build && cmake .. && make -j\$(nproc)"

# ---- Run benchmark ----------------------------------------------------------
echo "Starting G-Care benchmark at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs available: ${SLURM_CPUS_PER_TASK:-16}"

PYTHON="python3"
MAX_PARALLEL="${SLURM_CPUS_PER_TASK:-16}"

apptainer exec \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${SIF}" \
    $PYTHON "${SCRIPT_DIR}/run_all_experiments.py" \
        --max-parallel "${MAX_PARALLEL}" \
        "$@"

echo "G-Care benchmark finished at $(date)"
