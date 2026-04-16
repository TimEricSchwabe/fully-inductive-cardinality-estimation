#!/usr/bin/env bash
# SLURM job script for running the G-Care benchmark on the COMA cluster.
#
# Usage:
#   sbatch gcare_slurm.sh
#
# Adjust the resource settings below to match what the cluster allows.

#SBATCH --job-name=gcare-benchmark
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

apptainer exec \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${SIF}" \
    python3 "${SCRIPT_DIR}/run_all_experiments.py" \
        --max-parallel "${SLURM_CPUS_PER_TASK:-4}" \
        "$@"

echo "G-Care benchmark finished at $(date)"
