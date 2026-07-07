#!/bin/bash
#SBATCH -J rank_ds
#SBATCH -t 24:00:00
#SBATCH -n 32

# ============================================================
# Chain-sampled ranking dataset builder for one KG.
#
# For each shape with an existing raw query file, expand each
# parent BGP into chain-sampled sub-BGPs (siblings + extensions)
# and call QLever for the true cardinality of each sub-BGP.
#
# Output: data/${DATASET}_${SHAPE}_queries_ranking/raw/
#   - queries.jsonl  (resumable stream, one record per parent)
#   - queries.json   (final array, repacked at end of each shape)
#
# Usage:
#   DATASET=fb237_v4 sbatch scripts/cluster/build_ranking_dataset.sbatch
#
# Or in bulk: scripts/cluster/submit_all_ranking_jobs.sh
# ============================================================

set -u

# ---- Required env var ----
DATASET="${DATASET:?must set DATASET env var, e.g. DATASET=fb237_v4 sbatch ...}"

# ---- Tunables ----
MAX_QUERIES="${MAX_QUERIES:-3000}"
MAX_SIBLINGS_PER_STEP="${MAX_SIBLINGS_PER_STEP:-6}"
WORKERS="${WORKERS:-16}"
TIMEOUT="${TIMEOUT:-180}"
PORT="${PORT:-$((7000 + SLURM_JOB_ID % 1000))}"
QLEVER_MEMORY="${QLEVER_MEMORY:-64G}"
QLEVER_TIMEOUT="${QLEVER_TIMEOUT:-300s}"

# ---- Fixed paths ----
PROJECT_DIR="/storage/home/tschwabe/fully-inductive-cardinality-estimation"
QLEVER_SIF="/storage/home/tschwabe/qlever.sif"
INDEX_DIR="${PROJECT_DIR}/data/${DATASET}/qlever"

# ---- Sanity checks ----
if [ ! -d "$INDEX_DIR" ]; then
    echo "ERROR: QLever index dir not found: $INDEX_DIR"
    exit 1
fi
if [ ! -f "${INDEX_DIR}/${DATASET}.index.pso" ]; then
    echo "ERROR: QLever index files missing in $INDEX_DIR (expected ${DATASET}.index.*)"
    ls -la "$INDEX_DIR" | head
    exit 1
fi

cd "$PROJECT_DIR"

echo "================================================"
echo "Ranking dataset build for: $DATASET"
echo "Max queries/shape: $MAX_QUERIES"
echo "Max siblings/step: $MAX_SIBLINGS_PER_STEP"
echo "Workers: $WORKERS"
echo "QLever port: $PORT  mem: $QLEVER_MEMORY  timeout: $QLEVER_TIMEOUT"
echo "Started: $(date)"
echo "================================================"

# ---- Start QLever server ----
echo ""
echo "=== Starting QLever server ==="
LOG_DIR="${PROJECT_DIR}/data/${DATASET}/qlever"
apptainer exec "$QLEVER_SIF" \
    /qlever/qlever-server -i "${INDEX_DIR}/${DATASET}" -p "$PORT" -j 16 \
    -m "$QLEVER_MEMORY" -s "$QLEVER_TIMEOUT" \
    > "${LOG_DIR}/server_ranking.log" 2>&1 &
SERVER_PID=$!
echo "QLever server PID: $SERVER_PID"

cleanup() {
    echo "Stopping QLever server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait for server (up to 5 min)
SERVER_READY=0
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
        SERVER_READY=1
        echo "QLever ready on port $PORT (attempt $i)"
        break
    fi
    sleep 5
done
if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: QLever server did not start within 5 min"
    echo "--- server log tail ---"
    tail -50 "${LOG_DIR}/server_ranking.log"
    exit 1
fi

# ---- Resolve raw query file per shape ----
# Convention: data/${DATASET}_${SHAPE}_queries/raw/queries.json,
#   else hash-named single file, else skip.
raw_for() {
    local shape="$1"
    local rawdir="${PROJECT_DIR}/data/${DATASET}_${shape}_queries/raw"
    if [ -f "${rawdir}/queries.json" ]; then
        echo "${rawdir}/queries.json"
        return 0
    fi
    # Look for any single regular file (hash-named, e.g. 59651327)
    local cand
    cand=$(ls -1 "$rawdir" 2>/dev/null | head -1)
    if [ -n "$cand" ] && [ -f "${rawdir}/${cand}" ]; then
        echo "${rawdir}/${cand}"
        return 0
    fi
    return 1
}

# ---- Loop over shapes ----
SHAPES=(star path cycle diamond flower snowflake path_star_ends binary_tree)

for shape in "${SHAPES[@]}"; do
    echo ""
    echo "============================================"
    echo "==== Shape: $shape  ($(date +%H:%M:%S))"
    echo "============================================"

    if ! RAW=$(raw_for "$shape"); then
        echo "SKIP $DATASET/$shape — no raw query file in data/${DATASET}_${shape}_queries/raw/"
        continue
    fi
    OUT_DIR="${PROJECT_DIR}/data/${DATASET}_${shape}_queries_ranking/raw"
    mkdir -p "$OUT_DIR"

    echo "raw:    $RAW"
    echo "out:    $OUT_DIR"

    python3 "${PROJECT_DIR}/inference/build_ranking_dataset.py" \
        --raw "$RAW" \
        --out "$OUT_DIR" \
        --qlever-url "http://localhost:${PORT}/" \
        --max-queries "$MAX_QUERIES" \
        --max-siblings-per-step "$MAX_SIBLINGS_PER_STEP" \
        --workers "$WORKERS" \
        --timeout "$TIMEOUT" 2>&1
    rc=$?
    echo "exit code: $rc  ($(date +%H:%M:%S))"
done

echo ""
echo "================================================"
echo "Done $DATASET: $(date)"
echo "================================================"
