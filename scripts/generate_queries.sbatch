#!/bin/bash
#SBATCH -J query_gen
#SBATCH -t 24:00:00
#SBATCH -n 32

# ============================================================
# Query Generation Job
# Builds a QLever index, starts the server, and runs the
# rdf-subgraph-sampler to generate star and path queries.
#
# Usage: Edit the variables below, then:
#   sbatch scripts/generate_queries.sbatch
#
# One-time setup (on login node):
#   git clone https://github.com/DE-TUM/rdf-subgraph-sampler.git
#   salloc -t 01:00:00
#   apptainer pull qlever.sif docker://adfreiburg/qlever:latest
#   pip install tqdm requests  # in sampler venv
# ============================================================

# ---- User-configurable variables (edit per dataset) ----
DATASET_NAME="yago"
NT_FILE="/storage/home/tschwabe/fully-inductive-cardinality-estimation/data/yago/raw/Yago.nt"

# Query configurations: Python list of (size, count) tuples
STAR_CONFIGS="[(2, 1000), (3, 1000), (4, 1000), (5, 1000), (6, 1000), (7, 1000), (8, 1000), (9, 1000), (10, 1000)]"
PATH_CONFIGS="[(2, 1000), (3, 1000), (4, 1000), (5, 1000), (6, 1000), (7, 1000), (8, 1000), (9, 1000), (10, 1000)]"

# Sampling parameters
P_PREDICATE="1.0"        # Star: predicate instantiation probability
P_EDGE="1.0"             # Path: edge instantiation probability
P_NODE="0.0"             # Path: intermediate node instantiation probability
P_START_END="0.3"        # Path: start/end node instantiation probability
SAMPLING_METHOD="dfs"    # Path: "dfs", "bfs", or "random_walk"

PORT=7001

# QLever server settings
QLEVER_MEMORY="64G"       # RAM for QLever query processing (-m flag)
QLEVER_TIMEOUT="30s"    # Server-side query timeout (-s flag, needs unit: s/min/h)

# Sampler HTTP timeout (seconds) for cardinality requests to QLever.
# The defaults are too low (3s for path, 10s for star).
SAMPLER_TIMEOUT=30

# ---- Fixed paths (adjust to your cluster setup) ----
PROJECT_DIR="/storage/home/tschwabe/fully-inductive-cardinality-estimation"
SAMPLER_DIR="/storage/home/tschwabe/rdf-subgraph-sampler"
QLEVER_SIF="/storage/home/tschwabe/qlever.sif"
# Optional: activate a venv with tqdm/requests installed
VENV_PATH="/storage/home/tschwabe/rdf-subgraph-sampler/venv/bin/activate"  # e.g., "/storage/home/tschwabe/rdf-subgraph-sampler/venv/bin/activate"

# ---- Derived paths ----
INDEX_DIR="$PROJECT_DIR/data/${DATASET_NAME}/qlever"
STAR_OUTPUT_DIR="$PROJECT_DIR/data/${DATASET_NAME}_star_queries/raw"
PATH_OUTPUT_DIR="$PROJECT_DIR/data/${DATASET_NAME}_path_queries/raw"

echo "================================================"
echo "Query Generation for: $DATASET_NAME"
echo "NT file: $NT_FILE"
echo "Started: $(date)"
echo "================================================"

# Activate venv if specified
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Activated venv: $VENV_PATH"
fi

# ---- Step 1: Build QLever index ----
echo ""
echo "=== Step 1: Building QLever index ==="
mkdir -p "$INDEX_DIR"

cat "$NT_FILE" | apptainer exec "$QLEVER_SIF" \
    /qlever/qlever-index -i "$INDEX_DIR/$DATASET_NAME" -F nt -f - \
    2>&1 | tee "$INDEX_DIR/index-build.log"

if [ ${PIPESTATUS[1]} -ne 0 ]; then
    echo "ERROR: QLever index build failed"
    exit 1
fi
echo "Index built successfully."

# ---- Step 2: Start QLever server ----
echo ""
echo "=== Step 2: Starting QLever server on port $PORT ==="

apptainer exec "$QLEVER_SIF" \
    /qlever/qlever-server -i "$INDEX_DIR/$DATASET_NAME" -p "$PORT" -j 16 \
    -m "$QLEVER_MEMORY" -s "$QLEVER_TIMEOUT" \
    > "$INDEX_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "QLever server PID: $SERVER_PID"

# Cleanup handler
cleanup() {
    echo "Stopping QLever server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

# Wait for server to be ready (up to 5 minutes)
echo "Waiting for QLever server..."
SERVER_READY=0
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
        SERVER_READY=1
        echo "QLever server ready on port $PORT (attempt $i)"
        break
    fi
    sleep 5
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: QLever server did not start within 5 minutes"
    echo "--- Server log ---"
    cat "$INDEX_DIR/server.log"
    exit 1
fi

# ---- Step 3: Patch sampler timeouts ----
echo ""
echo "=== Step 3: Patching sampler timeouts to ${SAMPLER_TIMEOUT}s ==="
cd "$SAMPLER_DIR"

# Star sampler: CARDINALITY_TIMEOUT (default 10s)
sed -i "s/^CARDINALITY_TIMEOUT = .*/CARDINALITY_TIMEOUT = ${SAMPLER_TIMEOUT}/" \
    samplers/star_query_generator_in_memory.py

# Path sampler: FINAL_QUERY_TIMEOUT (default 3s)
sed -i "s/^FINAL_QUERY_TIMEOUT = .*/FINAL_QUERY_TIMEOUT = ${SAMPLER_TIMEOUT}/" \
    samplers/path_query_generator_in_memory.py

echo "Patched timeouts."

# ---- Step 4: Generate star queries ----
echo ""
echo "=== Step 4: Generating star queries ==="
mkdir -p "$STAR_OUTPUT_DIR"
cd "$SAMPLER_DIR"

cat > batch_sampler_config.py << PYEOF
ENDPOINT = "http://localhost:${PORT}"
SHAPE = "star"
DATASET = "${DATASET_NAME}"
USE_CACHE = True
IN_MEMORY = True
GRAPH_NAME = "${DATASET_NAME}"
RDF_FILE_PATH = "${NT_FILE}"
GET_CARDINALITY = True
P_PREDICATE = ${P_PREDICATE}
SAMPLING_METHOD = "${SAMPLING_METHOD}"
P_EDGE = ${P_EDGE}
P_NODE = ${P_NODE}
P_START_END = ${P_START_END}
QUERY_CONFIGURATIONS = ${STAR_CONFIGS}
PYEOF

#python batch_sampler.py
#echo "Star query generation exit code: $?"

# Move star query output files (filename format: {dataset}_SIZE_{n}_stars_...)
#mv ${DATASET_NAME}*stars*.json "$STAR_OUTPUT_DIR/" 2>/dev/null
#echo "Star query files moved to: $STAR_OUTPUT_DIR/"
#ls -la "$STAR_OUTPUT_DIR/"

# ---- Step 5: Generate path queries ----
echo ""
echo "=== Step 5: Generating path queries ==="
mkdir -p "$PATH_OUTPUT_DIR"

cat > batch_sampler_config.py << PYEOF
ENDPOINT = "http://localhost:${PORT}"
SHAPE = "path"
DATASET = "${DATASET_NAME}"
USE_CACHE = True
IN_MEMORY = True
GRAPH_NAME = "${DATASET_NAME}"
RDF_FILE_PATH = "${NT_FILE}"
GET_CARDINALITY = True
P_PREDICATE = ${P_PREDICATE}
SAMPLING_METHOD = "${SAMPLING_METHOD}"
P_EDGE = ${P_EDGE}
P_NODE = ${P_NODE}
P_START_END = ${P_START_END}
QUERY_CONFIGURATIONS = ${PATH_CONFIGS}
PYEOF

python batch_sampler.py
echo "Path query generation exit code: $?"

# Move path query output files (filename format: {dataset}_SIZE_{n}_path_...)
mv ${DATASET_NAME}*path*.json "$PATH_OUTPUT_DIR/" 2>/dev/null
echo "Path query files moved to: $PATH_OUTPUT_DIR/"
ls -la "$PATH_OUTPUT_DIR/"

# ---- Done ----
echo ""
echo "================================================"
echo "Query generation complete: $(date)"
echo "Star queries: $STAR_OUTPUT_DIR/"
echo "Path queries: $PATH_OUTPUT_DIR/"
echo "================================================"
