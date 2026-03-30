#!/bin/bash
#SBATCH -J query_gen
#SBATCH -t 24:00:00
#SBATCH -n 32

# ============================================================
# Query Generation Job
# Builds a QLever index, starts the server, and runs the
# rdf-subgraph-sampler to generate star, path, and complex queries.
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

# ---- Generation flags (set to true/false) ----
GENERATE_STAR=false
GENERATE_PATH=false
GENERATE_COMPLEX=true

# ---- User-configurable variables (edit per dataset) ----
DATASET_NAME="fb237_v4"
NT_FILE="/storage/home/tschwabe/fully-inductive-cardinality-estimation/data/fb237_v4/raw/59621618"

# Query configurations: Python list of (size, count) tuples
STAR_CONFIGS="[(2, 1000), (3, 1000), (4, 1000), (5, 1000), (6, 1000), (7, 1000), (8, 1000), (9, 1000), (10, 1000)]"
PATH_CONFIGS="[(2, 1000), (3, 1000), (4, 1000), (5, 1000), (6, 1000), (7, 1000), (8, 1000), (9, 1000), (10, 1000)]"
COMPLEX_N_QUERIES=5000  # Number of queries per complex shape

# Sampling parameters
P_PREDICATE="1.0"        # Star: predicate instantiation probability
P_EDGE="1.0"             # Path/Complex: edge instantiation probability
P_NODE="0.0"             # Path/Complex: intermediate node instantiation probability
P_START_END="0.3"        # Path: start/end node instantiation probability
SAMPLING_METHOD="dfs"    # Path: "dfs", "bfs", or "random_walk"

PORT=7004

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

# Complex shape output dirs (one per shape)
declare -A COMPLEX_SHAPES
COMPLEX_SHAPES=(
    ["cycle"]="?a B ?b, ?b B ?c, ?c B ?a"
    ["diamond"]="?a B ?b, ?a B ?c, ?b B ?d, ?c B ?d"
    ["flower"]="?a B ?l1, ?a B ?l2, ?a B ?b, ?b B ?m1, ?b B ?c, ?c B ?r1, ?c B ?r2"
    ["snowflake"]="?center B ?a, ?center B ?b, ?center B ?c, ?center B ?d, ?a B ?l1, ?a B ?l2, ?b B ?r1, ?b B ?r2"
    ["binary_tree"]="?r B ?a, ?r B ?b, ?a B ?c, ?a B ?d, ?b B ?e, ?b B ?f"
    ["path_star_ends"]="?s B ?o, ?s2 B ?o, ?o B ?o2, ?o2 B ?z1, ?o2 B ?z2"
)

echo "================================================"
echo "Query Generation for: $DATASET_NAME"
echo "NT file: $NT_FILE"
echo "Generate star: $GENERATE_STAR"
echo "Generate path: $GENERATE_PATH"
echo "Generate complex: $GENERATE_COMPLEX"
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
if [ "$GENERATE_STAR" = true ]; then
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

    python batch_sampler.py
    echo "Star query generation exit code: $?"

    # Move star query output files (filename format: {dataset}_SIZE_{n}_stars_...)
    mv ${DATASET_NAME}*stars*.json "$STAR_OUTPUT_DIR/" 2>/dev/null
    echo "Star query files moved to: $STAR_OUTPUT_DIR/"
    ls -la "$STAR_OUTPUT_DIR/"
else
    echo ""
    echo "=== Step 4: Skipping star queries (GENERATE_STAR=false) ==="
fi

# ---- Step 5: Generate path queries ----
if [ "$GENERATE_PATH" = true ]; then
    echo ""
    echo "=== Step 5: Generating path queries ==="
    mkdir -p "$PATH_OUTPUT_DIR"
    cd "$SAMPLER_DIR"

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
else
    echo ""
    echo "=== Step 5: Skipping path queries (GENERATE_PATH=false) ==="
fi

# ---- Step 6: Generate complex queries ----
if [ "$GENERATE_COMPLEX" = true ]; then
    echo ""
    echo "=== Step 6: Generating complex queries ==="

    for SHAPE_NAME in "${!COMPLEX_SHAPES[@]}"; do
        SHAPE_DEF="${COMPLEX_SHAPES[$SHAPE_NAME]}"
        COMPLEX_OUTPUT_DIR="$PROJECT_DIR/data/${DATASET_NAME}_${SHAPE_NAME}_queries/raw"
        mkdir -p "$COMPLEX_OUTPUT_DIR"
        cd "$SAMPLER_DIR"

        echo ""
        echo "--- Generating $SHAPE_NAME queries ($COMPLEX_N_QUERIES) ---"
        echo "    Shape: $SHAPE_DEF"

        cat > batch_sampler_config.py << PYEOF
ENDPOINT = "http://localhost:${PORT}"
SHAPE = "complex"
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
QUERY_SHAPE = "${SHAPE_DEF}"
QUERY_CONFIGURATIONS = [(1, ${COMPLEX_N_QUERIES})]
PYEOF

        python batch_sampler.py
        echo "$SHAPE_NAME query generation exit code: $?"

        # Move complex query output files
        mv ${DATASET_NAME}*${SHAPE_NAME}*.json "$COMPLEX_OUTPUT_DIR/" 2>/dev/null
        # Also move by SIZE pattern (batch_sampler uses {dataset}_SIZE_{n} naming)
        mv ${DATASET_NAME}_SIZE_*.json "$COMPLEX_OUTPUT_DIR/" 2>/dev/null
        echo "$SHAPE_NAME query files moved to: $COMPLEX_OUTPUT_DIR/"
        ls -la "$COMPLEX_OUTPUT_DIR/"
    done
else
    echo ""
    echo "=== Step 6: Skipping complex queries (GENERATE_COMPLEX=false) ==="
fi

# ---- Done ----
echo ""
echo "================================================"
echo "Query generation complete: $(date)"
echo "Star queries:   $STAR_OUTPUT_DIR/"
echo "Path queries:   $PATH_OUTPUT_DIR/"
if [ "$GENERATE_COMPLEX" = true ]; then
    for SHAPE_NAME in "${!COMPLEX_SHAPES[@]}"; do
        echo "Complex ($SHAPE_NAME): $PROJECT_DIR/data/${DATASET_NAME}_${SHAPE_NAME}_queries/raw/"
    done
fi
echo "================================================"
