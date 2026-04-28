# FICE C++ Inference Pipeline

This directory contains everything needed to run FICE cardinality estimation from C++ via libtorch. The pipeline separates FICE into two stages:

1. **Preparation (Python):** Load a trained checkpoint, compute entity/relation embeddings for the entire knowledge graph, and export the query GNN to TorchScript.
2. **Inference (C++):** Load the exported model and precomputed embeddings, parse a SPARQL BGP query, and predict its cardinality -- all without Python.

---

## Prerequisites

- A trained FICE checkpoint (`.pt` file from `scripts/train.py`)
- The corresponding dataset available in `data/` (e.g., `fb237_v4`)
- Python environment with the `fice` package installed (`uv sync` from the repo root)
- libtorch (bundled with PyTorch -- no separate install needed)

### Finding your libtorch path

libtorch ships with every PyTorch installation. To find the CMake prefix path:

```bash
uv run python -c "import torch; print(torch.utils.cmake_prefix_path)"
# Example output: /home/user/project/.venv/lib/python3.12/site-packages/torch/share/cmake
```

You will pass this path to CMake when building the C++ binary.

---

## Quick Start

### 1. Fill out the config

Edit `inference/inference_config.yaml`:

```yaml
# Path to a trained FICE checkpoint
checkpoint: /path/to/output/run_xxx/checkpoints/best_model.pt

# Dataset name (must match a key in DATASET_REGISTRY)
graph_dataset: fb237_v4

# Directory where inference artifacts will be saved
output_dir: /path/to/output/inference/

# Batch size for embedding computation (0 = full-graph mode, recommended for <500K nodes)
batch_size: 0

# Device for embedding computation
device: cpu
```

The checkpoint must have been trained with `Conditioning: "false"` (the default). Model hyperparameters are read from the checkpoint itself -- no need to duplicate them.

### 2. Generate embeddings and export the model

```bash
cd scripts
uv run python ../inference/prepare.py --config ../inference/inference_config.yaml
```

This produces the following files in `output_dir`:

| File | Contents |
|------|----------|
| `entity_embeddings.bin` | Float32 array, shape (num_entities, embedding_dim) |
| `relation_embeddings.bin` | Float32 array, shape (num_relations, embedding_dim) |
| `entity_occurrences.bin` | Float32 array, shape (num_entities, 3) |
| `relation_occurrences.bin` | Float32 array, shape (num_relations, 3) |
| `uri_to_id.json` | Entity/relation URI to integer ID mappings |
| `metadata.json` | Dimensions, graph stats, model config |
| `query_gnn_scripted.pt` | TorchScript model (via `torch.jit.script`) |
| `query_gnn_traced.pt` | TorchScript model (via `torch.jit.trace`) |

### 3. Build the C++ binary

```bash
cd inference
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(uv run python -c "import torch; print(torch.utils.cmake_prefix_path)")
make
```

### 4. Run a query

```bash
./fice_inference /path/to/output/inference/ "?x <http://example.org/some_relation> <http://example.org/some_entity> . ?x <http://example.org/other_relation> ?y"
```

The query format is a simplified SPARQL BGP (Basic Graph Pattern):
- Variables start with `?`
- URIs are enclosed in `<>`
- Triple patterns are separated by ` . `

Output:
```
Parsed 2 triple patterns
  ?x http://example.org/some_relation http://example.org/some_entity
  ?x http://example.org/other_relation ?y

Result:
  Log-cardinality: 3.14159
  Estimated cardinality: 22.14
```

### 5. Run the comparison (Python vs C++)

Validates that the C++ pipeline produces identical predictions to the native Python model:

```bash
cd scripts
uv run python ../inference/compare.py \
    --config ../inference/inference_config.yaml \
    --queries ../data/fb237_v4_star_queries/raw/fb237_v4_star_queries \
    --cpp-binary ../inference/build/fice_inference \
    --output-dir ../output/comparison/
```

Options:
- `--max-queries N` -- limit to first N queries (useful for quick checks)
- `--warmup N` -- number of warmup queries before timing (default: 100)

### 6. Run the C++ benchmark

Measures per-query timing broken down by query graph construction and model inference, and produces a boxplot figure grouped by query size:

```bash
cd scripts
uv run python ../inference/benchmark_cpp.py \
    --queries ../data/fb237_v4_star_queries/raw/fb237_v4_star_queries \
    --cpp-binary ../inference/build/fice_inference \
    --artifacts-dir ../output/inference/ \
    --output-dir ../output/benchmark/
```

Produces `output/benchmark/runtime_by_query_size.png` and a TSV with raw per-query timings.

---

## File Reference

### `inference_config.yaml`

Configuration file pointing to a trained checkpoint and the target knowledge graph. All model hyperparameters (embedding dimension, number of layers, readout type, etc.) are read from the checkpoint -- this file only specifies *which* checkpoint and graph to use, plus runtime options like batch size and device.

### `prepare.py`

The preparation script that bridges Python training and C++ inference. It performs four main tasks:

1. **Model reconstruction:** Loads the checkpoint, reconstructs the full FICE model from the saved config, and loads the trained weights. Handles backward compatibility with older checkpoints that lack certain config fields (e.g., `role_aware_degrees` is inferred from the weight shapes).

2. **Embedding computation:** Runs the embedding GNN (BipartiteGraphGNN or EntityGNN + RelationGNN) on the entire knowledge graph to produce a fixed embedding vector for every entity and relation. Since `Conditioning` must be `"false"`, these embeddings are query-independent and only need to be computed once. Supports two modes: full-graph (loads everything into memory) and batched (iterates over node neighborhoods for large graphs).

3. **Artifact export:** Saves embeddings and occurrence counts as raw contiguous float32 arrays (`.bin` files) that can be memory-mapped in C++ with a simple pointer cast. Also saves URI-to-ID mappings and metadata as JSON.

4. **TorchScript export:** Creates a `ScriptableTripleModel` (see below), transfers weights from the trained PyG model, verifies numerical equivalence, and exports via both `torch.jit.script` and `torch.jit.trace`.

### `scriptable_query_gnn.py`

A pure-PyTorch reimplementation of the query GNN (TripleModel) that is compatible with TorchScript. This is necessary because PyG's `MessagePassing.propagate()` uses Python-specific dynamic dispatch that neither `torch.jit.script` nor `torch.jit.trace` can handle.

The reimplementation uses only standard PyTorch operations:
- **`ScriptableGINEConv`:** Reimplements GINEConv as `message = relu(x[src] + lin(edge_attr))`, aggregated via `scatter_add_` to destination nodes, followed by `MLP(agg + (1 + eps) * x)`.
- **`ScriptableAttentionPool`:** Reimplements attentional aggregation with a per-graph softmax using the log-sum-exp trick for numerical stability.
- **`ScriptableTripleModel`:** Stacks GINEConv layers with LayerNorm and optional skip connections, applies global pooling (sum or attention), appends `log1p(graph_size)`, and passes through a final MLP head.

The `transfer_weights()` function copies weights from a trained PyG TripleModel into the scriptable version, handling the key remapping (`convs.N.*` -> `blocks.N.conv.*`, `norms.N.*` -> `blocks.N.norm.*`).

### `example_inference.cpp`

The C++ inference binary. It operates in two modes:

**Single-query mode:** `./fice_inference <dir> "<bgp_query>"`
Loads the TorchScript model and precomputed artifacts, parses the BGP query, builds the query graph tensors, runs inference, and prints the cardinality estimate.

**Batch mode:** `./fice_inference <dir> --batch <queries.bgp>`
Reads one BGP query per line from a file, runs inference on each, and outputs a TSV with per-query timing breakdowns (graph construction, model inference, total).

The query graph construction in C++ mirrors what FICE does in Python:
- Each unique subject/object in the BGP gets a local node ID.
- Bound entities and relations have their precomputed embeddings and occurrence counts looked up from the memory-mapped `.bin` files.
- Variable nodes (`?x`, `?y`, ...) get zero embeddings and zero counts.
- Occurrence counts are `log1p`-transformed before being passed to the model.
- The query graph tensors (`x`, `edge_index`, `edge_attr`, `batch`, `graph_size`) are passed directly to the TorchScript model.

Includes a minimal JSON parser (sufficient for `metadata.json` and `uri_to_id.json`) and an RAII wrapper for memory-mapped files.

### `CMakeLists.txt`

Standard CMake build configuration. Links against libtorch (found via `CMAKE_PREFIX_PATH` pointing to your PyTorch installation). Requires C++17.

### `compare.py`

Validation script that runs all queries through three inference paths and compares predictions:

- **Path A (Python native):** Loads the full FICE model, precomputes embeddings once, then for each query builds a PyG query graph, injects embeddings, and runs only the query GNN. This is what an optimized Python deployment would do.
- **Path B (TorchScript in Python):** Loads the exported TorchScript model and the `.bin` artifacts, builds raw tensors replicating the C++ logic exactly. Verifies the TorchScript export is correct.
- **Path C (C++ binary):** Invokes the C++ binary in batch mode via subprocess. Verifies the full C++ pipeline end-to-end.

Reports prediction agreement (MAE, max absolute error, correlation), Q-error against ground truth, and per-query timing with speedup ratios. Writes a per-query TSV for further analysis.

### `benchmark_cpp.py`

Benchmarking script focused on C++ runtime analysis. Runs all queries through the C++ binary in batch mode with a configurable warmup period, then produces:

- Summary statistics for graph construction time, model inference time, and total time.
- A per-query-size breakdown table.
- A matplotlib figure with boxplots showing runtime distributions for each query size (number of triple patterns).

---

## Limitations

- **Query GNN:** Only `TripleModel` is supported. Other query GNNs (`QueryCardinalityGNN`, `QueryTransformer`, etc.) would each need their own scriptable reimplementation.
- **Embedding space:** Only `euclidean`. Box, hyperbolic, and product spaces would need geometry operations added to the scriptable model.
- **Conditioning:** Must be `"false"`. Conditioning makes embeddings query-dependent, which defeats the purpose of precomputation.
- **Traced model:** The traced export (`query_gnn_traced.pt`) bakes in `batch_size=1` (single query). Use the scripted export for batched inference.
