# NeuSO Benchmark Setup

Complete pipeline for running NeuSO cardinality estimation benchmarks on RDF/TTL datasets.

## Overview

This directory contains scripts and notebooks to:
1. Transform TTL graph data and query JSON files to NeuSO format
2. Train NeuSO models for cardinality estimation
3. Run inference and evaluate performance
4. Generate results compatible with gcare benchmark format

## Directory Structure

```
benchmarks/neuso/
├── prepare_data.py          # Data transformation script
├── train_neuso.py           # Model training script
├── run_estimation.py        # Inference and evaluation script
├── Run_NeuSO.ipynb          # Jupyter notebook orchestration
├── helper.py                # Utility functions
├── transform_graph.py       # TTL to NeuSO graph transformation
├── transform_query.py       # Query JSON to NeuSO format
├── rdf_types.py             # RDF type URI mappings
├── NeuSO/                   # NeuSO source code
│   ├── config.yaml          # Training configuration
│   ├── train.py             # Main training script
│   ├── dataset/             # Generated datasets
│   └── saved/               # Trained model checkpoints
└── results/                 # Evaluation results (gcare format)
    └── {query_type}/
        └── {dataset}/
            └── neuso/
                ├── gts.npy
                ├── preds.npy
                ├── pred_times.npy
                ├── sizes.npy
                └── results.json
```

## Quick Start

### Option 1: Using Jupyter Notebook (Recommended)

1. Open [Run_NeuSO.ipynb](Run_NeuSO.ipynb)
2. Configure dataset paths in the configuration cell
3. Run all cells sequentially

The notebook will:
- Transform your data
- Train the NeuSO model
- Run inference
- Visualize results

### Option 2: Using Command Line Scripts

#### Step 1: Prepare Data

```bash
python prepare_data.py \
  --dataset-name swdf_star \
  --graph-path ../../data/swdf/raw/59320634 \
  --query-path ../../data/swdf_star_queries/raw/59320652
```

**Arguments:**
- `--dataset-name`: Name for the dataset (used for organization)
- `--graph-path`: Path to TTL graph file
- `--query-path`: Path to query JSON file
- `--rdf-type-uri`: (Optional) Custom rdf:type URI (auto-detected otherwise)

**Output:** Creates dataset in `NeuSO/dataset/{dataset_name}/` with:
- `data_graph/{dataset}.graph` - Data graph in NeuSO format
- `data_graph/{dataset}.emb.npy` - Random label embeddings
- `query_graph/query_*.graph` - Query graphs
- `filter/query_*.filter` - Cardinality filters
- `ccg/query_*.ccg` - Placeholder CCG files
- Mapping JSONs for entities, predicates, and labels

#### Step 2: Train Model

```bash
python train_neuso.py --dataset swdf_star
```

**Arguments:**
- `--dataset`: Dataset name (from prepare_data.py)
- `--mode`: 'train' or 'test' (default: train)

**Output:** Saves trained model to `NeuSO/saved/{dataset}_{timestamp}/`

**Note:** Training can take several hours depending on dataset size and configuration (default: 100 epochs).

#### Step 3: Run Inference

```bash
python run_estimation.py \
  --dataset swdf_star \
  --query-type star
```

**Arguments:**
- `--dataset`: Dataset name
- `--query-type`: Type of queries (for results organization)
- `--model-dir`: (Optional) Specific model directory (auto-detects latest)

**Output:** Results saved to `results/{query_type}/{dataset}/neuso/`:
- `gts.npy` - Ground truth cardinalities
- `preds.npy` - Predicted cardinalities
- `pred_times.npy` - Prediction times per query
- `sizes.npy` - Query sizes (number of triples)
- `results.json` - Aggregated metrics (Q-Error, MSE, MAE, timing)

## Configuration

### NeuSO Training Config

Edit `NeuSO/config.yaml` to modify training parameters:

```yaml
trainer:
  epochs: 100                    # Number of training epochs
  train_test_split: 0.8         # Train/test split ratio
  
optimizer:
  lr: 0.002                      # Learning rate
  weight_decay: 0.001            # Weight decay
  
data_loader:
  query_batch_size: 5            # Batch size for queries
  topology_emb: false            # Use embeddings (false = one-hot)
  topology_emb_dim: 128          # Embedding dimension
```

### Label Embeddings

The `prepare_data.py` script generates random label embeddings as placeholders. Set `topology_emb: false` in config to use one-hot encoding instead (recommended for initial runs).

To use proper embeddings later:
1. Generate embeddings using Node2Vec, TransE, or similar
2. Save as numpy array: shape `[num_labels, 128]`
3. Replace `NeuSO/dataset/{dataset}/data_graph/{dataset}.emb.npy`
4. Set `topology_emb: true` in config

## Results Format

Results are saved in the same format as gcare benchmarks for easy comparison:

```python
# Load results
import numpy as np
import json

gts = np.load("results/star/swdf_star/neuso/gts.npy")
preds = np.load("results/star/swdf_star/neuso/preds.npy")

with open("results/star/swdf_star/neuso/results.json") as f:
    metrics = json.load(f)
    
print(f"Median Q-Error: {metrics['median_q_error']}")
print(f"Mean Prediction Time: {metrics['mean_prediction_time_seconds']}s")
```

## Troubleshooting

### Common Issues

1. **"Dataset directory not found"**
   - Run `prepare_data.py` first to create the dataset

2. **"No model checkpoint found"**
   - Run `train_neuso.py` to train a model before inference

3. **Training takes too long**
   - Reduce `epochs` in `config.yaml` for testing
   - Use a smaller dataset

4. **Import errors during training**
   - The training script automatically changes directory to `NeuSO/`
   - Ensure all NeuSO dependencies are installed (see `NeuSO/requirements.txt`)

5. **"Config file not found"**
   - Ensure `NeuSO/config.yaml` exists
   - Run from the `benchmarks/neuso/` directory

## Supported Datasets

Pre-configured RDF type URIs:
- SWDF: `<http://ex.org/03>`
- LUBM: `<http://example.org/1>`
- Wikidata: `<http://www.wikidata.org/prop/direct/P31>`
- DBpedia: `<http://example.orggenre>`
- WN18RR: `<http://example.org_derivationally_related_form>`

For other datasets, specify `--rdf-type-uri` manually.

## Comparison with Other Methods

The results format is compatible with gcare benchmarks. To compare:

```python
# In the notebook or custom script
gcare_results = np.load("../gcare/results/star/swdf_star/wj/results.json")
neuso_results = np.load("results/star/swdf_star/neuso/results.json")

# Compare median Q-Error, timing, etc.
```

## Server Execution

For running on remote servers without Jupyter:

```bash
# SSH into server
ssh user@server

# Run pipeline
python prepare_data.py --dataset-name swdf_star --graph-path ... --query-path ...
python train_neuso.py --dataset swdf_star
python run_estimation.py --dataset swdf_star --query-type star

# Copy results back
scp -r results/ local_machine:/path/to/results/
```

## References

- NeuSO Paper: [Neural Optimizer for Subgraph Queries]
- Original Repository: [NeuSO/](NeuSO/)
- FICE Project: [../../README.md](../../README.md)
