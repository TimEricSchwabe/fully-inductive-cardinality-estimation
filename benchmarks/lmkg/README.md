# LMKG Benchmark

We only evaluated on LMKG-S since LMKG-U uses unsupervised training which FICE does not do.

The benchmark uses a **LOCO (Leave-One-Graph-Out)** setup: for each dataset, a model is trained on the merged mega-graph of all other N-1 datasets and evaluated on the held-out dataset.

## Requirements

**Python 3.8 is required.** TensorFlow 2.13 does not support Python 3.9 or later.
Using a different Python version or mismatched package versions will cause errors at import or runtime.

Install dependencies from within `LMKG-Code/lmkg-s/`:

```bash
pip install -r LMKG-Code/lmkg-s/requirements.txt
```

> **macOS (Apple Silicon / Intel):** Replace `tensorflow==2.13.0` with `tensorflow-macos==2.13.0` in `requirements.txt` before installing.

Key pinned versions:

| Package | Version |
|---|---|
| Python | 3.8 |
| tensorflow / tensorflow-macos | 2.13.0 |
| keras | 2.13.1 |
| numpy | 1.24.3 |
| scikit-learn | 1.3.2 |
| scipy | 1.10.1 |

## Running All Experiments

Run from the project root:

```bash
python benchmarks/lmkg/run_all_experiemnts.py
```

This executes four steps automatically:

1. **Convert** — FICE query JSON files → LMKG-S format (star/path files per dataset)
2. **Prepare** — Build LOCO mega-graph folds with globally distinct entity/relation IDs
3. **Train** — One model per fold (star variant + path variant, `alltypessizes` mode)
4. **Evaluate** — Evaluate each model on the held-out dataset

Key constants at the top of `run_all_experiemnts.py` control behaviour:

| Constant | Default | Description |
|---|---|---|
| `DATASETS` | 8 KGs | Datasets to include in LOCO folds |
| `SKIP_CONVERT` | `False` | Skip step 1 if data already converted |
| `SKIP_PREPARE` | `False` | Skip step 2 if folds already built |
| `EPOCHS` | `100` | Training epochs |
| `BATCH_SIZE` | `512` | Batch size |
| `LAYERS` / `NEURONS` | `2` / `256` | Model architecture |

## File Overview

| File | Purpose |
|---|---|
| `run_all_experiemnts.py` | Main entry point — runs the full 4-step pipeline |
| `fice_to_lmkg.py` | Converts FICE RDF + query JSON → LMKG-S `star_N.txt` / `chain_N.txt` files and `statistics/{ds}_stats.json` |
| `prepare_mega_graph.py` | Builds LOCO mega-graph folds by merging N-1 datasets with global ID offsets |
| `run_lmkgs.py` | Thin wrapper around `LMKG-Code/lmkg-s/lmkgs.py` (training and eval) |
| `helper.py` | Shared utilities: dataset conversion, model training/eval helpers, summary printer |
| `experiments_config.yaml` | Dataset paths (graph + query files) for all 8 KGs |
| `run_mega_no_lubm.py` | Partial run: LOCO folds excluding LUBM only |
| `run_swdf_only.py` | Partial run: SWDF dataset only |
| `LMKG-Code/` | Original LMKG source code (lmkg-s subdirectory used) |

## Output

- Converted query files: `LMKG-Code/lmkg-s/final_datasets/generated_{dataset}/`
- Dataset statistics: `LMKG-Code/lmkg-s/statistics/{dataset}_stats.json`
- Trained models: `LMKG-Code/lmkg-s/models/`
- Evaluation results: `LMKG-Code/lmkg-s/results/`
