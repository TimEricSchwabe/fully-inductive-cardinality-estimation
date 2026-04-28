# FICE — Fully Inductive Cardinality Estimator

FICE estimates the cardinality of basic graph pattern (BGP) queries over RDF
knowledge graphs. The model is *fully inductive*: it generalizes to graphs and
relations not seen at training time, without retraining or finetuning. This
repository contains the model implementation, training and evaluation
pipelines, baseline adapters, and a C++ inference subsystem suitable for
integration into a query optimizer.

## Repository layout

```
fice/                       Core package: models, datasets, embedding spaces.
configs/                    YAML training configurations.
scripts/                    Entry points (training, evaluation, plotting).
  scripts/utils/            Configuration, data loading, logging utilities.
  scripts/evaluation/       Unified cross-baseline evaluation pipeline.
inference/                  TorchScript export and C++ inference (own README).
benchmarks/                 Adapters and runners for baseline estimators.
data/                       Default dataset root (auto-created on first load).
output/                     Default location for run artifacts and evaluation results.
```

## Installation

Python 3.12 or newer is required. The project uses
[`uv`](https://github.com/astral-sh/uv) but plain `pip` works too.

```sh
# with uv (recommended):
uv sync

# with pip:
pip install -e .
# if a dependency fails to build, retry with:
pip install -e . --no-build-isolation
```

The authoritative dependency list lives in `pyproject.toml`. Core stack:
PyTorch 2.8, torch-geometric 2.7, geoopt (hyperbolic embeddings), pydantic,
pyyaml, pandas, matplotlib.

## Datasets

### Pre-built artifacts (recommended)

Knowledge graph `.nt` files, query `.json` files, and pre-computed
ranking-query files for the datasets used in the paper are available at:

> Datasets: `<RAW_DATA_LINK_PLACEHOLDER>`

Download and unpack into `data/` so each dataset sits at
`data/<dataset_name>/raw/<file>.nt` and each query set at
`data/<query_set_name>/raw/<file>.json`. The loaders generate a `processed/`
sibling directory on first use; subsequent loads read from `processed/`.

### Folder layout the loaders expect

```
data/<dataset_name>/
  raw/<dataset>.nt
  processed/                # auto-generated
data/<query_set_name>/
  raw/<file>.json
  processed/                # auto-generated
```

### Knowledge-graph format (N-Triples)

Plain N-Triples: one triple per line, three space-separated tokens (subject,
predicate, object) terminated by a period. Subjects and predicates are IRIs in
angle brackets; objects are IRIs or literals. Malformed lines are rejected.
Example:

```
<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
```

On first load, the dataset class builds an entity-to-id map, a relation-to-id
map, occurrence counts per entity and relation, and a meta-graph over
relations (head-to-head, tail-to-tail, head-to-tail, tail-to-head edges).
These artifacts are cached in `processed/`.

### Query format (JSON)

A query file is a JSON array of objects. Each object has at minimum a
`triples` field and a target cardinality `y`:

```json
[
  {
    "triples": [
      ["?v1", "<http://example.org/type>", "<http://example.org/Person>"],
      ["?v1", "<http://example.org/knows>", "?v2"]
    ],
    "y": 42
  }
]
```

Variables are strings starting with `?`. Constants are full IRIs (or literals
in standard N-Triples form). `y` is the integer cardinality used as the
training target.

### Ranking query format (JSON)

The pairwise-margin ranking loss (`fice.datasets.NTRankingQueries`) expects a
parent BGP plus its sub-BGPs along a sampled prefix chain:

```json
[
  {
    "triples": [...],
    "y": 1234,
    "n_triples": 4,
    "subqueries": [
      {
        "triples": [...],
        "y": 17,
        "size": 1,
        "mask": 1,
        "prefix_mask": 0
      },
      {
        "triples": [...],
        "y": 89,
        "size": 2,
        "mask": 3,
        "prefix_mask": 1
      }
    ]
  }
]
```

`mask` is the bitmask over the parent's triples that selects this sub-BGP;
`prefix_mask` is the bitmask of the prefix from which this sub-BGP was
extended. Two sub-BGPs are *siblings* iff they share the same `prefix_mask`
and `size` — the ranking loss is restricted to sibling pairs.

### Building your own ranking dataset

`inference/build_ranking_dataset.py` constructs ranking files from a regular
query JSON. It contacts a SPARQL endpoint that supports `COUNT(*)` over the
target KG to obtain ground-truth cardinalities for each sampled sub-BGP.

```sh
python inference/build_ranking_dataset.py \
    --raw   data/<query_set>/raw/<file>.json \
    --out   data/<query_set>_ranking/raw \
    --endpoint-url http://<your-sparql-endpoint>/ \
    --max-queries 100 \
    --max-siblings-per-step 6 \
    --workers 8
```

Output is a streamed `queries.jsonl` (resumable) plus the final
`queries.json` consumed by `NTRankingQueries`.

## Training

```sh
python scripts/train.py --config configs/train_local.yaml
```

`--checkpoint <path>` resumes a previous run. The provided configs are:

| Config | Purpose |
|---|---|
| `configs/train_local.yaml` | Single-KG sanity check; small batch, few epochs. |
| `configs/train_full.yaml` | Multi-KG training over all eight query shapes. |
| `configs/train_cluster.yaml` | Full-scale multi-KG training (LUBM, SWDF, Wikidata, …). |
| `configs/train_ranking_inductive.yaml` | Ranking-loss training with inductive validation on a held-out KG. |
| `configs/train_ranking_test.yaml` | Minimal ranking example. |

The full config schema is defined in `scripts/utils/config.py`. The fields
most likely to be tuned:

- `model` — `FICE`, `FiceBinary`, or `FiceIndependent`.
- `embedding_space` — geometric space for entity / relation embeddings:
  `euclidean`, `box`, `gaussian`, `hyperbolic`, `hgcn`, or `product`
  (Euclidean × Torus).
- `embedding_dim`, `n_embedding_layers` — embedding capacity.
- `Conditioning` — how occurrence counts feed the GNN: `false`, `seed only`,
  `all`.
- `entity_gnn`, `relation_gnn`, `query_gnn` — message-passing block selection
  (GINE, GAT, DeepSet, GPS) plus skip / norm / dropout.
- `ranking_loss` — boolean; requires a ranking-format query file.
- `prediction_mode` — `log-cardinality` or `log-selectivity`.
- `epochs`, `learning_rate`, `wandb`.

Each run produces a timestamped directory under `output/`:

```
output/<run_name>/
  checkpoints/best.pt           best validation checkpoint
  logs/*.csv                    train/test metrics, per-graph and per-query-type breakdowns
  config.yaml                   snapshot of the resolved configuration
```

## Evaluation

The unified evaluation pipeline ingests prediction files from FICE and the
baselines, normalizes them through per-method adapters, and emits unified
prediction tables, aggregated metrics, and plots.

```sh
python scripts/evaluation/run.py \
    --fice-root  output/final/FICE \
    --gnce-root  output/final/GNCE \
    --gcare-root output/final/gcare \
    --lmkg-root  output/final/lmkg_preds \
    --lss-root   output/final/lss \
    --out        output/evaluation/
```

Outputs (under `--out`):

- `unified_predictions.csv` — long-format predictions
  (`method, graph, query_dataset, query_type, prediction, true_value,
  q_error, cardinality_bin`).
- `tables/per_method.csv`, `per_method_graph.csv`,
  `per_method_graph_queryset.csv` — aggregated metrics: `median_q, mean_q,
  p90_q, p95_q, p99_q, max_q, count, log_pearson, zero_pred_rate`.
- `plots/` — CCDFs, q-error boxplots, signed-error boxplots, scatter panels.

Q-error per query (`scripts/evaluation/metrics.py`):

```
q_error = max(prediction, ε) / max(true_value, ε)
        and equivalently flipped, taking the max of the two ratios.
```
with `ε = 1`.

Companion runners:

- `scripts/evaluation/run_ablation.py` — FICE vs FICE-without-embeddings vs
  FICE-without-occurrences on the held-out KG.
- `scripts/evaluation/run_transductive.py` — each method evaluated on its own
  training KGs under a deterministic leave-one-out mapping.

Per-baseline adapters live in `scripts/evaluation/adapters/`. Each adapter
normalizes a baseline's native prediction format into the unified long-form
schema. Adding a new baseline is one new adapter.

## Plots and tables

CCDFs and boxplots are produced as part of `run.py`. To re-render from a
saved `unified_predictions.csv`, the helpers in
`scripts/evaluation/plotting.py` (`ccdf_overall`, `ccdf_per_graph`,
`boxplots_overall`, `boxplots_per_graph`, `boxplot_by_query_type`,
`signed_boxplot_overall`, `scatter_overall_panel`) can be called directly.

### Runtime plots

`scripts/evaluation/run_runtime_plot.py` consumes per-method runtime CSVs
(columns: `n_triple_pattern, runtime` in microseconds) and emits a boxplot,
a median-with-IQR line plot, and a summary CSV.

```sh
python scripts/evaluation/run_runtime_plot.py \
    --csv fice=output/runtime/fice_<dataset>.csv \
    --csv gnce=output/runtime/gnce_<dataset>.csv \
    --output-dir output/evaluation/runtime/ \
    --tag <dataset>
```

Per-method runtime CSVs are produced by the inference subsystem (see below).

### LaTeX comparison table

```sh
python scripts/evaluation/make_latex_table.py \
    output/evaluation/tables/per_method_graph.csv \
    --overall output/evaluation/tables/per_method.csv \
    --out output/evaluation/tables/comparison.tex
```

Rows are KGs, columns are methods, cells report median q-error and log-Pearson
correlation; the best method per row is bolded.

## Inference subsystem (`inference/`)

The `inference/` directory exports a trained FICE checkpoint to TorchScript
with mmap-loaded entity and relation embeddings, exposes a C++ integration
path for use inside a query optimizer, and provides per-query runtime
measurement and ranking-dataset construction tools. Build instructions, the
C++ API surface, and the runtime-measurement workflow are documented in
[`inference/README.md`](inference/README.md).

## Reproducing paper results

- Raw KG and query datasets: `<RAW_DATA_LINK_PLACEHOLDER>`
- Pre-computed prediction artifacts used in the paper (corresponds to
  `output/final/` referenced by the evaluation scripts):
  `<RESULTS_LINK_PLACEHOLDER>`

With both artifacts in place, the unified evaluation command above
reproduces every table and figure in the paper end to end.
