# FICE — Fully Inductive Cardinality Estimator

FICE estimates the cardinality of basic graph pattern (BGP) queries over RDF
knowledge graphs. The model is *fully inductive*: it generalizes to graphs and
relations not seen at training time, without retraining or finetuning. This
repository contains the model implementation, training and evaluation
pipelines, and a C++ inference subsystem suitable for
integration into a low-level query optimizer.

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

The full dependency list is in `pyproject.toml`.

## Datasets

### Pre-built artifacts (recommended)

Knowledge graph `.nt` files, query `.json` files, and pre-computed
ranking-query files for the datasets used in the paper are available at:

[Dataset Link](https://zenodo.org/records/19854409?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIxNTdhOTA1LWRhYTItNDJkYi04YzM5LWNmMDRmYjljODI3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI2ODdkYjkzNDA3Y2ZhMTkzYzFhYzA3Njc3ODRmNDA5MiJ9.GdKHwm47R90cA0P0PWT83QMyXwbmP1dbfv4z5XcMj1T2RnheuU2Ze9El-uxAp13hSmhEs1L5s8N1mA_-BynVYw)

Download and unpack into `data/` so each dataset is located at
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

FICE uses N-Triples: one triple per line, three space-separated tokens (subject,
predicate, object) terminated by a period. Subjects and predicates are IRIs in
angle brackets; objects are IRIs or literals. Malformed lines are rejected.
Example:

```
<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
```

On first load, the dataset class builds an entity-to-id map, a relation-to-id
map, occurrence counts per entity and relation, and factor graph view.
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
| `configs/train_ranking_inductive.yaml` | Ranking-loss training with inductive validation on a held-out KG. |
| `configs/train_ranking_test.yaml` | Minimal ranking example. |

The full config schema is defined in `scripts/utils/config.py`. `configs/train_local.yaml` has a
short explanation for every parameter.

Each run produces a timestamped directory under `output/`:

```
output/<run_name>/
  checkpoints/best.pt           best validation checkpoints
  logs/*.csv                    train/test metrics, per-graph and per-query-type breakdowns
  config.yaml                   snapshot of the resolved configuration
```

## Baselines

Three baselines are under `benchmarks/`. All three read the same
shared experiment config (`benchmarks/experiments_config.yaml`), which
lists the datasets, query files, and per-experiment LOCO folds.

### LSS — multi-graph LOCO

LSS is run in a leave-one-graph-out (LOCO) setup: for each held-out KG, a
single LSS model is trained on a shared mega-graph built from the
N KGs (with globally distinct entity / relation IDs and ProNE
embeddings per graph) and evaluated on the held-out KG.

```sh
pip install -r benchmarks/lss/requirements.txt
python benchmarks/lss/run_all_experiments.py
# optional flags:
#   --epochs 80
#   --experiments loco_swdf loco_lubm
```

Outputs are saved in `benchmarks/lss/results/run_<timestamp>/<exp_id>/predictions.csv`,
which is the layout the unified evaluation pipeline expects under `--lss-root`.

### LMKG-S — multi-graph LOCO

LMKG-S is also run in LOCO; for each held-out KG, two models (one for
star, one for path) are trained on the merged mega-graph of the other
N−1 KGs and evaluated on the held-out KG.

```sh
# Python 3.8 + tensorflow 2.13 are required (see benchmarks/lmkg/README.md).
pip install -r benchmarks/lmkg/LMKG-Code/lmkg-s/requirements.txt
python benchmarks/lmkg/run_all_experiemnts.py
```

Outputs are stored under `benchmarks/lmkg/LMKG-Code/lmkg-s/results/`. The
unified evaluation pipeline expects an `all_predictions.csv` under
`--lmkg-root` (produced by the LMKG run script). See
[`benchmarks/lmkg/README.md`](benchmarks/lmkg/README.md) for more details.

### G-CARE — single-graph estimators

G-CARE provides classical single-graph cardinality estimators
(`wj`, `cset`, `sumrdf`, …) via a compiled C++ binary. Unlike LSS and
LMKG-S, each experiment runs against a single KG (no LOCO mega-graph)
and there is no learned model — only a build-statistics step followed by
estimation. Build the binary once (e.g. inside the supplied Apptainer
image; see `benchmarks/gcare/gcare.def`), then:

```sh
python benchmarks/gcare/run_all_experiments.py \
    --gcare-home benchmarks/gcare/gcare \
    --max-parallel 5
# optional filters:
#   --dataset yago,hetionet
#   --query-types path,star
#   --methods wj,cset,sumrdf
```

Outputs are stored in
`benchmarks/gcare/results/run_<timestamp>/<graph>_<qtype>/<method>/{preds,gts}.npy`,
which is the layout the unified evaluation pipeline expects under
`--gcare-root`.

## Evaluation

The evaluation pipeline expects prediction files from FICE and the
baselines, normalizes them through per-method adapters, and creates
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


The ablations and transductive runs can be evaluated using:

- `scripts/evaluation/run_ablation.py` — FICE vs FICE-without-embeddings vs
  FICE-without-occurrences on the held-out KG.
- `scripts/evaluation/run_transductive.py` — each method evaluated on its own
  training KGs under a deterministic leave-one-out mapping.



### Runtime plots

`scripts/evaluation/run_runtime_plot.py` expects per-method runtime CSVs
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


## Inference subsystem (`inference/`)

The `inference/` directory exports a trained FICE checkpoint to TorchScript
with mmap-loaded entity and relation embeddings, exposes a C++ integration
path for use inside a query optimizer, and provides per-query runtime
measurement and ranking-dataset construction tools. Build instructions, the
C++ API surface, and the runtime-measurement workflow are documented in
[`inference/README.md`](inference/README.md).

## Reproducing paper results

- Raw KG and query datasets: [Dataset Link](https://zenodo.org/records/19854409?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIxNTdhOTA1LWRhYTItNDJkYi04YzM5LWNmMDRmYjljODI3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI2ODdkYjkzNDA3Y2ZhMTkzYzFhYzA3Njc3ODRmNDA5MiJ9.GdKHwm47R90cA0P0PWT83QMyXwbmP1dbfv4z5XcMj1T2RnheuU2Ze9El-uxAp13hSmhEs1L5s8N1mA_-BynVYw)
- Pre-computed prediction artifacts used in the paper (corresponds to
  `output/final/` referenced by the evaluation scripts):
  `[Results Link](https://zenodo.org/records/19854794?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFiYWQ5NTc2LWE2Y2YtNGY0My1iZGVmLTY1MzdlNmQ2ODg3YiIsImRhdGEiOnt9LCJyYW5kb20iOiJlNzBhYzk3M2FhZmE2NzJlM2NjYzlhNDc1MzU3MDFiMSJ9.9KDTlA-LG8CLP7gqAcExr_CWpz3Th9huKj0hyJVUQNyT7xalnBVYCu5dky38L5ivjdD6nPpDGleQK4qW4BcYew)

