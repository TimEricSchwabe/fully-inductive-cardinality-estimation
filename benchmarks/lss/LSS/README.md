# LSS: A Learned Sketch for Subgraph Counting

PyTorch + torch-geometric implementation of LSS, as described in:

> Kangfei Zhao, Jeffrey Xu Yu, Hao Zhang, Qiyan Li, Yu Rong.
> [A Learned Sketch for Subgraph Counting](https://dl.acm.org/doi/10.1145/3448016.3457289), SIGMOD '21

LSS estimates the number of homomorphic/isomorphic matches of a query graph inside a large data graph using a GNN-based learned sketch.

---

## Requirements

**Python 3.7**, with:

```
torch >= 1.7.1
torch-geometric == 1.6.3
torch-scatter, torch-sparse, torch-cluster, torch-spline-conv
numpy, scipy, scikit-learn
networkx, gpytorch
pandas, h5py, tqdm
```

Install via conda (recommended):

```bash
conda env create -f gpcard.yml
conda activate gpcard
```

---

## Data

Place your dataset files under `./data/` using the following structure:

```
data/
├── dataset/<name>/          # data graph(s) in G-Care text format
├── queryset_homo/<name>/    # query graphs in G-Care text format
├── true_homo/<name>/        # one cardinality per query (integer, one per file)
└── prone/<name>.emb.npy     # ProNE node embeddings (required if --embed_type prone)
```

A sample `aids` dataset is included to test your setup.

**Graph format** (G-Care `.txt`):
```
t # <graph_id>
v <node_id> <label> [<label2> ...]
e <src_id> <dst_id> [<edge_label> ...]
```

**Query format**: same as graph format, but using `t # s <query_id>` as header.

**Ground truth**: one file per query containing a single integer (the cardinality).

---

## Running

All modes are accessed via `active_train.py`. Run from inside `LSS/`.

**Cross-validation** (evaluate on held-out folds):
```bash
python active_train.py --dataset aids --embed_type prone --mode cross_val
```

**Train + active learning** (iteratively select uncertain queries to label):
```bash
python active_train.py --dataset aids --mode train --embed_type prone \
    --active_iters 1 --uncertainty consist
```

**Ensemble active learning** (multiple models vote on uncertainty):
```bash
python active_train.py --dataset aids --mode ensemble --embed_type prone \
    --ensemble_num 5 --active_iters 1
```

**Evaluate a saved model**:
```bash
python active_train.py --dataset aids --mode test --embed_type freq \
    --model_file models/<name>/best.pth
```

Results are saved to `./result/` by default. Model checkpoints go to `./models/`.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--dataset` | `aids` | Dataset name (must match folder names under `data/`) |
| `--mode` | `cross_val` | `cross_val`, `train`, `test`, `pretrain`, `ensemble` |
| `--embed_type` | `freq` | Node features: `freq` (frequency), `prone`, `n2v`, `nrp` |
| `--model_type` | `NNGINConcat` | GNN layer: `GIN`, `GINE`, `GAT`, `GCN`, `SAGE`, `NNGIN`, `NNGINConcat` |
| `--num_layers` | `3` | Number of GNN layers |
| `--epochs` | `80` | Training epochs |
| `--batch_size` | `2` | Mini-batch size |
| `--num_fold` | `5` | Number of cross-validation folds |
| `--k` | `3` | Query decomposition depth (hops) |
| `--active_iters` | `2` | Active learning iterations |
| `--budget` | `50` | Queries selected per active learning iteration |
| `--uncertainty` | `consist` | Uncertainty strategy: `entropy`, `margin`, `confident`, `consist`, `random` |
| `--ensemble_num` | `5` | Number of models for ensemble mode |
| `--matching` | `homo` | `homo` (homomorphic) or `iso` (isomorphic) |

All parameters and their defaults are defined at the top of `active_train.py`.

---

## Project Structure

```
active_train.py      # entry point and all parameter definitions
QuerySampler.py      # query decomposition and data loading
Queryset.py          # input encoding and PyTorch DataLoader
util.py              # model save/load helpers
baseline.py          # result evaluation utilities (legacy)
cardnet/
    model.py         # CardNet GNN model
    GINlayers.py     # custom GIN layer variants (NNGIN, NNGINConcat)
    layers.py        # basic layers
    loss.py          # loss functions
active/
    ActiveLearner.py # active learning loop
    active_util.py   # uncertainty sampling utilities
data/
    dataset/aids/    # sample data graph
    queryset_homo/aids/
    true_homo/aids/
    prone/aids/
```

---

## Citation

```bibtex
@inproceedings{DBLP:conf/sigmod/ZhaoYZLR21,
  author    = {Kangfei Zhao and Jeffrey Xu Yu and Hao Zhang and Qiyan Li and Yu Rong},
  title     = {A Learned Sketch for Subgraph Counting},
  booktitle = {{SIGMOD} '21},
  pages     = {2142--2155},
  publisher = {{ACM}},
  year      = {2021}
}
```
