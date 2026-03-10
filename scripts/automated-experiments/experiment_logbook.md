# Experiment Logbook

## Experiment 0: Fresh Baseline (Slurm 331598)
**Config:** factor-factor, 4 layers, 128-dim, GINE, TripleModel, no dropout, no conditioning, skip_conn on for encoders off for query, lr=5e-4, wd=0.01, grad_clip=1.0, scheduler on

**Results:**
| Metric | Value |
|---|---|
| test/overall/loss | 11.862 |
| test/overall/median_q_error | 5.707 |
| test/overall/q_error | 2201.756 |
| test/per_graph/dbpedia100k/loss | 17.325 |
| test/per_graph/dbpedia100k/median_q_error | 13.346 |
| test/per_graph/dbpedia100k/q_error | 4186.632 |
| test/per_graph/wn18rr_v4_ind/loss | 4.886 |
| test/per_graph/wn18rr_v4_ind/median_q_error | 3.859 |
| test/per_graph/wn18rr_v4_ind/q_error | 93.384 |

---

## Experiment 1: Dropout Regularization 0.1 (Slurm 331602)
**Hypothesis:** Adding dropout to encoder and query GNNs improves inductive generalization.
**Changes:** entity_gnn.dropout: 0.0→0.1, relation_gnn.dropout: 0.0→0.1, query_gnn.dropout: 0.0→0.1

**Results:**
| Metric | Baseline | Exp 1 | Change |
|---|---|---|---|
| overall/median_q_error | 5.707 | 6.405 | +12% worse |
| dbpedia100k/median_q_error | 13.346 | 14.656 | +10% worse |
| wn18rr_v4_ind/median_q_error | 3.859 | 4.487 | +16% worse |
| overall/mean_q_error | 2201.756 | 1840.046 | -16% better |

**Conclusion:** Dropout 0.1 hurts median q-errors but reduces extreme outliers. With only 10 epochs, the regularization is too aggressive — the model underfits.

---

## Experiment 2: Query GNN Skip Connections (Slurm pending)
**Hypothesis:** Skip connections in the query GNN preserve raw embedding info alongside GNN-learned interactions, helping the readout use both local statistics and structural patterns.
**Changes:** query_gnn.skip_connection: False→True, all dropout reverted to 0.0
