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

## Experiment 2: Query GNN Skip Connections (Slurm 331607)
**Hypothesis:** Skip connections in the query GNN preserve raw embedding info alongside GNN-learned interactions.
**Changes:** query_gnn.skip_connection: False→True, all dropout reverted to 0.0

**Results:**
| Metric | Baseline | Exp 2 | Change |
|---|---|---|---|
| overall/median_q_error | 5.707 | 5.423 | -5% better |
| dbpedia100k/median_q_error | 13.346 | 13.763 | +3% worse |
| wn18rr_v4_ind/median_q_error | 3.859 | 4.127 | +7% worse |
| overall/loss | 11.862 | 10.472 | -12% better |
| overall/mean_q_error | 2201.756 | 1631.529 | -26% better |

**Conclusion:** Skip connections help overall/training metrics significantly but hurt inductive generalization. The model fits training distributions better but doesn't transfer as well.

---

## Experiment 3: L2-Normalize Embeddings Before Query Injection (Slurm 331626)
**Hypothesis:** L2-normalizing encoder embeddings makes them scale-invariant across graphs.
**Changes:** Code change in FICE.forward() — F.normalize(embeddings, p=2, dim=-1) on node/edge embeddings. Config reverted to baseline.

**Results:**
| Metric | Baseline | Exp 3 | Change |
|---|---|---|---|
| overall/median_q_error | 5.707 | 5.467 | -4% better |
| dbpedia100k/median_q_error | 13.346 | 14.906 | +12% worse |
| wn18rr_v4_ind/median_q_error | 3.859 | 3.660 | -5% better |
| wn18rr_v4_ind/mean_q_error | 93.384 | 65.325 | -30% better |

**Conclusion:** L2 norm significantly helps wn18rr (smaller graph) but hurts dbpedia100k (larger graph). Magnitude information matters for complex graphs. Reverted.

---

## Experiment 4: 3-Layer Query GNN (Slurm 331630)
**Hypothesis:** 3rd message passing layer covers full path query structure.
**Changes:** TripleModel refactored to support configurable num_layers. query_gnn.num_layers=3.

**Results:**
| Metric | Baseline | Exp 4 | Change |
|---|---|---|---|
| overall/median_q_error | 5.707 | 5.980 | +5% worse |
| dbpedia100k/median_q_error | 13.346 | 15.642 | +17% worse |
| wn18rr_v4_ind/median_q_error | 3.859 | 4.431 | +15% worse |
| overall/mean_q_error | 2201.756 | 1426.083 | -35% better |
| dbpedia100k/mean_q_error | 4186.632 | 1207.079 | -71% better |

**Conclusion:** 3 layers dramatically reduces extreme outliers but hurts median. Extra parameters overfit with limited 10-epoch training budget. Reverted to 2 layers.

---

## Experiment 5: Huber Loss (Smooth L1) Instead of MSE (Slurm 331633)
**Hypothesis:** MSE's quadratic penalty forces the model to focus on outliers. Huber loss is robust to outliers while preserving good gradients for typical cases.
**Changes:** F.mse_loss → F.smooth_l1_loss in train and eval. Config reverted to baseline (2 layers).

**Results:**
| Metric | Baseline | Exp 5 | Change |
|---|---|---|---|
| overall/median_q_error | 5.707 | **5.196** | **-9% better** |
| dbpedia100k/median_q_error | 13.346 | 13.387 | flat |
| wn18rr_v4_ind/median_q_error | 3.859 | **3.752** | **-3% better** |
| wn18rr_v4_ind/mean_q_error | 93.384 | **59.427** | **-36% better** |

**Conclusion:** Clear win. Huber loss improves overall and wn18rr median q-errors. dbpedia100k flat. **Keeping Huber loss going forward.**

---

## Experiment 6: Huber Loss + Encoder Output LayerNorm (Slurm 331637)
**Hypothesis:** Final LayerNorm standardizes embeddings across graphs.
**Changes:** Added output_norm LayerNorm to BipartiteGraphGNN. Huber loss kept.

**Results:**
| Metric | Exp 5 (Huber) | Exp 6 | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.412 | +4% worse |
| dbpedia100k/median_q_error | 13.387 | 15.854 | +18% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.701 | -1% better |

**Conclusion:** Output LayerNorm hurts dbpedia100k badly. Extra normalization is too aggressive. Reverted.

---

## Experiment 7: Huber Loss + Jumping Knowledge in Encoder (Slurm pending)
**Hypothesis:** Different query patterns need different scales of neighborhood info. Star queries need 1-hop, path queries need multi-hop. Current encoder only outputs final (4th) layer. JK concatenates ALL layer outputs (0-hop input through 4-hop) and learns to combine them — giving adaptive multi-scale representations.
**Changes:** Code — BipartiteGraphGNN collects layer_outputs at each hop, concatenates, and projects via learned linear (jk_proj). Output LayerNorm removed. Huber loss kept.
