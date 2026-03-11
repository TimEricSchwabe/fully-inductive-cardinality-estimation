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

## Experiment 7: Huber Loss + Jumping Knowledge in Encoder (Slurm 331642)
**Hypothesis:** JK gives adaptive multi-scale representations for different query types.
**Changes:** BipartiteGraphGNN collects all layer outputs, concatenates, and projects via jk_proj.

**Results:**
| Metric | Exp 5 (Huber) | Exp 7 | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 6.031 | +16% worse |
| dbpedia100k/median_q_error | 13.387 | 15.439 | +15% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 4.077 | +9% worse |

**Conclusion:** JK adds too many params (~82K in jk_proj) that don't converge in 10 epochs. Reverted.

---

## Experiment 8: Huber Loss + Higher LR 1e-3 (Slurm 331644)
**Hypothesis:** Higher LR compensates for Huber's smaller gradients.
**Changes:** learning_rate: 5e-4 → 1e-3.

**Results:**
| Metric | Exp 5 (Huber, 5e-4) | Exp 8 (1e-3) | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.310 | +2% worse |
| dbpedia100k/median_q_error | 13.387 | 14.532 | +9% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.726 | -1% marginal |

**Conclusion:** LR 1e-3 too high, overshoots. Reverted to 5e-4.

---

## Experiment 9: Huber Loss + GPS + Relation-level Transformer (Slurm 331649)
**Hypothesis:** GPS (local GINE + global self-attention per layer) replaces GINE for encoders. Relation-level transformer (post-GNN) still active from earlier code.
**Changes:** GPS encoder (2 layers), relation transformer still present. Huber loss + LR 5e-4.
**Note:** This was intended as GPS-only but the relation transformer code from a previous experiment was still active, making this GPS + relation transformer combined.

**Results:**
| Metric | Exp 5 (Huber) | Exp 9 | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.962 | +15% worse |
| dbpedia100k/median_q_error | 13.387 | 13.143 | -2% better |
| wn18rr_v4_ind/median_q_error | 3.752 | 4.115 | +10% worse |
| overall/loss | ~11.8 (baseline) | 2.081 | much lower |
| overall/mean_q_error | 2201.756 | 2853.724 | +30% worse |

**Conclusion:** GPS helps dbpedia100k (best dbpedia result so far) but significantly hurts wn18rr and overall. Very slow: ~25 min/epoch vs ~1 min with GINE. Lower loss but higher median q-error suggests overfitting on training distribution.

---

## Experiment 10: Clean GPS (no relation transformer) (Slurm 331651)
**Hypothesis:** Isolate the GPS effect without the relation transformer confound.
**Changes:** Removed relation transformer from BipartiteGraphGNN. GPS encoder (2 layers). Huber loss + LR 5e-4.

**Results:** OOM — tried to allocate 158.89 GiB for attention matrix on 140 GiB H200. GPS global self-attention over all nodes in sampled neighborhoods is too expensive for larger graphs like dbpedia100k.

**Conclusion:** GPS is intractable even on sampled neighborhoods for large graphs. Need targeted attention on subsets of nodes instead.

---

## Experiment 11: GINE + Relation-only Self-Attention (Slurm 331658)
**Hypothesis:** Post-GINE self-attention over relation nodes gives relations a global view of all other relations. Entity attention OOM'd (sampled subgraphs have ~12K entity nodes), so relation-only.
**Changes:** BipartiteGraphGNN: 4 GINE layers + post-GNN transformer block (MultiheadAttention + FFN, pre-norm) over relation nodes using mapped_relation_indices. Huber loss + LR 5e-4.

**Results:**
| Metric | Exp 5 (Huber) | Exp 11 | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.962 | +15% worse |
| dbpedia100k/median_q_error | 13.387 | 15.860 | +18% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.868 | +3% worse |
| wn18rr_v4_ind/mean_q_error | 59.427 | 33.110 | -44% better |

**Conclusion:** Relation attention hurts median across the board but dramatically reduces wn18rr extreme outliers. Extra attention params don't converge in 10 epochs — same pattern as other capacity-adding changes.

---

## Experiment 12: GINE + Entity & Relation Attention, batch_size=8 (Slurm 331661)
**Hypothesis:** Entity attention OOM'd at bs=32. Smaller batch_size=8 enables entity + relation attention. Entity attention captures entity co-occurrence globally.
**Changes:** Both entity and relation post-GINE attention. batch_size: 32→8. 4 GINE layers. Huber loss + LR 5e-4.

**Results:**
| Metric | Exp 5 (Huber, bs=32) | Exp 12 (bs=8) | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.919 | +14% worse |
| dbpedia100k/median_q_error | 13.387 | 16.366 | +22% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.743 | flat |
| overall/mean_q_error | 2201.756 | 2409.873 | +9% worse |

**Conclusion:** Entity attention doesn't help — adding capacity that doesn't converge. Smaller batch size may also hurt optimization dynamics (fewer samples per gradient step). The attention approach is consistently harmful to median q-error.

---

## Experiment 13: Clean GPS Encoder, batch_size=8 (Slurm 331667)
**Hypothesis:** GPS with smaller batch avoids OOM. Clean test of GPS without relation transformer.
**Changes:** GPS encoder (2 layers), no attention, batch_size=8. Huber loss + LR 5e-4.

**Results:**
| Metric | Exp 5 (Huber, bs=32) | Exp 13 (GPS, bs=8) | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 6.146 | +18% worse |
| dbpedia100k/median_q_error | 13.387 | 13.841 | +3% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.703 | -1% better |
| wn18rr_v4_ind/mean_q_error | 59.427 | 41.353 | -30% better |

**Conclusion:** GPS doesn't improve median q-errors. Same pattern: extra capacity helps mean/outliers but hurts median. Also confounded by batch_size=8 vs baseline's 32.

---

## Experiment 14: Huber Loss + Lower Weight Decay 0.001 (Slurm 331680)
**Hypothesis:** WD=0.01 may be too aggressive, preventing convergence in 10 epochs.
**Changes:** weight_decay: 0.01→0.001. GINE 4 layers, Huber loss, LR 5e-4.

**Results:**
| Metric | Exp 5 (Huber, wd=0.01) | Exp 14 (wd=0.001) | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.567 | +7% worse |
| dbpedia100k/median_q_error | 13.387 | 15.024 | +12% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.739 | flat |
| overall/mean_q_error | 2201.756 | 1850.713 | -16% better |

**Conclusion:** Lower WD helps mean but hurts dbpedia inductive generalization. WD=0.01 is a good regularizer. Reverted.

---

## Experiment 15: Huber Loss beta=0.5 (Slurm 331681)
**Hypothesis:** Smaller Huber beta shifts loss toward pure L1, focusing more on typical cases.
**Changes:** smooth_l1_loss beta: 1.0→0.5. GINE 4 layers, LR 5e-4, WD 0.01.

**Results:**
| Metric | Exp 5 (beta=1) | Exp 15 (beta=0.5) | Change |
|---|---|---|---|
| overall/median_q_error | 5.196 | 5.891 | +13% worse |
| dbpedia100k/median_q_error | 13.387 | 16.113 | +20% worse |
| wn18rr_v4_ind/median_q_error | 3.752 | 3.929 | +5% worse |

**Conclusion:** Smaller beta hurts. The quadratic region for small errors provides better gradient signal. beta=1 is the right default. Reverted.

---

## Experiment 16: More Data + Longer Training (Slurm 331682)
**Hypothesis:** All capacity-adding experiments show underfitting patterns (better mean, worse median). More queries (8K vs 4K) and more epochs (20 vs 10) should give the model enough data and training time to converge properly.
**Changes:** num_queries: 4000→8000, epochs: 10→20. Otherwise same as Exp 5 (Huber, GINE 4 layers).
**Note:** Test set changes with more queries, so direct comparison with previous experiments is approximate.

**Results:**
| Metric | Exp 5 (Huber, 4K/10ep) | Exp 16 (8K/20ep) | Change |
|---|---|---|---|
| overall/loss | ~2.0 | 1.293 | much better |
| overall/median_q_error | 5.196 | **2.746** | **-47% better** |
| overall/mean_q_error | 2201.756 | **1679.889** | **-24% better** |
| dbpedia100k/median_q_error | 13.387 | **4.794** | **-64% better** |
| dbpedia100k/mean_q_error | ~4186 | 1738.516 | -58% better |
| wn18rr_v4_ind/median_q_error | 3.752 | **1.860** | **-50% better** |
| wn18rr_v4_ind/mean_q_error | 59.427 | **27.149** | **-54% better** |

**Conclusion:** Massive improvement across ALL metrics. The model was severely underfitting at 4K queries / 10 epochs. This is now the new baseline for all future experiments. Opens the door to re-test capacity-adding changes (GPS, attention, more layers, dropout) that previously failed due to underfitting.

---

## Experiment 17: 3-Layer Query GNN (Slurm 331684)
**Hypothesis:** Path queries have 2-3 edges, so 3 message passing layers match the query graph diameter. With 2 layers, endpoints of a 3-edge path can't communicate. Previously (exp 4, 4K/10ep), 3 layers showed -35% mean but +5% median — the underfitting pattern. With 8K/20ep, the extra capacity should now converge properly.
**Changes:** query_gnn.num_layers: 2→3. Otherwise same as Exp 16 (8K/20ep, Huber, GINE 4 layers).

**Results:**
| Metric | Exp 16 (2 layers) | Exp 17 (3 layers) | Change |
|---|---|---|---|
| overall/median_q_error | 2.746 | 2.741 | flat |
| dbpedia100k/median_q_error | 4.794 | **4.147** | **-13.5% better** |
| wn18rr_v4_ind/median_q_error | 1.860 | 2.001 | +7.6% worse |
| overall/mean_q_error | 1679.889 | 1668.547 | flat |
| dbpedia100k/mean_q_error | 1738.516 | 3004.192 | +73% worse |
| wn18rr_v4_ind/mean_q_error | 27.149 | **16.340** | **-40% better** |

**Conclusion:** Mixed. 3 layers significantly helps dbpedia100k median (-13.5%) but hurts wn18rr_v4_ind median (+7.6%). The extra receptive field helps larger/more complex graphs (dbpedia) but may overfit smaller ones (wn18rr). wn18rr mean improves dramatically suggesting outlier handling is better. Reverted to 2 layers for now — need a different approach to help both datasets.

---

## Experiment 18: Dropout 0.1 All GNNs (Slurm 331692)
**Hypothesis:** With 20 epochs, the model trains long enough to benefit from regularization. Previously (exp 1, 4K/10ep), dropout hurt median but improved mean — the classic underfitting pattern. With sufficient data and training, dropout should help inductive generalization by preventing the model from memorizing training graph structure.
**Changes:** entity_gnn.dropout: 0.0→0.1, relation_gnn.dropout: 0.0→0.1, query_gnn.dropout: 0.0→0.1. Otherwise same as Exp 16.

**Results:**
| Metric | Exp 16 (no dropout) | Exp 18 (dropout 0.1) | Change |
|---|---|---|---|
| overall/median_q_error | 2.746 | 2.930 | +6.7% worse |
| dbpedia100k/median_q_error | 4.794 | **3.985** | **-16.9% better** |
| wn18rr_v4_ind/median_q_error | 1.860 | 2.267 | +21.9% worse |
| overall/mean_q_error | 1679.889 | **1302.546** | **-22.5% better** |
| wn18rr_v4_ind/mean_q_error | 27.149 | 25.154 | -7% better |

**Conclusion:** Same pattern as exp 17: helps dbpedia100k significantly (best dbpedia median yet) but hurts wn18rr_v4_ind. Regularization helps the truly unseen graph (dbpedia) but hurts transfer from related training graph (wn18rr_v4→wn18rr_v4_ind). Reverted.

---

## Experiment 19: Query GNN Skip Connections (Slurm 331696)
**Hypothesis:** Skip connections in the query GNN preserve raw embedding info alongside GNN-learned interactions. Previously (exp 2, 4K/10ep), skip helped overall median (-5%) and mean (-26%) but hurt individual datasets slightly. With 8K/20ep, the model should converge properly. Unlike dropout/capacity changes that show a dbpedia-vs-wn18rr tradeoff, skip connections add information without removing capacity.
**Changes:** query_gnn.skip_connection: False→True. All dropout reverted to 0.0. Otherwise same as Exp 16.

**Results:**
| Metric | Exp 16 (no skip) | Exp 19 (skip) | Change |
|---|---|---|---|
| overall/median_q_error | 2.746 | 3.017 | +9.9% worse |
| dbpedia100k/median_q_error | 4.794 | 4.471 | -6.7% better |
| wn18rr_v4_ind/median_q_error | 1.860 | 2.456 | +32% worse |
| overall/mean_q_error | 1679.9 | 48676.5 | exploded |

**Conclusion:** Skip connections in query GNN are harmful. Raw embeddings bypassing learned transformations create extreme outliers (mean explodes). wn18rr significantly degraded. Reverted.

---

## Experiment 20: Larger Embedding Dimension 192 (Slurm 331702)
**Hypothesis:** 128-dim may bottleneck representation quality. Increasing to 192 gives 50% more capacity to encode structural information. Unlike regularization/architecture changes (exps 17-19) that show a dbpedia-vs-wn18rr tradeoff, wider embeddings improve expressiveness for ALL graphs without changing the learning dynamics.
**Changes:** embedding_dim: 128→192. Otherwise same as Exp 16.

**Results:**
| Metric | Exp 16 (128-dim) | Exp 20 (192-dim) | Change |
|---|---|---|---|
| overall/median_q_error | 2.746 | 2.696 | -1.8% better |
| dbpedia100k/median_q_error | 4.794 | 5.036 | +5% worse |
| wn18rr_v4_ind/median_q_error | 1.860 | 1.954 | +5% worse |
| overall/mean_q_error | 1679.9 | 2806.4 | +67% worse |

**Conclusion:** Wider embeddings marginally help overall but hurt both inductive test sets and increase outliers. 192-dim adds ~350K extra params that don't converge in 20 epochs. Reverted to 128.

---

## Experiment 21: 3-Layer Query GNN + Dropout 0.1 Combined (Slurm 331711)
**Hypothesis:** Exp 17 (3 layers) and Exp 18 (dropout) both significantly improved dbpedia median (-13.5% and -16.9%) while targeting different aspects: receptive field vs regularization. Combining them may compound the dbpedia gains. Both hurt wn18rr individually, but wn18rr at 1.86 is already very good — even at 2.0-2.3 it's still excellent for a fully inductive model.
**Changes:** query_gnn.num_layers: 2→3, all dropout: 0.0→0.1. Otherwise same as Exp 16.

**Results:**
| Metric | Exp 16 (baseline) | Exp 21 (combined) | Change |
|---|---|---|---|
| overall/median_q_error | 2.746 | 2.940 | +7.1% worse |
| dbpedia100k/median_q_error | 4.794 | 4.071 | -15.1% better |
| wn18rr_v4_ind/median_q_error | 1.860 | 2.235 | +20.2% worse |
| overall/mean_q_error | 1679.9 | 1741.6 | +3.7% worse |

**Conclusion:** Effects don't compound — results fall between individual experiments. The dbpedia-vs-wn18rr tradeoff is fundamental: changes that improve generalization to truly unseen graphs hurt transfer from related training graphs. Need a different approach that improves the encoder's learned representations rather than the query GNN or regularization.

---

## Experiment 22: Deeper Encoder — 6 Layers (Slurm TBD)
**Hypothesis:** Experiments 17-21 all modified the query side or regularization, revealing a fundamental dbpedia-vs-wn18rr tradeoff. A different approach: deepen the encoder (BipartiteGraphGNN) from 4→6 layers. More encoder layers capture higher-order structural motifs in the factor graph (3-hop+ neighborhoods), producing richer embeddings that should benefit ALL downstream graphs. Unlike query-side changes, this improves the quality of the learned representations themselves. With skip connections and layer norm already enabled, deeper encoders should train stably.
**Changes:** n_embedding_layers: 4→6. Otherwise identical to Exp 16 baseline.
