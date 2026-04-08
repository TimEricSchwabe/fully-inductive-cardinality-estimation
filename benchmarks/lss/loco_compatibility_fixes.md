# LOCO Compatibility Fixes for LSS

LSS was designed for a single-dataset workflow: train on dataset X, test on dataset X.
The FICE benchmark uses **Leave-One-Graph-Out (LOCO)** evaluation: train on a merged
mega-graph of N-1 datasets, evaluate on the held-out dataset. This violated three
assumptions in the original LSS code.

---

## Fix 1 — Model checkpoint size mismatch

**Files:** `LSS/util.py`, `LSS/active_train.py`

**Problem:** The mega-graph has more unique labels than any single held-out dataset
(e.g. 689 edge features vs. 170). LSS re-initialised the model from the test dataset's
feature counts, then tried to load checkpoint weights trained at the larger dimensions —
causing a `RuntimeError: size mismatch`.

**Fix:** `model_checkpoint()` in `util.py` now saves `num_node_feat` / `num_edge_feat`
in the checkpoint. In `active_train.py`, test mode peeks at the checkpoint *before*
constructing the `Queryset`, storing the training-time dimensions in
`args.loco_num_node_feat` / `args.loco_num_edge_feat`.

---

## Fix 2 — `edge_index` collapses to 1-D for single-edge subgraphs

**File:** `LSS/cardnet/model.py`

**Problem:** `QueryDecompose` creates one k-hop spanning tree per query node; leaf nodes
always produce a subgraph with exactly one edge. The DataLoader (batch_size=1) wraps
`edge_index` in a batch dimension, giving shape `[1, 2, 1]`. The original code called
`.squeeze()` (no dim argument), which removes *all* size-1 dimensions and collapses the
tensor to `[2]` (1-D). PyG then raises:
`ValueError: Expected 'edge_index' to be two-dimensional`.

**Fix:** Changed `.squeeze()` → `.squeeze(0)` in `CardNet.forward()`. This removes only
the batch dimension, preserving the `[2, num_edges]` layout regardless of edge count.

---

## Fix 3 — Feature dimension mismatch at inference

**File:** `LSS/Queryset.py`

**Problem:** `Queryset` sized feature vectors from the *current* graph's label count
(`len(edge_label_card)`). At test time this produced 170-dimensional edge features, while
the model's `edge_nn` weight matrix expected 689 (the training mega-graph dimension),
causing `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x170 and 689x32)`.

**Fix:** After computing `self.num_node_feat` / `self.num_edge_feat` from the local graph,
`Queryset.__init__` overrides them with the training-time values from `args` (if present).
`_get_nodes_attr_freq` and `_get_edges_index_freq` allocate tensors at the overridden size
and bounds-check label indices, zero-padding test features to the training dimension.

---

## Why these fixes cannot live in `run_all_experiments.py`

Fix 2 (squeeze) is pure model code — unreachable from the wrapper script. Fix 3 could
theoretically be handled by injecting dummy edges into the eval data graph to inflate its
label count, but that would corrupt the ProNE embeddings and graph structure. The chosen
fixes are minimal and do not affect training behaviour or data files.
