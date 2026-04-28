"""Diagnose Q-error discrepancy between training and inference pipeline.

Tests three embedding strategies:
  A) Full-graph embeddings (what inference pipeline does with batch_size=0)
  B) Batched embeddings with neighbor sampling (batch_size>0 in prepare.py)
  C) Per-query embeddings (what training actually does) — sample around query entities
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils.config import load_config_yaml
from utils.checkpoint_helper import load_checkpoint
from utils.data_helper import load_dataset, load_occurrences, get_or_build_graph
from utils.helper import set_random_seeds
from utils.neighborhood_sampler import MultigraphNeighborSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import build_fice_from_config, compute_embeddings_full_graph, compute_embeddings_batched


def build_query_and_predict(triples, entity_emb, relation_emb, eo, ro, em, rm, num_edges, model):
    """Build query graph with precomputed embeddings, run query_gnn only."""
    local_nodes = {}
    for s, p, o in triples:
        if s not in local_nodes:
            local_nodes[s] = len(local_nodes)
        if o not in local_nodes:
            local_nodes[o] = len(local_nodes)

    n_nodes = len(local_nodes)
    emb_dim = entity_emb.shape[1]
    rel_dim = relation_emb.shape[1]

    x = torch.zeros(n_nodes, 3 + emb_dim)
    for uri, lid in local_nodes.items():
        if not uri.startswith("?"):
            if uri in eo:
                x[lid, :3] = torch.tensor(eo[uri], dtype=torch.float32)
            if uri in em:
                x[lid, 3:] = entity_emb[em[uri]]

    src, dst = [], []
    edge_attr = torch.zeros(len(triples), 3 + rel_dim)
    for i, (s, p, o) in enumerate(triples):
        src.append(local_nodes[s])
        dst.append(local_nodes[o])
        if not p.startswith("?"):
            if p in ro:
                edge_attr[i, :3] = torch.tensor(ro[p], dtype=torch.float32)
            if p in rm:
                edge_attr[i, 3:] = relation_emb[rm[p]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x[:, :3] = torch.log1p(x[:, :3])
    edge_attr[:, :3] = torch.log1p(edge_attr[:, :3])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data])
    graph_size = torch.tensor([float(num_edges)])

    with torch.no_grad():
        return model.query_gnn(batch=batch, graph_size=graph_size).item()


def run_full_model_per_query(triples, model, graph_data, eo, ro, em, rm,
                             num_edges, sampler):
    """Run query the way training does: sample subgraph, run full FICE model."""
    offset = graph_data.original_num_nodes

    local_nodes = {}
    for s, p, o in triples:
        if s not in local_nodes:
            local_nodes[s] = len(local_nodes)
        if o not in local_nodes:
            local_nodes[o] = len(local_nodes)

    n_nodes = len(local_nodes)
    emb_dim = model.vector_dim

    # Global embedding indices
    node_embedding_indices = torch.full((n_nodes,), -1, dtype=torch.long)
    edge_embedding_indices = torch.full((len(triples),), -1, dtype=torch.long)

    for uri, lid in local_nodes.items():
        if not uri.startswith("?") and uri in em:
            node_embedding_indices[lid] = em[uri]

    for i, (s, p, o) in enumerate(triples):
        if not p.startswith("?") and p in rm:
            edge_embedding_indices[i] = rm[p]

    # Seeds for neighbor sampling
    seeds = torch.cat([node_embedding_indices, edge_embedding_indices + offset])
    seeds = seeds[seeds >= 0].unique()

    if seeds.numel() == 0:
        # All variables
        x = torch.zeros(n_nodes, 3 + emb_dim)
        edge_attr = torch.zeros(len(triples), 3 + emb_dim)
        src, dst = [], []
        for i, (s, p, o) in enumerate(triples):
            src.append(local_nodes[s])
            dst.append(local_nodes[o])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            return model.query_gnn(batch=batch, graph_size=torch.tensor([float(num_edges)])).item()

    # Sample subgraph
    sub_data, global_to_local = sampler.sample(graph_data, seeds)
    sub_data.original_num_nodes = graph_data.original_num_nodes
    sub_data.original_num_relations = graph_data.original_num_relations

    # Build query batch with RAW occurrence counts (FICE.forward applies log1p internally)
    x = torch.zeros(n_nodes, 3 + emb_dim)
    for uri, lid in local_nodes.items():
        if not uri.startswith("?") and uri in eo:
            x[lid, :3] = torch.tensor(eo[uri], dtype=torch.float32)

    src, dst = [], []
    edge_attr = torch.zeros(len(triples), 3 + emb_dim)
    for i, (s, p, o) in enumerate(triples):
        src.append(local_nodes[s])
        dst.append(local_nodes[o])
        if not p.startswith("?") and p in ro:
            edge_attr[i, :3] = torch.tensor(ro[p], dtype=torch.float32)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # NOTE: do NOT apply log1p here — FICE.forward() does it at line 1053-1054

    # Remap embedding indices to local subgraph IDs
    local_node_emb = torch.full_like(node_embedding_indices, -1)
    for lid in range(n_nodes):
        gid = node_embedding_indices[lid].item()
        if gid >= 0:
            local_node_emb[lid] = global_to_local[gid].item()

    local_edge_emb = torch.full_like(edge_embedding_indices, -1)
    for i in range(len(triples)):
        gid = edge_embedding_indices[i].item()
        if gid >= 0:
            local_edge_emb[i] = global_to_local[gid + offset].item()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_embedding_indices = local_node_emb
    data.edge_embedding_indices = local_edge_emb
    batch = Batch.from_data_list([data])
    graph_size = torch.tensor([float(num_edges)])

    with torch.no_grad():
        pred = model(data=sub_data, relation_graph=sub_data,
                     batch=batch, graph_size=graph_size)
    return pred.item()


def report_qerror(name, q_errors):
    print(f"\n  {name} (n={len(q_errors)}):")
    print(f"    Mean: {q_errors.mean():.2f}  Median: {np.median(q_errors):.2f}")
    print(f"    P90: {np.percentile(q_errors, 90):.2f}  "
          f"P95: {np.percentile(q_errors, 95):.2f}  "
          f"P99: {np.percentile(q_errors, 99):.2f}  Max: {q_errors.max():.2f}")


def compute_qerrors(predictions, ground_truths):
    pred_card = np.expm1(np.clip(predictions, 0, None))
    gt_clamped = np.maximum(ground_truths, 1.0)
    pred_clamped = np.maximum(pred_card, 1.0)
    return np.maximum(pred_clamped / gt_clamped, gt_clamped / pred_clamped)


def main():
    dataset_name = "fb237_v4"
    query_name = "fb237_v4_star_queries"
    queries_path = f"/home/tim/fully-inductive-cardinality-estimation/data/{query_name}/raw/{query_name}"
    with open(queries_path) as f:
        all_queries = json.load(f)

    train_config = load_config_yaml("/home/tim/fully-inductive-cardinality-estimation/output/BigModel/config.yaml")
    checkpoint = load_checkpoint("/home/tim/fully-inductive-cardinality-estimation/output/BigModel/checkpoints/best_model.pt")
    state_dict = checkpoint["model_state_dict"]

    model = build_fice_from_config(train_config, state_dict=state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    graph = load_dataset(dataset_name)
    relation_source = train_config.relation_gnn.source
    graph_data = get_or_build_graph(graph, dataset_name,
                                    "factor" if relation_source == "factor" else "bipartite")
    num_edges = graph.edge_index.size(1)
    print(f"Graph: {graph_data.original_num_nodes} entities, "
          f"{graph_data.original_num_relations} relations, {num_edges} edges")

    eo, ro, em, rm = load_occurrences(dataset_name)
    num_neighbors = train_config.GNNNeighborhood or [100, 100]

    # Reproduce exact val split from training (train-val 80/20 on first 20000)
    set_random_seeds(42)
    num_queries = 20000
    first_n = all_queries[:num_queries]
    indices = list(range(num_queries))
    random.shuffle(indices)
    val_indices = indices[int(num_queries * 0.8):]
    queries = [first_n[i] for i in val_indices[:200]]  # use 200 for speed
    N = len(queries)
    ground_truths = np.array([q["y"] for q in queries], dtype=np.float64)
    sizes = [len(q["triples"]) for q in queries]
    print(f"Testing on {N} val queries (exact training split), num_neighbors={num_neighbors}")
    print(f"  Size distribution: {dict(sorted(Counter(sizes).items()))}")

    # --- Strategy A: Full-graph embeddings ---
    print("\n[A] Full-graph embeddings...")
    entity_emb_full, relation_emb_full = compute_embeddings_full_graph(
        model, graph_data, device="cpu")
    preds_a = np.array([
        build_query_and_predict(q["triples"], entity_emb_full, relation_emb_full,
                                eo, ro, em, rm, num_edges, model)
        for q in queries
    ])

    # --- Strategy B: Batched neighbor-sampled embeddings ---
    print("[B] Batched neighbor-sampled embeddings...")
    entity_emb_batch, relation_emb_batch = compute_embeddings_batched(
        model, graph_data, device="cpu", batch_size=256, num_neighbors=num_neighbors)
    preds_b = np.array([
        build_query_and_predict(q["triples"], entity_emb_batch, relation_emb_batch,
                                eo, ro, em, rm, num_edges, model)
        for q in queries
    ])

    # --- Strategy B2: L2-normalized batched embeddings ---
    print("[B2] Batched embeddings + L2 normalization...")
    ent_norm = torch.nn.functional.normalize(entity_emb_batch, dim=1)
    rel_norm = torch.nn.functional.normalize(relation_emb_batch, dim=1)
    preds_b2 = np.array([
        build_query_and_predict(q["triples"], ent_norm, rel_norm,
                                eo, ro, em, rm, num_edges, model)
        for q in queries
    ])

    # --- Strategy B3: Standardized batched embeddings (zero mean, unit var) ---
    print("[B3] Batched embeddings + standardization...")
    ent_mean, ent_std = entity_emb_batch.mean(0), entity_emb_batch.std(0).clamp(min=1e-6)
    rel_mean, rel_std = relation_emb_batch.mean(0), relation_emb_batch.std(0).clamp(min=1e-6)
    ent_std_emb = (entity_emb_batch - ent_mean) / ent_std
    rel_std_emb = (relation_emb_batch - rel_mean) / rel_std
    preds_b3 = np.array([
        build_query_and_predict(q["triples"], ent_std_emb, rel_std_emb,
                                eo, ro, em, rm, num_edges, model)
        for q in queries
    ])

    # --- Strategy C: Per-query full model (like training) ---
    print(f"[C] Per-query full model (like training, n={N})...")
    sampler = MultigraphNeighborSampler(num_neighbors=num_neighbors)
    preds_c = []
    for i, q in enumerate(queries):
        p = run_full_model_per_query(q["triples"], model, graph_data,
                                     eo, ro, em, rm, num_edges, sampler)
        preds_c.append(p)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}")
    preds_c = np.array(preds_c)

    # --- Results ---
    all_strats = [
        ("[A] Full-graph embeddings", preds_a),
        ("[B] Batched neighbor-sampled", preds_b),
        ("[B2] Batched + L2 norm", preds_b2),
        ("[B3] Batched + standardized", preds_b3),
        ("[C] Per-query (like training)", preds_c),
    ]
    qerrors = {name: compute_qerrors(p, ground_truths) for name, p in all_strats}

    print("\n" + "=" * 70)
    print(f"Q-Error Diagnosis ({N} queries)")
    print("=" * 70)
    for name, qe in qerrors.items():
        report_qerror(name, qe)

    print("\n  Prediction divergence vs C (log-cardinality):")
    for name, preds in all_strats[:-1]:
        mae = np.abs(preds - preds_c).mean()
        mx = np.abs(preds - preds_c).max()
        print(f"    {name}: MAE={mae:.4f}  Max={mx:.4f}")

    print("=" * 70)

    # --- Scatter plots ---
    fig, axes = plt.subplots(1, len(all_strats), figsize=(5 * len(all_strats), 5))
    gt = np.maximum(ground_truths, 1.0)
    pred_cards = {name: np.maximum(np.expm1(np.clip(p, 0, None)), 1.0)
                  for name, p in all_strats}
    all_vals = np.concatenate([gt] + list(pred_cards.values()))
    lo, hi = max(all_vals[all_vals > 0].min() * 0.5, 0.5), all_vals.max() * 2

    for ax, (name, _) in zip(axes, all_strats):
        pred = pred_cards[name]
        qe = qerrors[name]
        ax.scatter(gt, pred, alpha=0.5, s=15, edgecolors='none')
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='y=x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('True cardinality')
        ax.set_ylabel('Predicted cardinality')
        ax.set_title(f"{name}\nmed q={np.median(qe):.2f}, mean q={qe.mean():.1f}")
        ax.set_aspect('equal')
        ax.legend(fontsize=8)

    fig.suptitle(f"True vs Predicted Cardinality — {dataset_name} star queries (n={N})", y=1.02)
    fig.tight_layout()
    out_path = "/home/tim/fully-inductive-cardinality-estimation/inference/diagnose_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nScatter plot saved to {out_path}")


if __name__ == "__main__":
    main()
