"""Compare FICE inference: training-identical vs TorchScript vs C++ binary.

Runs fb237_v4 star queries (val split) through three paths:
  A) Training-identical: per-query neighbor sampling + full model (EntityGNN + RelationGNN + QueryGNN)
  B) TorchScript: precomputed .bin embeddings + scripted QueryGNN
  C) C++ binary: precomputed .bin embeddings + libtorch QueryGNN (batch mode)

Reports prediction agreement and per-query timing.


cd /Users/timschwabe/Projects/fully-inductive-cardinality-estimation/scripts && uv run python ../inference/compare.py --config ../inference/inference_config.yaml --queries
      ../data/fb237_v4_star_queries/raw/fb237_v4_star_queries --cpp-binary ../inference/build/fice_inference --output-dir ../output/comparison/ --max-queries 200

Usage:
    cd scripts  # needed for utils imports
    python ../inference/compare.py \
        --config ../inference/inference_config.yaml \
        --queries ../data/fb237_v4_star_queries/raw/fb237_v4_star_queries \
        --cpp-binary ../inference/build/fice_inference \
        --output-dir ../output/comparison/
"""

import sys
import json
import argparse
import logging
import random
import time
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data, Batch

# Ensure scripts/ is on the path for utils imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.config import Config
from utils.checkpoint_helper import load_checkpoint
from utils.data_helper import load_dataset, load_occurrences, get_or_build_graph
from utils.neighborhood_sampler import MultigraphNeighborSampler

# Import from sibling inference module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import build_fice_from_config
from diagnose_qerror import run_full_model_per_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path A: build query graph for native Python model (PyG TripleModel)
# ---------------------------------------------------------------------------

def build_query_graph_native(query, entity_emb, relation_emb,
                             entity_occ_dict, relation_occ_dict,
                             entity_mapping, relation_mapping, feat_dim):
    """Build a PyG Data query graph with precomputed embeddings injected.

    Replicates fice/datasets.py:process_query_single + FICE.forward embedding
    injection + log1p count transform.
    """
    local_entity_map = {}
    edge_index_list = []

    for triple in query["triples"]:
        h, r, t = triple[0], triple[1], triple[2]
        if h not in local_entity_map:
            local_entity_map[h] = len(local_entity_map)
        if t not in local_entity_map:
            local_entity_map[t] = len(local_entity_map)
        edge_index_list.append([local_entity_map[h], local_entity_map[t]])

    num_nodes = len(local_entity_map)
    num_edges = len(query["triples"])
    embedding_dim = feat_dim - 3

    # Node features: [counts(3) | embedding(D)]
    x = torch.zeros(num_nodes, feat_dim)
    for entity, idx in local_entity_map.items():
        if entity in entity_occ_dict:
            occ = entity_occ_dict[entity]
            x[idx, 0] = occ[0]
            x[idx, 1] = occ[1]
            x[idx, 2] = occ[2]
        if entity in entity_mapping:
            global_id = entity_mapping[entity]
            x[idx, 3:] = entity_emb[global_id]

    # Edge features: [counts(3) | embedding(D)]
    edge_attr = torch.zeros(num_edges, feat_dim)
    for i, triple in enumerate(query["triples"]):
        rel = triple[1]
        if rel in relation_occ_dict:
            occ = relation_occ_dict[rel]
            edge_attr[i, 0] = occ[0]
            edge_attr[i, 1] = occ[1]
            edge_attr[i, 2] = occ[2]
        if rel in relation_mapping:
            global_id = relation_mapping[rel]
            edge_attr[i, 3:] = relation_emb[global_id]

    # Apply log1p to counts (matches fice/models.py:1052-1053)
    x[:, :3] = torch.log1p(x[:, :3])
    edge_attr[:, :3] = torch.log1p(edge_attr[:, :3])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Path B: build query graph for TorchScript model (replicates C++ logic)
# ---------------------------------------------------------------------------

def build_query_graph_torchscript(query, entity_emb_np, relation_emb_np,
                                  entity_occ_np, relation_occ_np,
                                  uri_to_id, feat_dim, graph_num_edges):
    """Build raw tensors replicating C++ build_query_graph exactly."""
    entity_ids = uri_to_id["entities"]
    relation_ids = uri_to_id["relations"]
    embedding_dim = feat_dim - 3

    node_map = {}
    src_nodes, dst_nodes = [], []
    rel_ids = []

    for triple in query["triples"]:
        s, p, o = triple[0], triple[1], triple[2]
        # Strip angle brackets to match uri_to_id.json keys
        s_key = s.strip("<>")
        p_key = p.strip("<>")
        o_key = o.strip("<>")

        if s not in node_map:
            node_map[s] = len(node_map)
        if o not in node_map:
            node_map[o] = len(node_map)

        src_nodes.append(node_map[s])
        dst_nodes.append(node_map[o])
        rel_ids.append(relation_ids.get(p_key, -1))

    num_nodes = len(node_map)
    num_edges = len(query["triples"])

    x = torch.zeros(num_nodes, feat_dim)
    for name, local_id in node_map.items():
        is_var = name.startswith("?")
        if not is_var:
            key = name.strip("<>")
            eid = entity_ids.get(key, -1)
            if eid >= 0:
                x[local_id, :3] = torch.from_numpy(entity_occ_np[eid].copy())
                x[local_id, 3:] = torch.from_numpy(entity_emb_np[eid].copy())

    edge_attr = torch.zeros(num_edges, feat_dim)
    for e in range(num_edges):
        rid = rel_ids[e]
        if rid >= 0:
            edge_attr[e, :3] = torch.from_numpy(relation_occ_np[rid].copy())
            edge_attr[e, 3:] = torch.from_numpy(relation_emb_np[rid].copy())

    # Apply log1p to counts (matches C++ line 361-362)
    x[:, :3] = torch.log1p(x[:, :3])
    edge_attr[:, :3] = torch.log1p(edge_attr[:, :3])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    graph_size = torch.tensor([float(graph_num_edges)])

    return x, edge_index, edge_attr, batch, graph_size


# ---------------------------------------------------------------------------
# BGP file generation for C++
# ---------------------------------------------------------------------------

def write_bgp_file(queries, output_path):
    """Write queries as one BGP string per line (angle brackets preserved)."""
    with open(output_path, "w") as f:
        for q in queries:
            parts = []
            for s, p, o in q["triples"]:
                parts.append(f"{s} {p} {o}")
            f.write(" . ".join(parts) + "\n")
    logger.info(f"Wrote {len(queries)} queries to {output_path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(name, values):
    """Print summary statistics for an array of values."""
    a = np.array(values)
    print(f"  {name}:")
    print(f"    Mean: {a.mean():.4f}  Median: {np.median(a):.4f}  "
          f"Std: {a.std():.4f}")
    print(f"    Min: {a.min():.4f}  Max: {a.max():.4f}  "
          f"P95: {np.percentile(a, 95):.4f}  P99: {np.percentile(a, 99):.4f}")


def compute_q_errors(pred_log, ground_truth):
    """Compute Q-errors: max(pred/true, true/pred) in cardinality space."""
    pred_card = np.expm1(np.clip(pred_log, 0, None))
    true_card = np.array(ground_truth, dtype=np.float64)
    # Clamp to avoid division by zero
    pred_card = np.maximum(pred_card, 1.0)
    true_card = np.maximum(true_card, 1.0)
    q_errors = np.maximum(pred_card / true_card, true_card / pred_card)
    return q_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare FICE Python vs C++ inference")
    parser.add_argument("--config", type=str, default="inference/inference_config.yaml")
    parser.add_argument("--queries", type=str, required=True,
                        help="Path to star queries JSON file")
    parser.add_argument("--cpp-binary", type=str, default=None,
                        help="Path to compiled fice_inference binary (skip C++ if omitted)")
    default_out = str(Path(__file__).resolve().parent.parent / "output" / "comparison")
    parser.add_argument("--output-dir", type=str, default=default_out)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Number of warmup queries before timing")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        inf_cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(inf_cfg["output_dir"])

    # ---- Load queries (same val split as diagnose_qerror.py) ----
    logger.info(f"Loading queries from {args.queries}")
    with open(args.queries) as f:
        raw_queries = json.load(f)

    num_queries = 20000
    random.seed(42)
    indices = list(range(num_queries))
    random.shuffle(indices)
    val_indices = indices[int(num_queries * 0.8):]
    all_queries = [raw_queries[i] for i in val_indices]
    if args.max_queries > 0:
        all_queries = all_queries[:args.max_queries]
    logger.info(f"Loaded {len(all_queries)} val queries (same split as diagnose_qerror.py)")

    # ---- Load FICE model (Path A: training-identical) ----
    logger.info("Loading FICE model from checkpoint...")
    checkpoint_path = Path(inf_cfg["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path)
    config = Config(**checkpoint["config"])

    model = build_fice_from_config(config, state_dict=checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load graph
    graph_dataset_name = inf_cfg["graph_dataset"]
    graph = load_dataset(graph_dataset_name)
    graph_num_edges = int(graph.num_edges)

    relation_source = config.relation_gnn.source
    graph_type = "factor" if relation_source == "factor" else "bipartite"
    graph_data = get_or_build_graph(graph, graph_dataset_name, graph_type)

    # Load occurrence dicts
    entity_occ_dict, relation_occ_dict, entity_mapping, relation_mapping = \
        load_occurrences(graph_dataset_name)

    num_neighbors = config.GNNNeighborhood or [100, 100]
    sampler = MultigraphNeighborSampler(num_neighbors=num_neighbors)
    feat_dim = config.embedding_dim + 3

    # ---- Load TorchScript model + precomputed artifacts (Path B) ----
    logger.info("Loading TorchScript model and precomputed artifacts...")
    scripted_model = torch.jit.load(str(artifacts_dir / "query_gnn_scripted.pt"))
    scripted_model.eval()

    with open(artifacts_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(artifacts_dir / "uri_to_id.json") as f:
        uri_to_id = json.load(f)

    num_entities = metadata["num_entities"]
    num_relations = metadata["num_relations"]
    embedding_dim = metadata["embedding_dim"]

    entity_emb_np = np.fromfile(
        str(artifacts_dir / "entity_embeddings.bin"), dtype=np.float32
    ).reshape(num_entities, embedding_dim)
    relation_emb_np = np.fromfile(
        str(artifacts_dir / "relation_embeddings.bin"), dtype=np.float32
    ).reshape(num_relations, embedding_dim)
    entity_occ_np = np.fromfile(
        str(artifacts_dir / "entity_occurrences.bin"), dtype=np.float32
    ).reshape(num_entities, 3)
    relation_occ_np = np.fromfile(
        str(artifacts_dir / "relation_occurrences.bin"), dtype=np.float32
    ).reshape(num_relations, 3)

    # ---- Warmup TorchScript ----
    logger.info(f"Warming up TorchScript with {args.warmup} queries...")
    warmup_n = min(args.warmup, len(all_queries))
    with torch.no_grad():
        for q in all_queries[:warmup_n]:
            x, ei, ea, bt, gs = build_query_graph_torchscript(
                q, entity_emb_np, relation_emb_np,
                entity_occ_np, relation_occ_np,
                uri_to_id, feat_dim, graph_num_edges)
            scripted_model(x, ei, ea, bt, gs)

    # ---- Run Path A (training-identical) and Path B (TorchScript) ----
    logger.info(f"Running {len(all_queries)} queries...")
    native_preds = []
    ts_preds = []
    native_times = []
    ts_times = []
    ground_truths = []

    for i, q in enumerate(all_queries):
        ground_truths.append(q["y"])

        # Path A: training-identical (per-query full model)
        t0 = time.perf_counter()
        out_a = run_full_model_per_query(
            q["triples"], model, graph_data,
            entity_occ_dict, relation_occ_dict,
            entity_mapping, relation_mapping,
            graph_num_edges, sampler)
        t1 = time.perf_counter()

        native_preds.append(out_a)
        native_times.append((t1 - t0) * 1e6)

        # Path B: TorchScript (precomputed embeddings)
        with torch.no_grad():
            x, ei, ea, bt, gs = build_query_graph_torchscript(
                q, entity_emb_np, relation_emb_np,
                entity_occ_np, relation_occ_np,
                uri_to_id, feat_dim, graph_num_edges)

            t0 = time.perf_counter()
            out_b = scripted_model(x, ei, ea, bt, gs)
            t1 = time.perf_counter()

        ts_preds.append(out_b.item())
        ts_times.append((t1 - t0) * 1e6)

        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(all_queries)}")

    native_preds = np.array(native_preds)
    ts_preds = np.array(ts_preds)

    # ---- Run Path C: C++ binary (optional) ----
    cpp_preds = None
    cpp_times = None
    if args.cpp_binary:
        bgp_path = output_dir / "queries.bgp"
        write_bgp_file(all_queries, bgp_path)

        logger.info(f"Running C++ binary on {len(all_queries)} queries...")
        result = subprocess.run(
            [args.cpp_binary, str(artifacts_dir), "--batch", str(bgp_path)],
            capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            logger.error(f"C++ binary failed:\n{result.stderr}")
        else:
            logger.info(f"C++ stderr: {result.stderr.strip()}")
            cpp_preds_list = []
            cpp_times_list = []
            for line in result.stdout.strip().split("\n"):
                if line.startswith("query_idx"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 6:
                    # New format: idx, num_triples, log_card, build_us, infer_us, total_us
                    cpp_preds_list.append(float(parts[2]))
                    cpp_times_list.append(float(parts[4]))  # inference_us
                elif len(parts) >= 3:
                    # Old format fallback: idx, log_card, infer_us
                    cpp_preds_list.append(float(parts[1]))
                    cpp_times_list.append(float(parts[2]))
            cpp_preds = np.array(cpp_preds_list)
            cpp_times = np.array(cpp_times_list)
            logger.info(f"Parsed {len(cpp_preds)} C++ predictions")

    # ---- Report ----
    print("\n" + "=" * 70)
    print("FICE Inference Comparison Results")
    print(f"Queries: {len(all_queries)}  |  Graph: fb237_v4 ({graph_num_edges} edges)")
    print("=" * 70)

    # Prediction agreement
    diff_ab = np.abs(native_preds - ts_preds)
    print("\n--- Prediction Agreement (log-cardinality space) ---")
    print(f"\nTraining-identical vs TorchScript:")
    print(f"  MAE: {diff_ab.mean():.6e}  Max: {diff_ab.max():.6e}  "
          f"Correlation: {np.corrcoef(native_preds, ts_preds)[0,1]:.8f}")

    if cpp_preds is not None and len(cpp_preds) == len(native_preds):
        diff_ac = np.abs(native_preds - cpp_preds)
        diff_bc = np.abs(ts_preds - cpp_preds)
        print(f"\nTraining-identical vs C++:")
        print(f"  MAE: {diff_ac.mean():.6e}  Max: {diff_ac.max():.6e}  "
              f"Correlation: {np.corrcoef(native_preds, cpp_preds)[0,1]:.8f}")
        print(f"\nTorchScript vs C++ (should be ~0):")
        print(f"  MAE: {diff_bc.mean():.6e}  Max: {diff_bc.max():.6e}  "
              f"Queries with diff > 1e-4: {(diff_bc > 1e-4).sum()}")

    # Q-error vs ground truth
    print("\n--- Q-Error vs Ground Truth ---")
    for name, preds in [("Training-identical", native_preds), ("TorchScript", ts_preds)]:
        q_errors = compute_q_errors(preds, ground_truths)
        print(f"\n{name}:")
        print(f"  Mean: {q_errors.mean():.2f}  Median: {np.median(q_errors):.2f}")
        print(f"  P90: {np.percentile(q_errors, 90):.2f}  "
              f"P95: {np.percentile(q_errors, 95):.2f}  "
              f"P99: {np.percentile(q_errors, 99):.2f}  "
              f"Max: {q_errors.max():.2f}")

    if cpp_preds is not None and len(cpp_preds) == len(native_preds):
        q_errors = compute_q_errors(cpp_preds, ground_truths)
        print(f"\nC++:")
        print(f"  Mean: {q_errors.mean():.2f}  Median: {np.median(q_errors):.2f}")
        print(f"  P90: {np.percentile(q_errors, 90):.2f}  "
              f"P95: {np.percentile(q_errors, 95):.2f}  "
              f"P99: {np.percentile(q_errors, 99):.2f}  "
              f"Max: {q_errors.max():.2f}")

    # Timing
    print("\n--- Per-Query Inference Time (microseconds) ---")
    print_stats("Training-identical (full model)", native_times)
    print_stats("TorchScript (Python)", ts_times)

    if cpp_times is not None and len(cpp_times) > 0:
        print_stats("C++ (libtorch)", cpp_times)
        print(f"\n  Speedup C++ vs Training-identical: "
              f"{np.mean(native_times) / np.mean(cpp_times):.2f}x")
        print(f"  Speedup C++ vs TorchScript:   "
              f"{np.mean(ts_times) / np.mean(cpp_times):.2f}x")

    # ---- Write per-query TSV ----
    tsv_path = output_dir / "compare_per_query.tsv"
    with open(tsv_path, "w") as f:
        header = "query_idx\tground_truth\ttraining_identical_log_pred\tts_log_pred"
        header += "\ttraining_identical_time_us\tts_time_us"
        if cpp_preds is not None and len(cpp_preds) == len(native_preds):
            header += "\tcpp_log_pred\tcpp_time_us"
        f.write(header + "\n")

        for i in range(len(all_queries)):
            row = f"{i}\t{ground_truths[i]}\t{native_preds[i]:.6f}\t{ts_preds[i]:.6f}"
            row += f"\t{native_times[i]:.1f}\t{ts_times[i]:.1f}"
            if cpp_preds is not None and i < len(cpp_preds):
                row += f"\t{cpp_preds[i]:.6f}\t{cpp_times[i]:.1f}"
            f.write(row + "\n")

    logger.info(f"Per-query results written to {tsv_path}")
    print("\n" + "=" * 70)

    # ---- Scatter plots: true vs predicted cardinality ----
    gt = np.maximum(np.array(ground_truths, dtype=np.float64), 1.0)

    methods = [
        ("Training-identical\n(per-query emb)", native_preds),
        ("TorchScript\n(batched emb)", ts_preds),
    ]
    if cpp_preds is not None and len(cpp_preds) == len(native_preds):
        methods.append(("C++ libtorch\n(batched emb)", cpp_preds))

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    all_pred_cards = []
    for name, preds in methods:
        pc = np.maximum(np.expm1(np.clip(preds, 0, None)), 1.0)
        all_pred_cards.append(pc)

    all_vals = np.concatenate([gt] + all_pred_cards)
    lo = max(all_vals[all_vals > 0].min() * 0.5, 0.5)
    hi = all_vals.max() * 2

    for ax, (name, preds), pred_card in zip(axes, methods, all_pred_cards):
        qe = compute_q_errors(preds, ground_truths)
        ax.scatter(gt, pred_card, alpha=0.5, s=15, edgecolors='none')
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

    n = len(all_queries)
    fig.suptitle(f"True vs Predicted — {graph_dataset_name} star queries (n={n})", y=1.02)
    fig.tight_layout()
    scatter_path = output_dir / "compare_scatter.png"
    fig.savefig(str(scatter_path), dpi=150, bbox_inches='tight')
    print(f"\nScatter plot saved to {scatter_path}")


if __name__ == "__main__":
    main()
