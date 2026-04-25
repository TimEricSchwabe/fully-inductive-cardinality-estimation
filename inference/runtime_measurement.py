"""Per-query latency benchmark for FICE cardinality estimation on dbpedia100k.

Loads the TorchScript query GNN and precomputed (mmap'd) entity/relation
embeddings, iterates every query across all 8 dbpedia100k query sets, and
records a CSV of (n_triple_pattern, runtime) pairs. Runtime is measured in
microseconds from "query dict in hand" through building the query graph
tensors and running the scripted forward pass to cardinality.

Usage:
    cd scripts
    uv run python ../inference/runtime_measurement.py \\
        --artifacts-dir ../output/inference_dbpedia100k/ \\
        --queries-root ../data/ \\
        --output ../output/runtime_measurement_dbpedia100k.csv
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


torch.set_num_threads(1)
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


QUERY_FILES = [
    ("dbpedia100k_star_queries",            "59687708"),
    ("dbpedia100k_path_queries",            "59898644"),
    ("dbpedia100k_cycle_queries",           "queries.json"),
    ("dbpedia100k_diamond_queries",         "queries.json"),
    ("dbpedia100k_flower_queries",          "queries.json"),
    ("dbpedia100k_snowflake_queries",       "queries.json"),
    ("dbpedia100k_path_star_ends_queries",  "queries.json"),
    ("dbpedia100k_binary_tree_queries",     "queries.json"),
]


def build_query_graph_torchscript(query, entity_emb_np, relation_emb_np,
                                  entity_occ_np, relation_occ_np,
                                  uri_to_id, feat_dim, graph_num_edges):
    """Build raw tensors replicating C++ build_query_graph exactly.

    Copied from inference/compare.py:116 so this script does not drag in
    compare.py's broken import of diagnose_qerror.
    """
    entity_ids = uri_to_id["entities"]
    relation_ids = uri_to_id["relations"]

    node_map = {}
    src_nodes, dst_nodes = [], []
    rel_ids = []

    for triple in query["triples"]:
        s, p, o = triple[0], triple[1], triple[2]
        p_key = p.strip("<>")

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
        if not name.startswith("?"):
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

    x[:, :3] = torch.log1p(x[:, :3])
    edge_attr[:, :3] = torch.log1p(edge_attr[:, :3])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    graph_size = torch.tensor([float(graph_num_edges)])

    return x, edge_index, edge_attr, batch, graph_size


def load_artifacts(artifacts_dir: Path):
    with open(artifacts_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(artifacts_dir / "uri_to_id.json") as f:
        uri_to_id = json.load(f)

    num_entities = metadata["num_entities"]
    num_relations = metadata["num_relations"]
    embedding_dim = metadata["embedding_dim"]
    feat_dim = metadata["feat_dim"]
    graph_num_edges = metadata["graph_num_edges"]

    entity_emb = np.fromfile(
        artifacts_dir / "entity_embeddings.bin", dtype=np.float32
    ).reshape(num_entities, embedding_dim)
    relation_emb = np.fromfile(
        artifacts_dir / "relation_embeddings.bin", dtype=np.float32
    ).reshape(num_relations, embedding_dim)
    entity_occ = np.fromfile(
        artifacts_dir / "entity_occurrences.bin", dtype=np.float32
    ).reshape(num_entities, 3)
    relation_occ = np.fromfile(
        artifacts_dir / "relation_occurrences.bin", dtype=np.float32
    ).reshape(num_relations, 3)

    scripted_model = torch.jit.load(str(artifacts_dir / "query_gnn_scripted.pt"))
    scripted_model.eval()

    return (scripted_model, entity_emb, relation_emb, entity_occ, relation_occ,
            uri_to_id, feat_dim, graph_num_edges)


def load_all_queries(queries_root: Path) -> List[Tuple[str, dict]]:
    """Return list of (query_set_name, query_dict) across all 8 dbpedia100k sets."""
    pairs = []
    for set_dir, filename in QUERY_FILES:
        path = queries_root / set_dir / "raw" / filename
        if not path.exists():
            logger.warning(f"Skipping missing query file: {path}")
            continue
        with open(path) as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries):>6} queries from {set_dir}")
        for q in queries:
            pairs.append((set_dir, q))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=str, required=True,
                        help="Directory produced by inference/prepare.py "
                             "(contains .bin files and query_gnn_scripted.pt)")
    parser.add_argument("--queries-root", type=str, required=True,
                        help="Directory containing dbpedia100k_*_queries/raw/... "
                             "(typically the repo's data/ folder)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Number of warmup queries before timing (default 100)")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    queries_root = Path(args.queries_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading artifacts from {artifacts_dir}")
    (scripted_model, entity_emb, relation_emb, entity_occ, relation_occ,
     uri_to_id, feat_dim, graph_num_edges) = load_artifacts(artifacts_dir)
    logger.info(f"feat_dim={feat_dim}, graph_num_edges={graph_num_edges}, "
                f"entities={entity_emb.shape[0]}, relations={relation_emb.shape[0]}")

    pairs = load_all_queries(queries_root)
    if not pairs:
        logger.error("No queries loaded; aborting.")
        sys.exit(1)
    total = len(pairs)
    logger.info(f"Total queries across all sets: {total}")

    # Warmup
    warmup_n = min(args.warmup, total)
    logger.info(f"Warming up with {warmup_n} queries...")
    for _, q in pairs[:warmup_n]:
        x, ei, ea, bt, gs = build_query_graph_torchscript(
            q, entity_emb, relation_emb, entity_occ, relation_occ,
            uri_to_id, feat_dim, graph_num_edges)
        _ = scripted_model(x, ei, ea, bt, gs).item()

    # Timed loop
    logger.info("Starting timed measurement...")
    runtimes_us = np.empty(total, dtype=np.float64)
    n_triples = np.empty(total, dtype=np.int64)

    for i, (_, q) in enumerate(pairs):
        t0 = time.perf_counter()
        x, ei, ea, bt, gs = build_query_graph_torchscript(
            q, entity_emb, relation_emb, entity_occ, relation_occ,
            uri_to_id, feat_dim, graph_num_edges)
        _ = scripted_model(x, ei, ea, bt, gs).item()
        t1 = time.perf_counter()

        runtimes_us[i] = (t1 - t0) * 1e6
        n_triples[i] = len(q["triples"])

        if (i + 1) % 10000 == 0:
            logger.info(f"  {i + 1}/{total} queries measured "
                        f"(median so far: {np.median(runtimes_us[:i+1]):.1f} us)")

    # Write CSV
    logger.info(f"Writing CSV to {output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_triple_pattern", "runtime"])
        for n, r in zip(n_triples, runtimes_us):
            writer.writerow([int(n), f"{r:.3f}"])

    # Summary
    logger.info(f"Rows written: {total}")
    logger.info(f"Runtime (us) — mean: {runtimes_us.mean():.1f}  "
                f"median: {np.median(runtimes_us):.1f}  "
                f"p95: {np.percentile(runtimes_us, 95):.1f}  "
                f"p99: {np.percentile(runtimes_us, 99):.1f}  "
                f"max: {runtimes_us.max():.1f}")


if __name__ == "__main__":
    main()
