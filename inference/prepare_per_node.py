"""FICE inference preparation for per_node-mode checkpoints (GT+PPR encoder).

Counterpart to prepare.py for models trained with
`neighborhood_mode: per_node`: every entity/relation gets its embedding from
its own deterministic per-seed subgraph (PPR top-cap selection, cached RWPE,
optional true degree/fan-out features) via forward_per_node — exactly the
training/eval code path. The query GNN (decoder) is unchanged, so the exported
artifacts are drop-in compatible with the existing C++/QLever plugin.

Requires conditioning='false' (embeddings must be query-independent) and
deterministic per-seed sampling (selection 'ppr', or pool_size == mc_samples).

Usage:
    python inference/prepare_per_node.py \
        --checkpoint output/run_2026-07-03_11-38-39/checkpoints/best_model.pt \
        --graph-dataset dbpedia100k \
        --output-dir output/fice_prepared_gtpprdeg/ \
        --chunk-size 256

Outputs: same files as prepare.py (entity/relation embeddings + occurrences
as raw float32 .bin, uri_to_id.json, metadata.json, scripted/traced query GNN).
"""

import sys
import json
import time
import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config
from utils.checkpoint_helper import load_checkpoint
from utils.data_helper import load_dataset, load_occurrences, get_or_build_graph
from utils.subgraph_provider import PerNodeSubgraphProvider

from prepare import (
    build_occurrence_tensors,
    export_query_gnn,
    save_binary,
)

from fice.models import FICE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_model(config: Config) -> FICE:
    """Mirror train.py initialize_model_components for model='FICE'.

    Unlike prepare.py's build_fice_from_config, this passes the GPS options
    and extra_node_feats, and trusts config.relation_gnn.role_aware_degrees
    (the weight-shape inference there miscounts when degree features add 5
    input columns).
    """
    entity_graph_mode = (
        "bipartite" if config.entity_gnn.source in ("bipartite", "factor")
        else "KG"
    )
    pe_dim = config.random_walk_pe
    if pe_dim > 0 and config.relation_gnn.source == "factor":
        pe_dim *= 3

    return FICE(
        vector_dim=config.embedding_dim,
        n_embedding_layers=config.n_embedding_layers,
        conditioning=config.Conditioning,
        entity_graph_mode=entity_graph_mode,
        relation_graph_mode="bipartite",
        entity_message_func=config.entity_gnn.message_func,
        entity_layer_norm=config.entity_gnn.layer_norm,
        entity_skip_connection=config.entity_gnn.skip_connection,
        entity_dropout=config.entity_gnn.dropout,
        relation_message_func=config.relation_gnn.message_func,
        relation_layer_norm=config.relation_gnn.layer_norm,
        relation_skip_connection=config.relation_gnn.skip_connection,
        relation_dropout=config.relation_gnn.dropout,
        relation_attention=config.relation_gnn.relation_attention,
        relational=config.relation_gnn.relational,
        role_aware_degrees=config.relation_gnn.role_aware_degrees,
        relation_heads=config.relation_gnn.heads,
        relation_attn_type=config.relation_gnn.attn_type,
        relation_local_mp=config.relation_gnn.local_mp,
        extra_node_feats=(5 if config.degree_features else 0),
        query_gnn_type=config.query_gnn.model,
        query_message_func=config.query_gnn.message_func,
        query_layer_norm=config.query_gnn.layer_norm,
        query_skip_connection=config.query_gnn.skip_connection,
        query_gnn_dropout=config.query_gnn.dropout,
        query_gnn_num_layers=config.query_gnn.num_layers,
        query_gnn_readout=config.query_gnn.readout,
        query_gnn_message_activation=config.query_gnn.message_activation,
        pe_dim=pe_dim,
        prediction_mode=config.prediction_mode,
        embedding_space=config.embedding_space,
        count_features=config.count_features,
        disable_embedding_gnn=config.disable_embedding_gnn,
    )


def compute_embeddings_per_node(model: FICE, graph_data, config: Config,
                                device: torch.device, chunk_size: int):
    """Embed all entity and relation nodes through their per-seed subgraphs.

    Factor-graph node layout is [entities | relations | triples]; seeds are
    the first num_entities + num_relations ids. mc_samples > 1 averages the
    fixed first M pool neighborhoods (eval semantics, training=False).
    """
    provider = PerNodeSubgraphProvider(
        graph_data,
        caps=config.GNNNeighborhood,
        walk_length=config.random_walk_pe,
        role_aware=(config.relation_gnn.source == "factor"),
        pool_size=config.mc_pool_size,
        selection=config.neighbor_selection,
        ppr_alpha=config.ppr_alpha,
        degree_features=config.degree_features,
    )

    num_entities = graph_data.original_num_nodes
    num_relations = graph_data.original_num_relations
    total = num_entities + num_relations
    M = config.mc_samples

    out = torch.zeros(total, model.vector_dim)
    model.eval()
    t0 = time.time()
    done = 0
    for i in range(0, total, chunk_size):
        seeds = torch.arange(i, min(i + chunk_size, total))
        data = provider.build_batch(seeds, mc_samples=M, training=False)
        data = data.to(device)
        with torch.no_grad():
            emb = model.embedding_model.forward_per_node(data)  # (S*M, D)
        if M > 1:
            emb = emb.view(-1, M, emb.shape[-1]).mean(dim=1)
        out[seeds] = emb.cpu().float()
        # Seeds are visited once; keeping cached subgraphs would grow to GBs.
        provider.cache.clear()
        done += seeds.numel()
        elapsed = time.time() - t0
        rate = done / elapsed
        logger.info(f"  {done}/{total} seeds ({rate:.0f}/s, "
                    f"ETA {(total - done) / rate:.0f}s)")

    entity_emb = out[:num_entities]
    relation_emb = out[num_entities:total]
    return entity_emb, relation_emb


# ---------------------------------------------------------------------------
# Parallel embedding computation (spawn workers over memmapped CSR staging)
# ---------------------------------------------------------------------------

def _provider_from_staging(staging: Path, config: Config) -> PerNodeSubgraphProvider:
    """Reconstruct a read-only provider from parent-staged memmapped arrays,
    skipping the expensive CSR build in __init__."""
    params = json.loads((staging / "params.json").read_text())
    prov = PerNodeSubgraphProvider.__new__(PerNodeSubgraphProvider)
    prov.caps = tuple(int(c) for c in config.GNNNeighborhood)
    prov.walk_length = config.random_walk_pe
    prov.role_aware = params["role_aware"]
    prov.pool_size = config.mc_pool_size
    prov.selection = config.neighbor_selection
    prov.ppr_alpha = config.ppr_alpha
    prov._ppr_cache = {}
    prov._ppr_p = None
    prov._ppr_r = None
    prov.num_nodes = params["num_nodes"]
    prov.n_entities = params["n_entities"]
    prov.n_relations = params["n_relations"]
    prov.indices = np.load(staging / "indices.npy", mmap_mode="r")
    prov.edge_counts = np.load(staging / "edge_counts.npy", mmap_mode="r")
    prov.edge_roles = np.load(staging / "edge_roles.npy", mmap_mode="r")
    prov.indptr = np.load(staging / "indptr.npy", mmap_mode="r")
    prov.cache = {}
    prov._train_rng = np.random.default_rng()
    prov._node_role_deg = (np.load(staging / "node_role_deg.npy", mmap_mode="r")
                           if (staging / "node_role_deg.npy").exists() else None)
    prov._rel_fanout = (np.load(staging / "rel_fanout.npy", mmap_mode="r")
                        if (staging / "rel_fanout.npy").exists() else None)
    return prov


def _worker_run(worker_id: int, staging: str, checkpoint_path: str,
                out_path: str, chunk_size: int, total: int, emb_dim: int):
    torch.set_num_threads(1)
    staging = Path(staging)
    checkpoint = load_checkpoint(Path(checkpoint_path))
    config = Config(**checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prov = _provider_from_staging(staging, config)
    seeds_all = np.load(staging / f"seeds_w{worker_id}.npy")
    out = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(total, emb_dim))
    M = config.mc_samples
    t0 = time.time()
    for i in range(0, len(seeds_all), chunk_size):
        seeds = torch.from_numpy(seeds_all[i:i + chunk_size].copy())
        data = prov.build_batch(seeds, mc_samples=M, training=False)
        with torch.no_grad():
            emb = model.embedding_model.forward_per_node(data)
        if M > 1:
            emb = emb.view(-1, M, emb.shape[-1]).mean(dim=1)
        out[seeds.numpy()] = emb.numpy().astype(np.float32)
        prov.cache.clear()
        done = i + len(seeds)
        if (i // chunk_size) % 20 == 0 or done == len(seeds_all):
            rate = done / (time.time() - t0)
            print(f"[w{worker_id}] {done}/{len(seeds_all)} ({rate:.0f}/s)",
                  flush=True)
    out.flush()
    return worker_id


def compute_embeddings_parallel(graph_data, config: Config, checkpoint_path: str,
                                workers: int, chunk_size: int, staging: Path):
    """Same result as compute_embeddings_per_node, computed by `workers`
    spawned processes over a shared CSR staging area. Deterministic:
    every seed's subgraph depends only on (seed id, sample idx)."""
    staging.mkdir(parents=True, exist_ok=True)
    logger.info("Building CSR staging arrays (one-time)...")
    provider = PerNodeSubgraphProvider(
        graph_data,
        caps=config.GNNNeighborhood,
        walk_length=config.random_walk_pe,
        role_aware=(config.relation_gnn.source == "factor"),
        pool_size=config.mc_pool_size,
        selection=config.neighbor_selection,
        ppr_alpha=config.ppr_alpha,
        degree_features=config.degree_features,
    )
    np.save(staging / "indices.npy", provider.indices)
    np.save(staging / "edge_counts.npy", provider.edge_counts)
    np.save(staging / "edge_roles.npy", provider.edge_roles)
    np.save(staging / "indptr.npy", provider.indptr)
    if provider._node_role_deg is not None:
        np.save(staging / "node_role_deg.npy", provider._node_role_deg)
        np.save(staging / "rel_fanout.npy", provider._rel_fanout)
    n_ent, n_rel = provider.n_entities, provider.n_relations
    (staging / "params.json").write_text(json.dumps({
        "num_nodes": provider.num_nodes,
        "n_entities": n_ent,
        "n_relations": n_rel,
        "role_aware": provider.role_aware,
    }))
    del provider

    total = n_ent + n_rel
    # Relation seeds are the extreme hubs (slowest PPR by far) — deal them
    # round-robin across workers instead of leaving them in one stripe.
    ent_stripes = np.array_split(np.arange(n_ent, dtype=np.int64), workers)
    rel_ids = np.arange(n_ent, total, dtype=np.int64)
    for w in range(workers):
        seeds_w = np.concatenate([rel_ids[w::workers], ent_stripes[w]])
        np.save(staging / f"seeds_w{w}.npy", seeds_w)

    out_path = staging / "embeddings.f32"
    out = np.memmap(out_path, dtype=np.float32, mode="w+",
                    shape=(total, config.embedding_dim))
    del out

    logger.info(f"Spawning {workers} workers over {total} seeds...")
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker_run, w, str(staging), str(checkpoint_path),
                          str(out_path), chunk_size, total,
                          config.embedding_dim)
                for w in range(workers)]
        for f in as_completed(futs):
            logger.info(f"worker {f.result()} finished "
                        f"({time.time() - t0:.0f}s elapsed)")

    emb = torch.from_numpy(np.fromfile(out_path, dtype=np.float32)
                           .reshape(total, config.embedding_dim))
    return emb[:n_ent].clone(), emb[n_ent:total].clone()


def main():
    parser = argparse.ArgumentParser(
        description="FICE inference preparation (per_node checkpoints)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (CPU only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # ---- 1. Load checkpoint and reconstruct model ----
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(Path(args.checkpoint))
    config = Config(**checkpoint["config"])

    if config.neighborhood_mode != "per_node":
        raise ValueError(
            f"This script is for per_node checkpoints; got "
            f"neighborhood_mode='{config.neighborhood_mode}' — use prepare.py."
        )
    if config.Conditioning != "false":
        raise ValueError(
            "Precomputed embeddings require conditioning='false', got "
            f"'{config.Conditioning}'."
        )
    if config.model != "FICE":
        raise ValueError(f"Inference only supports FICE, got {config.model}")
    if config.neighbor_selection != "ppr" and config.mc_pool_size > config.mc_samples:
        raise ValueError(
            "Stochastic selection with pool_size > mc_samples makes eval "
            "embeddings depend on the pool subset; refusing to precompute."
        )

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # ---- 2. Load graph, build factor/bipartite graph ----
    logger.info(f"Loading graph dataset: {args.graph_dataset}")
    graph = load_dataset(args.graph_dataset)
    source = "factor" if config.relation_gnn.source == "factor" else "bipartite"
    graph_data = get_or_build_graph(graph, args.graph_dataset, source)

    num_entities = graph_data.original_num_nodes
    num_relations = graph_data.original_num_relations
    logger.info(f"Graph: {num_entities} entities, {num_relations} relations, "
                f"{graph.num_edges} edges")

    # ---- 3. Per-node embeddings ----
    logger.info(f"Computing per-node embeddings (caps={config.GNNNeighborhood}, "
                f"selection={config.neighbor_selection}, mc_samples={config.mc_samples}, "
                f"degree_features={config.degree_features}, workers={args.workers})...")
    if args.workers > 1:
        entity_emb, relation_emb = compute_embeddings_parallel(
            graph_data, config, args.checkpoint, args.workers,
            args.chunk_size, output_dir / "_staging")
    else:
        entity_emb, relation_emb = compute_embeddings_per_node(
            model, graph_data, config, device, args.chunk_size)

    # ---- 4. Occurrences, mappings, metadata, decoder export (as prepare.py) ----
    entity_occ_dict, relation_occ_dict, entity_mapping, relation_mapping = \
        load_occurrences(args.graph_dataset)
    entity_occ, relation_occ = build_occurrence_tensors(
        entity_occ_dict, relation_occ_dict, entity_mapping, relation_mapping)

    save_binary(entity_emb, output_dir / "entity_embeddings.bin")
    save_binary(relation_emb, output_dir / "relation_embeddings.bin")
    save_binary(entity_occ, output_dir / "entity_occurrences.bin")
    save_binary(relation_occ, output_dir / "relation_occurrences.bin")

    def _strip_brackets(mapping: dict) -> dict:
        return {k.strip("<>"): v for k, v in mapping.items()}

    with open(output_dir / "uri_to_id.json", "w") as f:
        json.dump({
            "entities": _strip_brackets(entity_mapping),
            "relations": _strip_brackets(relation_mapping),
        }, f, indent=2)
    logger.info(f"Saved {output_dir / 'uri_to_id.json'}")

    metadata = {
        "embedding_dim": config.embedding_dim,
        "feat_dim": config.embedding_dim + 3,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "graph_num_edges": int(graph.num_edges),
        "query_gnn_type": config.query_gnn.model,
        "query_gnn_num_layers": config.query_gnn.num_layers,
        "query_gnn_readout": config.query_gnn.readout,
        "query_gnn_layer_norm": config.query_gnn.layer_norm,
        "query_gnn_skip_connection": config.query_gnn.skip_connection,
        "prediction_mode": config.prediction_mode,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {output_dir / 'metadata.json'}")

    # Encoder provenance in a sibling file: the C++ plugin reads metadata.json
    # with a minimal JSON parser that cannot handle nested objects.
    encoder_info = {
        "neighborhood_mode": config.neighborhood_mode,
        "neighbor_selection": config.neighbor_selection,
        "caps": list(config.GNNNeighborhood),
        "mc_samples": config.mc_samples,
        "degree_features": config.degree_features,
        "checkpoint": str(args.checkpoint),
    }
    with open(output_dir / "encoder_info.json", "w") as f:
        json.dump(encoder_info, f, indent=2)

    logger.info("Exporting query GNN to TorchScript...")
    model.cpu()
    export_query_gnn(model, config, output_dir)

    logger.info(f"Done! All artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
