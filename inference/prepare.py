"""FICE Inference Preparation Script.

Loads a trained FICE checkpoint and a knowledge graph, computes embeddings for
all entities and relations, exports the query GNN to TorchScript, and saves
everything needed for C++ inference via libtorch.

Usage:
    cd scripts  # needed for utils imports
    python ../inference/prepare.py --config ../inference/inference_config.yaml

Outputs (written to output_dir):
    entity_embeddings.bin     — float32, shape (num_entities, embedding_dim)
    relation_embeddings.bin   — float32, shape (num_relations, embedding_dim)
    entity_occurrences.bin    — float32, shape (num_entities, 3)
    relation_occurrences.bin  — float32, shape (num_relations, 3)
    uri_to_id.json            — entity/relation URI→ID mappings
    metadata.json             — graph and model metadata
    query_gnn_scripted.pt     — TorchScript model (via torch.jit.script)
    query_gnn_traced.pt       — TorchScript model (via torch.jit.trace)
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import yaml

# Ensure scripts/ is on the path so we can import utils
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.config import Config
from utils.checkpoint_helper import load_checkpoint
from utils.data_helper import (
    load_dataset,
    load_occurrences,
    get_or_build_graph,
)
from utils.neighborhood_sampler import MultigraphNeighborSampler

# Import after path setup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scriptable_query_gnn import (
    ScriptableTripleModel,
    build_scriptable_from_config,
    transfer_weights,
)

from fice.models import FICE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model reconstruction (mirrors train.py initialize_model_components)
# ---------------------------------------------------------------------------

def _infer_role_aware_degrees(config: Config, state_dict: dict) -> bool:
    """Infer role_aware_degrees from checkpoint weights when config is ambiguous.

    Old checkpoints lack role_aware_degrees in their config, so Pydantic fills
    the default (True). But the actual model may have been trained with False.
    We detect this from the feature_encoder input dimension:
      - role_aware_degrees=True  → degree_dim=6 → raw_in = 1+6+pe+1 = 8+pe
      - role_aware_degrees=False → degree_dim=1 → raw_in = 1+1+pe+1 = 3+pe
    """
    key = "embedding_model.feature_encoder.weight"
    if key not in state_dict:
        return config.relation_gnn.role_aware_degrees

    raw_in_dim = state_dict[key].shape[1]
    pe_dim = config.random_walk_pe
    if pe_dim > 0 and config.relation_gnn.source == "factor":
        pe_dim *= 3

    # raw_in = base(1) + degree_dim + pe_dim + seed(1)
    degree_dim = raw_in_dim - 1 - pe_dim - 1
    return degree_dim == 6


def build_fice_from_config(config: Config, state_dict: dict = None) -> FICE:
    """Reconstruct FICE model from a Config object (no weights loaded).

    If state_dict is provided, uses it to resolve ambiguous config defaults
    (e.g. role_aware_degrees missing from old checkpoints).
    """
    entity_graph_mode = (
        "bipartite" if config.entity_gnn.source in ("bipartite", "factor")
        else "KG"
    )
    pe_dim = config.random_walk_pe
    if pe_dim > 0 and config.relation_gnn.source == "factor":
        pe_dim *= 3

    role_aware_degrees = config.relation_gnn.role_aware_degrees
    if state_dict is not None:
        role_aware_degrees = _infer_role_aware_degrees(config, state_dict)
        if role_aware_degrees != config.relation_gnn.role_aware_degrees:
            logger.info(f"Inferred role_aware_degrees={role_aware_degrees} "
                        f"from checkpoint weights (config default was "
                        f"{config.relation_gnn.role_aware_degrees})")

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
        role_aware_degrees=role_aware_degrees,
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
        count_features=getattr(config, 'count_features', 'raw'),
    )


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings_full_graph(model: FICE, graph_data, device: torch.device):
    """Compute embeddings for ALL nodes by running the full graph at once.

    Returns: (entity_embeddings, relation_embeddings) as CPU tensors.
    """
    model.eval()
    embedding_model = model.embedding_model
    graph_data = graph_data.to(device)

    with torch.no_grad():
        # batch=None → bs=1, no conditioning
        all_emb = embedding_model(data=graph_data, batch=None)  # (1, N, D)

    all_emb = all_emb.squeeze(0).cpu()  # (N, D)

    num_entities = graph_data.original_num_nodes
    num_relations = graph_data.original_num_relations

    entity_emb = all_emb[:num_entities]
    relation_emb = all_emb[num_entities:num_entities + num_relations]

    return entity_emb, relation_emb


def compute_embeddings_batched(model: FICE, graph_data, device: torch.device,
                               batch_size: int, num_neighbors: list):
    """Compute embeddings for ALL nodes via batched neighbor sampling.

    For each batch of seed nodes, samples their k-hop neighborhood, runs
    the embedding GNN on the subgraph, and extracts the seed embeddings.

    Returns: (entity_embeddings, relation_embeddings) as CPU tensors.
    """
    model.eval()
    embedding_model = model.embedding_model

    num_entities = graph_data.original_num_nodes
    num_relations = graph_data.original_num_relations
    total_nodes = num_entities + num_relations
    emb_dim = model.vector_dim

    sampler = MultigraphNeighborSampler(num_neighbors=num_neighbors)
    # Register graph in sampler cache
    if not hasattr(graph_data, 'dataset_name'):
        graph_data.dataset_name = '_inference_graph'

    all_embeddings = torch.zeros(total_nodes, emb_dim)
    all_node_ids = torch.arange(total_nodes)

    num_batches = (total_nodes + batch_size - 1) // batch_size
    logger.info(f"Computing embeddings in {num_batches} batches "
                f"(batch_size={batch_size}, total_nodes={total_nodes})")

    with torch.no_grad():
        for i in range(0, total_nodes, batch_size):
            seeds = all_node_ids[i:i + batch_size]
            sub_data, global_to_local = sampler.sample(graph_data, seeds)
            sub_data.original_num_nodes = graph_data.original_num_nodes
            sub_data.original_num_relations = graph_data.original_num_relations
            sub_data = sub_data.to(device)

            # Run embedding model on subgraph
            emb = embedding_model(data=sub_data, batch=None)  # (1, sub_N, D)
            emb = emb.squeeze(0).cpu()  # (sub_N, D)

            # Map seed node embeddings back to global IDs
            for seed_global in seeds:
                local_idx = global_to_local[seed_global.item()]
                if local_idx >= 0:
                    all_embeddings[seed_global] = emb[local_idx]

            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == num_batches:
                logger.info(f"  Batch {batch_num}/{num_batches}")

    entity_emb = all_embeddings[:num_entities]
    relation_emb = all_embeddings[num_entities:num_entities + num_relations]
    return entity_emb, relation_emb


# ---------------------------------------------------------------------------
# Occurrence counts
# ---------------------------------------------------------------------------

def build_occurrence_tensors(entity_occurrences: dict, relation_occurrences: dict,
                             entity_mapping: dict, relation_mapping: dict):
    """Convert URI-keyed occurrence dicts to dense float32 tensors indexed by ID.

    Returns: (entity_occ, relation_occ) as tensors of shape (N, 3) and (M, 3).
    """
    num_entities = len(entity_mapping)
    num_relations = len(relation_mapping)

    entity_occ = torch.zeros(num_entities, 3)
    for uri, occ in entity_occurrences.items():
        if uri in entity_mapping:
            idx = entity_mapping[uri]
            entity_occ[idx] = torch.tensor(occ[:3], dtype=torch.float32)

    relation_occ = torch.zeros(num_relations, 3)
    for uri, occ in relation_occurrences.items():
        if uri in relation_mapping:
            idx = relation_mapping[uri]
            relation_occ[idx] = torch.tensor(occ[:3], dtype=torch.float32)

    return entity_occ, relation_occ


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_query_gnn(model: FICE, config: Config, output_dir: Path):
    """Export the query GNN as TorchScript (both scripted and traced)."""
    if config.query_gnn.model != "TripleModel":
        raise NotImplementedError(
            f"TorchScript export only supports TripleModel, "
            f"got {config.query_gnn.model}"
        )
    if config.embedding_space != "euclidean":
        raise NotImplementedError(
            f"TorchScript export only supports euclidean embedding space, "
            f"got {config.embedding_space}"
        )

    feat_dim = config.embedding_dim + 3  # counts (3) + embedding (D)
    scriptable = build_scriptable_from_config(
        feat_dim=feat_dim,
        num_layers=config.query_gnn.num_layers,
        layer_norm=config.query_gnn.layer_norm,
        skip_connection=config.query_gnn.skip_connection,
        dropout=config.query_gnn.dropout,
        use_graph_size=True,
        readout=config.query_gnn.readout,
    )

    # Transfer weights from the original PyG TripleModel
    original_query_gnn = model.query_gnn
    transfer_weights(original_query_gnn, scriptable)
    scriptable.eval()

    # Verify numerical equivalence on a synthetic example
    _verify_equivalence(model, scriptable, feat_dim)

    # Export via torch.jit.script
    try:
        scripted = torch.jit.script(scriptable)
        scripted_path = output_dir / "query_gnn_scripted.pt"
        scripted.save(str(scripted_path))
        logger.info(f"Saved scripted model to {scripted_path}")
    except Exception as e:
        logger.warning(f"torch.jit.script failed: {e}")
        logger.warning("Falling back to traced-only export.")

    # Export via torch.jit.trace
    # Trace with batch_size=1 (single query) — this bakes in num_graphs=1,
    # which matches the C++ inference use case (one query at a time).
    example_x = torch.randn(3, feat_dim)
    example_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    example_edge_attr = torch.randn(2, feat_dim)
    example_batch = torch.zeros(3, dtype=torch.long)  # single graph
    example_graph_size = torch.tensor([1000.0])

    traced = torch.jit.trace(
        scriptable,
        (example_x, example_edge_index, example_edge_attr,
         example_batch, example_graph_size),
    )
    traced_path = output_dir / "query_gnn_traced.pt"
    traced.save(str(traced_path))
    logger.info(f"Saved traced model to {traced_path}")
    logger.info("Note: traced model assumes batch_size=1 (single query). "
                "Use the scripted model for batched inference.")


def _verify_equivalence(model: FICE, scriptable: ScriptableTripleModel,
                         feat_dim: int):
    """Run the same input through original and scriptable query GNN, assert match."""
    from torch_geometric.data import Data, Batch

    # Create a small synthetic query batch
    q1 = Data(
        x=torch.randn(3, feat_dim),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.randn(2, feat_dim),
    )
    q2 = Data(
        x=torch.randn(2, feat_dim),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.randn(1, feat_dim),
    )
    batch = Batch.from_data_list([q1, q2])
    graph_size = torch.tensor([5000.0])

    model.query_gnn.eval()
    scriptable.eval()

    with torch.no_grad():
        out_original = model.query_gnn(batch=batch, graph_size=graph_size)
        out_scriptable = scriptable(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, graph_size,
        )

    if torch.allclose(out_original, out_scriptable, atol=1e-5):
        logger.info("Numerical equivalence verified (atol=1e-5)")
    else:
        max_diff = (out_original - out_scriptable).abs().max().item()
        logger.warning(
            f"Numerical mismatch! Max diff: {max_diff:.6e}. "
            "Check weight transfer."
        )


# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------

def save_binary(tensor: torch.Tensor, path: Path):
    """Save a contiguous float32 tensor as raw bytes (mmap-friendly)."""
    t = tensor.contiguous().to(torch.float32)
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())
    logger.info(f"Saved {path} ({t.shape}, {t.numel() * 4 / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FICE inference preparation")
    parser.add_argument("--config", type=str, default="inference/inference_config.yaml",
                        help="Path to inference_config.yaml")
    args = parser.parse_args()

    # Load inference config
    with open(args.config) as f:
        inf_cfg = yaml.safe_load(f)

    checkpoint_path = Path(inf_cfg["checkpoint"])
    graph_dataset_name = inf_cfg["graph_dataset"]
    output_dir = Path(inf_cfg["output_dir"])
    batch_size = inf_cfg.get("batch_size", 0)
    device_str = inf_cfg.get("device", "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if device_str != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # ---- 1. Load checkpoint and reconstruct model ----
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    config = Config(**checkpoint["config"])

    if config.Conditioning != "false":
        raise ValueError(
            f"Precomputed embeddings require conditioning='false', "
            f"but checkpoint has conditioning='{config.Conditioning}'. "
            "Conditioning makes embeddings query-dependent."
        )
    if config.model != "FICE":
        raise ValueError(
            f"Inference only supports FICE model, got {config.model}"
        )

    model = build_fice_from_config(config, state_dict=checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # ---- 2. Load graph and build factor/bipartite graph ----
    logger.info(f"Loading graph dataset: {graph_dataset_name}")
    graph = load_dataset(graph_dataset_name)

    relation_source = config.relation_gnn.source
    if relation_source == "factor":
        graph_data = get_or_build_graph(graph, graph_dataset_name, "factor")
    else:
        graph_data = get_or_build_graph(graph, graph_dataset_name, "bipartite")

    num_entities = graph_data.original_num_nodes
    num_relations = graph_data.original_num_relations
    logger.info(f"Graph: {num_entities} entities, {num_relations} relations, "
                f"{graph.num_edges} edges")

    # ---- 3. Compute embeddings ----
    if batch_size <= 0:
        logger.info("Computing embeddings (full-graph mode)...")
        entity_emb, relation_emb = compute_embeddings_full_graph(
            model, graph_data, device)
    else:
        num_neighbors = config.GNNNeighborhood or [100, 100]
        logger.info(f"Computing embeddings (batched, neighbors={num_neighbors})...")
        entity_emb, relation_emb = compute_embeddings_batched(
            model, graph_data, device, batch_size, num_neighbors)

    logger.info(f"Entity embeddings: {entity_emb.shape}")
    logger.info(f"Relation embeddings: {relation_emb.shape}")

    # ---- 4. Load URI mappings and occurrence counts ----
    entity_occ_dict, relation_occ_dict, entity_mapping, relation_mapping = \
        load_occurrences(graph_dataset_name)

    entity_occ, relation_occ = build_occurrence_tensors(
        entity_occ_dict, relation_occ_dict, entity_mapping, relation_mapping)

    # ---- 5. Save embeddings and metadata ----
    save_binary(entity_emb, output_dir / "entity_embeddings.bin")
    save_binary(relation_emb, output_dir / "relation_embeddings.bin")
    save_binary(entity_occ, output_dir / "entity_occurrences.bin")
    save_binary(relation_occ, output_dir / "relation_occurrences.bin")

    # URI-to-ID mapping — strip angle brackets for clean lookup
    def _strip_brackets(mapping: dict) -> dict:
        return {k.strip("<>"): v for k, v in mapping.items()}

    uri_to_id = {
        "entities": _strip_brackets(entity_mapping),
        "relations": _strip_brackets(relation_mapping),
    }
    uri_to_id_path = output_dir / "uri_to_id.json"
    with open(uri_to_id_path, "w") as f:
        json.dump(uri_to_id, f, indent=2)
    logger.info(f"Saved {uri_to_id_path}")

    # Metadata
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
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {metadata_path}")

    # ---- 6. Export query GNN to TorchScript ----
    logger.info("Exporting query GNN to TorchScript...")
    model.cpu()  # Move to CPU for export
    export_query_gnn(model, config, output_dir)

    logger.info(f"Done! All artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
