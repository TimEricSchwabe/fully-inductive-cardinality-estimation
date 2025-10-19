import torch
import json
import random
import logging
from pathlib import Path
from utils.models import *


# Import project modules
from fice import datasets
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def load_dataset(context: TrainingContext):
    """Load the graph dataset and verify its properties."""
    config = context.config
    device = context.device
    
    data_root = Path(config["data_root"])

    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    logger.info(f"Data root: {data_root}")


    # Load dataset
    logger.info("Loading dataset...")
    nt_path = data_root / "raw/swdf.nt"

    if not nt_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {nt_path}")

    graph = datasets.NTDatasets(
        root=data_root, nt_path=nt_path
    )  # TODO: passing the paramaters will change eventual, but for now this is the solution
    
    # Move graph to GPU once and keep it there for the entire training
    graph = graph[0].to(device)
    
    logger.info(f"Graph loaded successfully and moved to {graph.device}!")
    logger.info(f"Number of nodes: {graph.num_nodes}")
    logger.info(f"Number of edges: {graph.num_edges}")
    logger.info(
        f"Node features shape: {graph.x.shape if hasattr(graph, 'x') and graph.x is not None else 'None'}"
    )
    logger.info(f"Edge indices shape: {graph.edge_index.shape}")
    logger.info(
        f"Edge types shape: {graph.edge_type.shape if hasattr(graph, 'edge_type') and graph.edge_type is not None else 'None'}"
    )

    return graph


def load_occurrences(context: TrainingContext):
    """Load entity and relation mappings and occurrence data."""
    config = context.config
    
    logger.info("Loading occurrences...")
    data_root = Path(config["data_root"]) # TODO: check if this is necessary
    entity_occurrences_path = data_root / "processed/swdf_entity_occurrences.json"
    relation_occurrences_path = data_root / "processed/swdf_relation_occurrences.json"
    entity_mapping_path = data_root / "processed/swdf_entity_to_id.json"
    relation_mapping_path = data_root / "processed/swdf_relation_to_id.json"

    with open(entity_occurrences_path, "r") as f:
        entity_occurrences = json.load(f)
    with open(relation_occurrences_path, "r") as f:
        relation_occurrences = json.load(f)
    with open(entity_mapping_path, "r") as f:
        entity_mapping = json.load(f)
    with open(relation_mapping_path, "r") as f:
        relation_mapping = json.load(f)

    logger.info("Metadata loaded successfully!")
    logger.info(f"Entities with occurrences: {len(entity_occurrences)}")
    logger.info(f"Relations with occurrences: {len(relation_occurrences)}")
    logger.info(f"Entity mapping size: {len(entity_mapping)}")
    logger.info(f"Relation mapping size: {len(relation_mapping)}")

    return entity_occurrences, relation_occurrences, entity_mapping, relation_mapping

def load_queries(
    context: TrainingContext,
    entity_occurrences,
    relation_occurrences,
    entity_mapping,
    relation_mapping,
):
    """Load queries and create train/test split."""
    config = context.config
    # Keep queries on CPU initially - they'll be moved to GPU batch by batch
    
    logger.info("Loading queries...")
    queries_root = Path(config["queries_root"])
    if not queries_root.exists():
        raise FileNotFoundError(f"Queries root directory not found: {queries_root}")
    logger.info(f"Queries root: {queries_root}")

    query_path = queries_root / "raw/Joined_Queries.json"

    if not query_path.exists():
        raise FileNotFoundError(f"Queries file not found: {query_path}")



    queries = datasets.NTQueries(
        root=queries_root,
        queries_path=query_path,
        entity_occurrences=entity_occurrences,
        relation_occurrences=relation_occurrences,
        global_entity_mapping=entity_mapping,
        global_relation_mapping=relation_mapping,
        device=torch.device("cpu"),  # Keep queries on CPU
    ) #TODO: will change hopefully

    logger.info(f"Queries loaded (on CPU). Total queries: {len(queries)}")
    return queries

def train_test_split(context: TrainingContext, queries):
    config = context.config
    
    # Process and filter queries
    queries_list = list(queries)
    random.shuffle(queries_list)
    if config["DEBUG_num_queries"] and config["DEBUG_num_queries"] > 0:
        num_queries = min(config["DEBUG_num_queries"], len(queries_list))
    else:
        num_queries = len(queries_list)
    queries_list = queries_list[:num_queries]
    logger.info(f"Using debug mode with {num_queries} queries")

    train_size = int(len(queries_list) * config["train_ratio"])
    test_size = len(queries_list) - train_size

    train_queries = queries_list[:train_size]
    test_queries = queries_list[train_size:]


    return train_queries, test_queries

