import torch
import json
import random
import logging
from typing import List, Tuple
from pathlib import Path
from utils.models import *
import numpy as np


# Import project modules
from fice import datasets
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# Dataset mapping: config name -> dataset class
DATASET_REGISTRY = {
    "LUBM": datasets.LUBM,
    "SWDF": datasets.SWDF,
    "Wikidata": datasets.Wikidata,
    "fb237_v4": datasets.fb237_v4,
    "fb237_v4_ind": datasets.fb237_v4_ind,
    "wn18rr_v4": datasets.WN18RR_v4,
    "wn18rr_v4_ind": datasets.WN18RR_v4_ind,
    "dbpedia100k": datasets.DBpedia100k,
}

# Query mapping: config name -> query class
QUERY_REGISTRY = {
    "LUBMQueries": datasets.LUBMQueries,
    "SWDF_Star_Queries": datasets.SWDF_Star_Queries,
    "WIKIDATA_Star_Queries": datasets.WIKIDATA_Star_Queries,
    "fb237_v4_star_queries": datasets.fb237_v4_star_queries,
}


def get_dataset_class(dataset_name: str):
    """Get the dataset class from the registry based on config name."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]


def get_query_class(query_name: str):
    """Get the query class from the registry based on config name."""
    if query_name not in QUERY_REGISTRY:
        raise ValueError(
            f"Unknown query type: {query_name}. Available query types: {list(QUERY_REGISTRY.keys())}"
        )
    return QUERY_REGISTRY[query_name]


def get_dataset_processed_dir(dataset_name: str):
    """Get the processed directory for a dataset based on its name."""
    # Use the dataset name in lowercase for the processed directory
    dataset_name_lower = dataset_name.lower()
    return Path(datasets.DEFAULT_DATA_ROOT) / dataset_name_lower / "processed"

def load_dataset(dataset_name: str) -> List:
    """Load the graph dataset and verify its properties."""
    # Get the appropriate dataset class
    DatasetClass = get_dataset_class(dataset_name)
    
    # Instantiate and load the dataset
    graph_dataset = DatasetClass()
    graph = graph_dataset[0].to("cpu")
    
    
    # no logging here — logging is centralized in `load_data`

    return graph


def load_occurrences(dataset_name: str):
    """Load entity and relation mappings and occurrence data."""
    # Get the processed directory for the dataset
    processed_dir = get_dataset_processed_dir(dataset_name)
    
    # Construct file paths using the dataset name
    entity_occurrences_path = processed_dir / "entity_occurrences.json"
    relation_occurrences_path = processed_dir / "relation_occurrences.json"
    entity_mapping_path = processed_dir / "entity_to_id.json"
    relation_mapping_path = processed_dir / "relation_to_id.json"

    # Load the JSON files
    with open(entity_occurrences_path, "r") as f:
        entity_occurrences = json.load(f)
    with open(relation_occurrences_path, "r") as f:
        relation_occurrences = json.load(f)
    with open(entity_mapping_path, "r") as f:
        entity_mapping = json.load(f)
    with open(relation_mapping_path, "r") as f:
        relation_mapping = json.load(f)

    # no logging here — logging is centralized in `load_data`
 
    return entity_occurrences, relation_occurrences, entity_mapping, relation_mapping

def load_queries(
    query_name: str,
    entity_occurrences,
    relation_occurrences,
    entity_mapping,
    relation_mapping,
):
    """Load queries and create train/test split."""
    # Get the appropriate query class
    QueryClass = get_query_class(query_name)
    
    # Instantiate the query dataset
    # Keep queries on CPU initially - they'll be moved to GPU batch by batch
    queries = QueryClass(
        entity_occurrences=entity_occurrences,
        relation_occurrences=relation_occurrences,
        global_entity_mapping=entity_mapping,
        global_relation_mapping=relation_mapping,
    )

    # no logging here — logging is centralized in `load_data`
    return queries


def load_data(dataset_name: str, queries_name: str):
    """Load dataset, occurrences and queries and emit consolidated logging.

    Parameters:
    - dataset_name, queries_name: as before
    - train_ratio: optional float; if provided, a temporary train/test split is computed
      and logged (does not change return value). This allows callers to ask for split
      logging without changing the API behavior.
    - num_queries: optional int passed to the split when `train_ratio` is provided.

    Returns the same tuple as before: (dataset, (eo, ro, em, rm), queries)
    """
    dataset = load_dataset(dataset_name)
    eo, ro, em, rm = load_occurrences(dataset_name)
    queries = load_queries(
        queries_name,
        entity_occurrences=eo,
        relation_occurrences=ro,
        entity_mapping=em,
        relation_mapping=rm,
    )

    # Consolidated logging for the whole load sequence
    logger.info(f"Loaded dataset: {dataset_name}")
    try:
        device = dataset.edge_index.device
    except Exception:
        device = "unknown"
    logger.info(f"Graph moved to device: {device}")
    logger.info(f"Number of nodes: {getattr(dataset, 'num_nodes', 'unknown')}")
    logger.info(f"Number of edges: {getattr(dataset, 'num_edges', 'unknown')}")
    node_feat_shape = getattr(dataset, 'x', None)
    if node_feat_shape is not None:
        try:
            node_feat_shape = node_feat_shape.shape
        except Exception:
            node_feat_shape = "unknown"
    else:
        node_feat_shape = "None"
    logger.info(f"Node features shape: {node_feat_shape}")
    logger.info(f"Edge indices shape: {getattr(dataset, 'edge_index', 'unknown').shape if getattr(dataset, 'edge_index', None) is not None else 'unknown'}")
    edge_type = getattr(dataset, 'edge_type', None)
    try:
        edge_type_shape = edge_type.shape if edge_type is not None else 'None'
    except Exception:
        edge_type_shape = 'unknown'
    logger.info(f"Edge types shape: {edge_type_shape}")

    # Occurrence and mapping metadata
    logger.info("Loaded occurrence and mapping metadata")
    logger.info(f"Entities with occurrences: {len(eo)}")
    logger.info(f"Relations with occurrences: {len(ro)}")
    logger.info(f"Entity mapping size: {len(em)}")
    logger.info(f"Relation mapping size: {len(rm)}")

    # Queries summary
    logger.info(f"Queries loaded (on CPU). Total queries: {len(queries)}")

    return dataset, (eo, ro, em, rm), queries

def train_test_split(queries, train_ratio: float, num_queries: int = -1, min_cardinality: int = None) -> Tuple[List, List]:
    # TODO: takes a long time for some reason fix
    # Process and filter queries
    # Optional: filter queries by minimum cardinality threshold specified in config (e.g., min_cardinality: 2)
    # TODO:
    min_card = min_cardinality
    # Handle string "None" or convert to int if needed
    if isinstance(min_card, str):
        if min_card.lower() == "none" or min_card.lower() == "null":
            min_card = None
        else:
            min_card = int(min_card)
    if min_card is not None:
        original_count = len(queries)
        queries = [q for q in queries if q.y.item() >= min_card]
        logger.info(f"Filtered queries with cardinality >= {min_card}: {len(queries)}/{original_count}")

    queries_list = list(queries)
    random.shuffle(queries_list)

    if num_queries > 0:
        num_queries = min(num_queries, len(queries_list))
    else:
        num_queries = len(queries_list)

    queries_list = queries_list[:num_queries]
    # logging moved to `load_data` to centralize runtime messages

    train_size = int(len(queries_list) * train_ratio)
    test_size = len(queries_list) - train_size

    train_queries = queries_list[:train_size]
    test_queries = queries_list[train_size:]

    return train_queries, test_queries

