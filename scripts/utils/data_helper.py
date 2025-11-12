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

# Dataset mapping: config name -> dataset class
DATASET_REGISTRY = {
    "LUBM": datasets.LUBM,
    "SWDF": datasets.SWDF,
    "Wikidata": datasets.Wikidata,
}

# Query mapping: config name -> query class
QUERY_REGISTRY = {
    "LUBMQueries": datasets.LUBMQueries,
    "SWDF_Star_Queries": datasets.SWDF_Star_Queries,
    "WIKIDATA_Star_Queries": datasets.WIKIDATA_Star_Queries,
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


def load_dataset(context: TrainingContext):
    """Load the graph dataset and verify its properties."""
    config = context.config
    dataset_name = config.dataset  # Use attribute access
    
    logger.info(f"Loading dataset: {dataset_name}...")
    
    # Get the appropriate dataset class
    DatasetClass = get_dataset_class(dataset_name)
    
    # Instantiate and load the dataset
    graph_dataset = DatasetClass()

    data = graph_dataset[0].to(torch.device('cpu'))  # Load on CPU initially

    logger.info(f"Graph loaded successfully!")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {data.num_edges}")
    logger.info(
        f"Node features shape: {data.x.shape if hasattr(data, 'x') and data.x is not None else 'None'}"
    )
    logger.info(f"Edge indices shape: {data.edge_index.shape}")
    logger.info(
        f"Edge types shape: {data.edge_type.shape if hasattr(data, 'edge_type') and data.edge_type is not None else 'None'}"
    )

    return data


def load_occurrences(context: TrainingContext):
    """Load entity and relation mappings and occurrence data."""
    config = context.config
    dataset_name = config.dataset  # Use attribute access
    
    logger.info(f"Loading occurrences for dataset: {dataset_name}...")
    
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
    query_name = config.queries  # Use attribute access
    
    logger.info(f"Loading queries: {query_name}...")
    
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

    logger.info(f"Queries loaded (on {queries[0].x.device}). Total queries: {len(queries)}")
    return queries

def train_test_split(context: TrainingContext, queries):
    config = context.config

    # Process and filter queries
    # Optional: filter queries by minimum cardinality threshold specified in config (e.g., min_cardinality: 2)
    min_card = getattr(config, "min_cardinality", None)
    if min_card is not None:
        original_count = len(queries)
        queries = [q for q in queries if q.y.item() >= min_card]
        logger.info(f"Filtered queries with cardinality >= {min_card}: {len(queries)}/{original_count}")

    queries_list = list(queries)
    random.shuffle(queries_list)
    if config.DEBUG_num_queries and config.DEBUG_num_queries > 0:
        num_queries = min(config.DEBUG_num_queries, len(queries_list))
    else:
        num_queries = len(queries_list)
    queries_list = queries_list[:num_queries]
    logger.info(f"Using debug mode with {num_queries} queries")

    train_size = int(len(queries_list) * config.train_ratio)
    test_size = len(queries_list) - train_size

    train_queries = queries_list[:train_size]
    test_queries = queries_list[train_size:]

    return train_queries, test_queries

