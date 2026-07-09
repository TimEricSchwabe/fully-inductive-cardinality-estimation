import torch
import json
import random
import logging
from typing import Dict, List, Tuple, Union
from pathlib import Path
from torch.utils.data import Subset
from utils.config import DataConfig
from utils.multigraph import MultigraphLoader
from utils.models import *


# Import project modules
from fice import datasets
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "cycle_dataset": datasets.cycle_dataset,
    "path_dataset": datasets.path_dataset,
    "implication_dataset": datasets.implication_dataset,
    "fano_dataset": datasets.fano_dataset,
    "ising_dataset": datasets.ising_dataset,
    "orthogonal_dataset": datasets.orthogonal_dataset,
    "two_clusters_dataset": datasets.two_clusters_dataset,
    "LUBM": datasets.LUBM,
    "SWDF": datasets.SWDF,
    "Wikidata": datasets.Wikidata,
    "fb237_v4": datasets.fb237_v4,
    "fb237_v4_ind": datasets.fb237_v4_ind,
    "wn18rr_v4": datasets.WN18RR_v4,
    "wn18rr_v4_ind": datasets.WN18RR_v4_ind,
    "dbpedia100k": datasets.DBpedia100k,
    "yago": datasets.Yago,
    "aids": datasets.AIDS,
    "hetionet": datasets.Hetionet,
    "codex_l": datasets.CodexL,
}

QUERY_REGISTRY = {
    "cycle_queries": datasets.cycle_queries,
    "path_queries": datasets.path_queries,
    "path_queries_new": datasets.path_queries_new,
    "fano_queries": datasets.fano_queries,
    "ising_queries": datasets.ising_queries,
    "orthogonal_queries": datasets.orthogonal_queries,
    "two_clusters_queries": datasets.two_clusters_queries,
    # LUBM
    "LUBMQueries": datasets.LUBMQueries,
    "lubm_path_queries": datasets.lubm_path_queries,
    "lubm_cycle_queries": datasets.lubm_cycle_queries,
    "lubm_diamond_queries": datasets.lubm_diamond_queries,
    "lubm_flower_queries": datasets.lubm_flower_queries,
    "lubm_snowflake_queries": datasets.lubm_snowflake_queries,
    "lubm_tree_queries": datasets.lubm_tree_queries,
    "lubm_path_star_ends_queries": datasets.lubm_path_star_ends_queries,

    # SWDF
    "SWDF_Star_Queries": datasets.SWDF_Star_Queries,
    "swdf_path_queries": datasets.swdf_path_queries,
    "swdf_cycle_queries": datasets.swdf_cycle_queries,
    "swdf_diamond_queries": datasets.swdf_diamond_queries,
    "swdf_flower_queries": datasets.swdf_flower_queries,
    "swdf_snowflake_queries": datasets.swdf_snowflake_queries,
    "swdf_tree_queries": datasets.swdf_tree_queries,
    "swdf_path_star_ends_queries": datasets.swdf_path_star_ends_queries,

    # Wikidata
    "WIKIDATA_Star_Queries": datasets.WIKIDATA_Star_Queries,
    "wikidata_path_queries": datasets.wikidata_path_queries,
    "wikidata_cycle_queries": datasets.wikidata_cycle_queries,
    "wikidata_diamond_queries": datasets.wikidata_diamond_queries,
    "wikidata_flower_queries": datasets.wikidata_flower_queries,
    "wikidata_snowflake_queries": datasets.wikidata_snowflake_queries,
    "wikidata_tree_queries": datasets.wikidata_tree_queries,
    "wikidata_path_star_ends_queries": datasets.wikidata_path_star_ends_queries,

    # fb237_v4
    "fb237_v4_star_queries": datasets.fb237_v4_star_queries,
    "fb237_v4_path_queries": datasets.fb237_v4_path_queries,
    "fb237_v4_cycle_queries": datasets.fb237_v4_cycle_queries,
    "fb237_v4_diamond_queries": datasets.fb237_v4_diamond_queries,
    "fb237_v4_flower_queries": datasets.fb237_v4_flower_queries,
    "fb237_v4_snowflake_queries": datasets.fb237_v4_snowflake_queries,
    "fb237_v4_tree_queries": datasets.fb237_v4_tree_queries,
    "fb237_v4_path_star_ends_queries": datasets.fb237_v4_path_star_ends_queries,

    # fb237_v4_ind
    "fb237_v4_ind_star_queries": datasets.fb237_v4_ind_star_queries,
    "fb237_v4_ind_path_queries": datasets.fb237_v4_ind_path_queries,

    # wn18rr_v4
    "wn18rr_v4_star_queries": datasets.wn18rr_v4_star_queries,
    "wn18rr_v4_path_queries": datasets.wn18rr_v4_path_queries,
    "wn18rr_v4_cycle_queries": datasets.wn18rr_v4_cycle_queries,
    "wn18rr_v4_diamond_queries": datasets.wn18rr_v4_diamond_queries,
    "wn18rr_v4_flower_queries": datasets.wn18rr_v4_flower_queries,
    "wn18rr_v4_snowflake_queries": datasets.wn18rr_v4_snowflake_queries,
    "wn18rr_v4_tree_queries": datasets.wn18rr_v4_tree_queries,

    # wn18rr_v4_ind
    "wn18rr_v4_ind_star_queries": datasets.wn18rr_v4_ind_star_queries,
    "wn18rr_v4_ind_path_queries": datasets.wn18rr_v4_ind_path_queries,

    # dbpedia100k
    "dbpedia100k_star_queries": datasets.dbpedia100k_star_queries,
    "dbpedia100k_path_queries": datasets.dbpedia100k_path_queries,
    "dbpedia100k_cycle_queries": datasets.dbpedia100k_cycle_queries,
    "dbpedia100k_diamond_queries": datasets.dbpedia100k_diamond_queries,
    "dbpedia100k_flower_queries": datasets.dbpedia100k_flower_queries,
    "dbpedia100k_snowflake_queries": datasets.dbpedia100k_snowflake_queries,
    "dbpedia100k_path_star_ends_queries": datasets.dbpedia100k_path_star_ends_queries,
    "dbpedia100k_path_queries_ranking": datasets.dbpedia100k_path_queries_ranking,
    "dbpedia100k_star_queries_ranking": datasets.dbpedia100k_star_queries_ranking,
    "dbpedia100k_cycle_queries_ranking": datasets.dbpedia100k_cycle_queries_ranking,
    "dbpedia100k_diamond_queries_ranking": datasets.dbpedia100k_diamond_queries_ranking,
    "dbpedia100k_flower_queries_ranking": datasets.dbpedia100k_flower_queries_ranking,
    "dbpedia100k_snowflake_queries_ranking": datasets.dbpedia100k_snowflake_queries_ranking,
    "dbpedia100k_path_star_ends_queries_ranking": datasets.dbpedia100k_path_star_ends_queries_ranking,
    "dbpedia100k_binary_tree_queries_ranking": datasets.dbpedia100k_binary_tree_queries_ranking,

    # Yago
    "yago_star_queries": datasets.yago_star_queries,
    "yago_path_queries": datasets.yago_path_queries,
    "yago_cycle_queries": datasets.yago_cycle_queries,
    "yago_diamond_queries": datasets.yago_diamond_queries,
    "yago_flower_queries": datasets.yago_flower_queries,
    "yago_snowflake_queries": datasets.yago_snowflake_queries,
    "yago_binary_tree_queries": datasets.yago_binary_tree_queries,
    "yago_path_star_ends_queries": datasets.yago_path_star_ends_queries,

    # AIDS
    "aids_star_queries": datasets.aids_star_queries,
    "aids_path_queries": datasets.aids_path_queries,
    "aids_cycle_queries": datasets.aids_cycle_queries,
    "aids_diamond_queries": datasets.aids_diamond_queries,
    "aids_flower_queries": datasets.aids_flower_queries,
    "aids_snowflake_queries": datasets.aids_snowflake_queries,
    "aids_binary_tree_queries": datasets.aids_binary_tree_queries,
    "aids_path_star_ends_queries": datasets.aids_path_star_ends_queries,

    # Hetionet
    "hetionet_star_queries": datasets.hetionet_star_queries,
    "hetionet_path_queries": datasets.hetionet_path_queries,
    "hetionet_cycle_queries": datasets.hetionet_cycle_queries,
    "hetionet_diamond_queries": datasets.hetionet_diamond_queries,
    "hetionet_flower_queries": datasets.hetionet_flower_queries,
    "hetionet_snowflake_queries": datasets.hetionet_snowflake_queries,
    "hetionet_binary_tree_queries": datasets.hetionet_binary_tree_queries,
    "hetionet_path_star_ends_queries": datasets.hetionet_path_star_ends_queries,

    # CodexL
    "codex_l_star_queries": datasets.codex_l_star_queries,
    "codex_l_path_queries": datasets.codex_l_path_queries,
    "codex_l_cycle_queries": datasets.codex_l_cycle_queries,
    "codex_l_diamond_queries": datasets.codex_l_diamond_queries,
    "codex_l_flower_queries": datasets.codex_l_flower_queries,
    "codex_l_snowflake_queries": datasets.codex_l_snowflake_queries,
    "codex_l_binary_tree_queries": datasets.codex_l_binary_tree_queries,
    "codex_l_path_star_ends_queries": datasets.codex_l_path_star_ends_queries,

    # ----- Multi-KG ranking datasets (chain-sampled, generated on coma) -----

    # aids ranking
    "aids_star_queries_ranking": datasets.aids_star_queries_ranking,
    "aids_path_queries_ranking": datasets.aids_path_queries_ranking,
    "aids_cycle_queries_ranking": datasets.aids_cycle_queries_ranking,
    "aids_diamond_queries_ranking": datasets.aids_diamond_queries_ranking,
    "aids_flower_queries_ranking": datasets.aids_flower_queries_ranking,
    "aids_snowflake_queries_ranking": datasets.aids_snowflake_queries_ranking,
    "aids_path_star_ends_queries_ranking": datasets.aids_path_star_ends_queries_ranking,
    "aids_binary_tree_queries_ranking": datasets.aids_binary_tree_queries_ranking,

    # codex_l ranking
    "codex_l_star_queries_ranking": datasets.codex_l_star_queries_ranking,
    "codex_l_path_queries_ranking": datasets.codex_l_path_queries_ranking,
    "codex_l_cycle_queries_ranking": datasets.codex_l_cycle_queries_ranking,
    "codex_l_diamond_queries_ranking": datasets.codex_l_diamond_queries_ranking,
    "codex_l_flower_queries_ranking": datasets.codex_l_flower_queries_ranking,
    "codex_l_path_star_ends_queries_ranking": datasets.codex_l_path_star_ends_queries_ranking,
    "codex_l_binary_tree_queries_ranking": datasets.codex_l_binary_tree_queries_ranking,

    # fb237_v4 ranking
    "fb237_v4_star_queries_ranking": datasets.fb237_v4_star_queries_ranking,
    "fb237_v4_path_queries_ranking": datasets.fb237_v4_path_queries_ranking,
    "fb237_v4_cycle_queries_ranking": datasets.fb237_v4_cycle_queries_ranking,
    "fb237_v4_diamond_queries_ranking": datasets.fb237_v4_diamond_queries_ranking,
    "fb237_v4_flower_queries_ranking": datasets.fb237_v4_flower_queries_ranking,
    "fb237_v4_snowflake_queries_ranking": datasets.fb237_v4_snowflake_queries_ranking,
    "fb237_v4_path_star_ends_queries_ranking": datasets.fb237_v4_path_star_ends_queries_ranking,
    "fb237_v4_binary_tree_queries_ranking": datasets.fb237_v4_binary_tree_queries_ranking,

    # hetionet ranking
    "hetionet_star_queries_ranking": datasets.hetionet_star_queries_ranking,
    "hetionet_path_queries_ranking": datasets.hetionet_path_queries_ranking,

    # lubm ranking
    "lubm_diamond_queries_ranking": datasets.lubm_diamond_queries_ranking,
    "lubm_flower_queries_ranking": datasets.lubm_flower_queries_ranking,
    "lubm_snowflake_queries_ranking": datasets.lubm_snowflake_queries_ranking,
    "lubm_path_star_ends_queries_ranking": datasets.lubm_path_star_ends_queries_ranking,
    "lubm_binary_tree_queries_ranking": datasets.lubm_binary_tree_queries_ranking,

    # swdf ranking
    "swdf_path_queries_ranking": datasets.swdf_path_queries_ranking,
    "swdf_cycle_queries_ranking": datasets.swdf_cycle_queries_ranking,
    "swdf_diamond_queries_ranking": datasets.swdf_diamond_queries_ranking,
    "swdf_flower_queries_ranking": datasets.swdf_flower_queries_ranking,
    "swdf_snowflake_queries_ranking": datasets.swdf_snowflake_queries_ranking,
    "swdf_path_star_ends_queries_ranking": datasets.swdf_path_star_ends_queries_ranking,
    "swdf_binary_tree_queries_ranking": datasets.swdf_binary_tree_queries_ranking,

    # wikidata ranking
    "wikidata_star_queries_ranking": datasets.wikidata_star_queries_ranking,
    "wikidata_path_queries_ranking": datasets.wikidata_path_queries_ranking,
    "wikidata_cycle_queries_ranking": datasets.wikidata_cycle_queries_ranking,

    # wn18rr_v4 ranking
    "wn18rr_v4_star_queries_ranking": datasets.wn18rr_v4_star_queries_ranking,
    "wn18rr_v4_path_queries_ranking": datasets.wn18rr_v4_path_queries_ranking,
    "wn18rr_v4_cycle_queries_ranking": datasets.wn18rr_v4_cycle_queries_ranking,
    "wn18rr_v4_diamond_queries_ranking": datasets.wn18rr_v4_diamond_queries_ranking,
    "wn18rr_v4_flower_queries_ranking": datasets.wn18rr_v4_flower_queries_ranking,
    "wn18rr_v4_snowflake_queries_ranking": datasets.wn18rr_v4_snowflake_queries_ranking,
    "wn18rr_v4_binary_tree_queries_ranking": datasets.wn18rr_v4_binary_tree_queries_ranking,

    # yago ranking
    "yago_star_queries_ranking": datasets.yago_star_queries_ranking,
    "yago_path_queries_ranking": datasets.yago_path_queries_ranking,
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

def load_dataset(dataset_name: str) -> datasets.NTDatasets:
    """Load the graph dataset and verify its properties."""
    DatasetClass = get_dataset_class(dataset_name)
    
    graph_dataset = DatasetClass()
    graph = graph_dataset[0].to("cpu")
    return graph


def load_occurrences(dataset_name: str):
    """Load entity and relation mappings and occurrence data."""
    processed_dir = get_dataset_processed_dir(dataset_name)
    
    entity_occurrences_path = processed_dir / "entity_occurrences.json"
    relation_occurrences_path = processed_dir / "relation_occurrences.json"
    entity_mapping_path = processed_dir / "entity_to_id.json"
    relation_mapping_path = processed_dir / "relation_to_id.json"

    with open(entity_occurrences_path, "r") as f:
        entity_occurrences = json.load(f)
    with open(relation_occurrences_path, "r") as f:
        relation_occurrences = json.load(f)
    with open(entity_mapping_path, "r") as f:
        entity_mapping = json.load(f)
    with open(relation_mapping_path, "r") as f:
        relation_mapping = json.load(f)

 
    return entity_occurrences, relation_occurrences, entity_mapping, relation_mapping

def load_queries(
    query_name: str,
    entity_occurrences,
    relation_occurrences,
    entity_mapping,
    relation_mapping,
):
    """Load queries and create train/test split."""
    QueryClass = get_query_class(query_name)
    
    # Keep queries on CPU initially - they'll be moved to GPU batch by batch
    queries = QueryClass(
        entity_occurrences=entity_occurrences,
        relation_occurrences=relation_occurrences,
        global_entity_mapping=entity_mapping,
        global_relation_mapping=relation_mapping,
    )

    return queries


def load_data(dataset_name: str, queries_name: str) -> Tuple[datasets.NTDatasets, Tuple[Dict, Dict, Dict, Dict], List]:
    """Load dataset, occurrences and queries and emit consolidated logging.

    Parameters:
    - dataset_name, queries_name: as before
    - train_ratio: optional float; if provided, a temporary train/test split is computed
      and logged (does not change return value). This allows callers to ask for split
      logging without changing the API behavior.
    - num_queries: optional int passed to the split when `train_ratio` is provided.

    Returns the same tuple as before: (dataset, (eo, ro, em, rm), queries)
    """
    dataset: datasets.NTDatasets = load_dataset(dataset_name)
    eo, ro, em, rm = load_occurrences(dataset_name)
    queries = load_queries(
        queries_name,
        entity_occurrences=eo,
        relation_occurrences=ro,
        entity_mapping=em,
        relation_mapping=rm,
    )

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

def train_test_split(queries, train_ratio: float, shuffle: bool = False) -> Tuple[Subset, Subset]:
    n = len(queries)
    indices = list(range(n))

    if shuffle:
        random.shuffle(indices)

    train_size = int(n * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Return Subset objects that lazily index into the original dataset
    return Subset(queries, train_indices), Subset(queries, test_indices)


class RankingSubset(torch.utils.data.Dataset):
    """Subset of a ranking query set that preserves group boundaries.

    Splits at the parent-group level, not the item level, so every parent's
    sub-BGPs are kept together. Exposes `group_boundaries` in subset-local
    index space so `GroupBatchSampler` works unchanged.
    """

    def __init__(self, dataset, group_indices: List[int]):
        self.dataset = dataset
        parent_boundaries = list(dataset.group_boundaries)
        self.item_indices: List[int] = []
        self.group_boundaries: List[Tuple[int, int]] = []
        offset = 0
        for g in group_indices:
            s, e = parent_boundaries[g]
            self.item_indices.extend(range(s, e))
            size = e - s
            self.group_boundaries.append((offset, offset + size))
            offset += size

    def __len__(self):
        return len(self.item_indices)

    def __getitem__(self, idx):
        return self.dataset[self.item_indices[idx]]


def ranking_train_test_split(dataset, train_ratio: float, shuffle: bool = True) -> Tuple[RankingSubset, RankingSubset]:
    n = len(dataset.group_boundaries)
    order = list(range(n))
    if shuffle:
        random.shuffle(order)
    split = int(n * train_ratio)
    return RankingSubset(dataset, order[:split]), RankingSubset(dataset, order[split:])

def preprocess_data(data: List[any], min_cardinality: int = None, num_queries: int = None) -> List[any]:
    # Optional: filter queries by minimum cardinality threshold specified in config (e.g., min_cardinality: 2)
    processed_data = data
    if min_cardinality is not None:
        processed_data = [d for d in data if d.y.item() >= min_cardinality] 

    if num_queries is not None or num_queries > 0:
        actual_num_queries = min(num_queries, len(processed_data))
        processed_data = processed_data[:actual_num_queries]

    
    return processed_data

# TODO: https://gemini.google.com/share/fa266d99aba4
def make_bipartit_graph(data):
    """ Convert a homogeneous graph to a bipartite graph representation """

    edge_index = data.edge_index
    edge_type = data.edge_type 
    
    # Calculate actual number of relation types
    num_relations = edge_type.max().item() + 1
    
    # Offset relation types to create their node IDs
    relation_nodes = edge_type + data.num_nodes
    
    # 1. Create Directed Bipartite Edges (s -> p and p -> o)
    # Head part: Source -> RelationNode
    edge_index_head = torch.vstack((edge_index[0], relation_nodes))
    
    # Tail part: RelationNode -> Target
    edge_index_tail = torch.vstack((relation_nodes, edge_index[1]))
    
    # Combine them
    bipartite_edge_index = torch.hstack((edge_index_head, edge_index_tail))
    
    # Correct total number of nodes
    total_num_nodes = data.num_nodes + num_relations

    # Create initial weights of 1 (counts)
    edge_weight = torch.ones(bipartite_edge_index.size(1), device=bipartite_edge_index.device)

    # Collapse duplicate edges to get counts
    # unique_edges shape: [2, num_unique_edges]
    # counts shape: [num_unique_edges]
    unique_edges, counts = coalesce(
        bipartite_edge_index, 
        edge_weight, 
        num_nodes=total_num_nodes,
        reduce='add'
    )

    # 2. Make Undirected (Add Reverse Edges) with Direction Attribute
    # Forward direction: 1
    # Reverse direction: -1
    
    # Prepare attributes for forward edges: [Count, 1]
    # Shape: [num_unique_edges, 2]
    forward_attr = torch.stack([counts, torch.ones_like(counts)], dim=1)
    
    # Prepare reverse edges: swap rows 0 and 1
    reverse_edges = torch.stack([unique_edges[1], unique_edges[0]], dim=0)
    
    # Prepare attributes for reverse edges: [Count, -1]
    reverse_attr = torch.stack([counts, -torch.ones_like(counts)], dim=1)
    
    # Concatenate everything
    final_edge_index = torch.cat([unique_edges, reverse_edges], dim=1)
    final_edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)

    # offset = data.num_nodes
    # min_relation_index = offset # inclusive
    # max_relation_index = offset + full_bipartite_graph.original_num_relations # exclusive
    # relation_indices = torch.arange(min_relation_index, max_relation_index, device=device)

    # min_entity_index = 0 # inclusive
    # max_entity_index = offset # exclusive
    # entity_indices = torch.arange(min_entity_index, max_entity_index, device=device)


    
    bipartite_data = Data(
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,  # Shape: [2*E_unique, 2] -> (Count, Direction)
        num_nodes=total_num_nodes, 
        dataset_name=f"{data.dataset_name}_bipartite",
        original_edge_index=data.edge_index,
        original_edge_type=data.edge_type,
        original_num_nodes=data.num_nodes,
        original_num_relations=num_relations,
    )

    return bipartite_data

def make_factor_graph(data):
    """Convert a homogeneous graph to a factor graph representation.

    Each triple (s, p, o) introduces a triple node connected to subject, predicate,
    and object nodes. Node layout: entities [0, N), relations [N, N+R), triples [N+R, N+R+T).
    Edge attr is [count, role] where role encodes subject(±1), predicate(±2), object(±3).
    """
    edge_index = data.edge_index
    edge_type = data.edge_type
    num_entities = data.num_nodes
    num_relations = edge_type.max().item() + 1
    num_triples = edge_index.size(1)

    triple_node_ids = torch.arange(num_triples) + num_entities + num_relations

    # 3 directed edge sets: triple→subject (role 1), triple→predicate (role 2), triple→object (role 3)
    subject_edges = torch.vstack((triple_node_ids, edge_index[0]))
    predicate_edges = torch.vstack((triple_node_ids, edge_type + num_entities))
    object_edges = torch.vstack((triple_node_ids, edge_index[1]))

    combined = torch.hstack((subject_edges, predicate_edges, object_edges))
    role_types = torch.cat([
        torch.ones(num_triples) * 1,
        torch.ones(num_triples) * 2,
        torch.ones(num_triples) * 3,
    ])

    total_nodes = num_entities + num_relations + num_triples

    # Coalesce duplicate edges (sum counts)
    edge_weight = torch.ones(combined.size(1))
    unique_edges, counts = coalesce(combined, edge_weight, num_nodes=total_nodes, reduce='add')

     # We need to track roles through coalesce — but since each triple node is unique,
    # there are no duplicate (src, dst) pairs, so we can coalesce role_types too. not necessary
    # if you assume unique triples AND ignores self-loops (gives them role 4) - but emperically found to work better
    _, unique_roles = coalesce(combined, role_types, num_nodes=total_nodes, reduce='add')


    # Make undirected: add reverse edges with negated role
    forward_attr = torch.stack([counts, unique_roles], dim=1)
    reverse_edges = torch.stack([unique_edges[1], unique_edges[0]], dim=0)
    reverse_attr = torch.stack([counts, -unique_roles], dim=1)

    final_edge_index = torch.cat([unique_edges, reverse_edges], dim=1)


    final_edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)

    return Data(
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
        num_nodes=total_nodes,
        dataset_name=f"{data.dataset_name}_factor",
        original_edge_index=data.edge_index,
        original_edge_type=data.edge_type,
        original_num_nodes=num_entities,
        original_num_relations=num_relations,
    )


def get_or_build_graph(dataset, dataset_name, graph_type):
    """Build or load cached bipartite/factor graph for a dataset."""
    processed_dir = get_dataset_processed_dir(dataset_name)
    cache_path = processed_dir / f"{dataset_name}_{graph_type}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached {graph_type} graph from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    logger.info(f"Building {graph_type} graph for {dataset_name}...")
    graph = make_factor_graph(dataset) if graph_type == "factor" else make_bipartit_graph(dataset)
    torch.save(graph, cache_path)
    logger.info(f"Cached {graph_type} graph to {cache_path}")
    return graph


def prepare_data(config: Union[Config, List[DataConfig]]) -> DataComponents:

    data_configs: List[DataConfig]

    if isinstance(config, Config):
        data_configs = config.data
    else:
        data_configs = config

    splits = [dc.type for dc in data_configs]
    if not ('train' in splits or 'train-val' in splits):
        raise ValueError("At least one training dataset must be provided in the data configuration.")

    train_datasets = [] 
    train_sets = [] 
    train_batch_size = []

    validation_datasets = []
    validation_sets = [] 
    validation_batch_size = []

    test_dataset = []
    test_sets = []
    test_batch_size = []

    for data_config in data_configs:
        dataset_name = data_config.dataset
        queries_name = data_config.queries
        dataset, _, queries = load_data(dataset_name=dataset_name, queries_name=queries_name)

        is_ranking = isinstance(queries, datasets.NTRankingQueries)
        if not is_ranking:
            queries = preprocess_data(queries, min_cardinality=config.min_cardinality, num_queries=data_config.num_queries)

        # Build only the graph type needed by relation_gnn.source
        relation_source = config.relation_gnn.source if isinstance(config, Config) else 'bipartite'
        if relation_source == 'factor':
            dataset.factor_graph = get_or_build_graph(dataset, dataset_name, 'factor')
        else:
            dataset.bipartite_graph = get_or_build_graph(dataset, dataset_name, 'bipartite')

        if data_config.type == 'train':
            train_datasets.append(dataset)
            train_sets.append(queries)
            train_batch_size.append(data_config.batch_size)

        elif data_config.type == 'validation':
            validation_datasets.append(dataset)
            validation_sets.append(queries)
            validation_batch_size.append(data_config.batch_size)

        elif data_config.type == 'test':
            test_dataset.append(dataset)
            test_sets.append(queries)
            test_batch_size.append(data_config.batch_size)

        elif data_config.type == 'train-val':
            train_datasets.append(dataset)
            validation_datasets.append(dataset)
            train_batch_size.append(data_config.batch_size)
            validation_batch_size.append(data_config.batch_size)

            if is_ranking:
                train_q, test_q = ranking_train_test_split(
                    dataset=queries, train_ratio=config.train_ratio, shuffle=True,
                )
            else:
                train_q, test_q = train_test_split(queries=queries, train_ratio=config.train_ratio, shuffle=True)
            train_sets.append(train_q)
            validation_sets.append(test_q)

    group_batch_size = getattr(config, "ranking_group_batch", 4) if isinstance(config, Config) else 4

    train_loader = MultigraphLoader(
        data=train_datasets,
        queries=train_sets,
        batch_size=train_batch_size,
        shuffle=True,
        group_batch_size=group_batch_size,
    )
    validation_loader = MultigraphLoader(
        data=validation_datasets,
        queries=validation_sets,
        batch_size=validation_batch_size,
        shuffle=True,
        group_batch_size=group_batch_size,
    )

    if len(test_dataset) == 0 or len(test_sets) == 0:
        test_dataset = validation_datasets
        test_sets = validation_sets
        test_batch_size = validation_batch_size

    test_loader = MultigraphLoader(
        data=test_dataset,
        queries=test_sets,
        batch_size=test_batch_size,
        shuffle=True,
        group_batch_size=group_batch_size,
    )

    data_component = DataComponents(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )

    return data_component

