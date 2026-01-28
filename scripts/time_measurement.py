import sys
import logging
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import pandas as pd
import torch.nn.functional as F
import copy

# Setup paths to allow imports
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Project imports
from fice.models import FICE
from utils.checkpoint_helper import load_checkpoint
from utils.config import Config, load_config_yaml
from utils.data_helper import load_data, preprocess_data, make_bipartit_graph
from utils.helper import get_device, set_random_seeds, process_seed
from utils.neighborhood_sampler import MultigraphNeighborSampler
from utils.multigraph import MultigraphLoader
from train import initialize_model_components, prepare_bipartite_graph, prepare_bipartite_relation_graph

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path('/Users/lukas/Workspace/Uni/BachelorsThesis/FICE/notebooks/fice_runs/swdf/checkpoints/best_model_19.pt')
CONFIG_PATH = Path('/Users/lukas/Workspace/Uni/BachelorsThesis/runtime-branch/configs/train_full.yaml')



def prepare_data(config: Config):
    data_configs = config.data 
    datasets = []
    queries = []
    bs = []
    
    for dc in data_configs:
        bs.append(1)
        dc.num_queries = 1000

        d, _, q = load_data(dataset_name=dc.dataset, queries_name=dc.queries)

        if config.relation_gnn.model in ['BipartiteGraphGNN']:
            bipartite_data = make_bipartit_graph(d)
            d.bipartite_graph = bipartite_data



        if config.laplacian_k > 0:
            transform = AddLaplacianEigenvectorPE(k=config.laplacian_k)
            d = transform(d)

            if config.relation_gnn.model in ['BipartiteGraphGNN']:
                d.bipartite_graph = transform(d.bipartite_graph)

            if hasattr(d, 'relation_graph') and d.relation_graph is not None and config.relation_gnn.model in ['RelationGNN']:
                d.relation_graph = transform(d.relation_graph)
            

        q = preprocess_data(q, num_queries=dc.num_queries)
        import itertools
        if d.dataset_name == 'wn18rr_v4':
            all_types = list(itertools.chain.from_iterable([f.edge_embedding_indices.tolist() for f in q]))
            unique_types = np.unique(all_types)
            assert len(unique_types) == 9
        datasets.append(d)
        queries.append(q)



    loader = MultigraphLoader(
        data=datasets,
        queries=queries,
        batch_size=bs,
        shuffle=False,
    )
    return loader

def main():
    if not CHECKPOINT_PATH.exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = load_checkpoint(checkpoint_path=CHECKPOINT_PATH)
    config = Config(**checkpoint['config'])

    # overwrite config with local config
    local_config = load_config_yaml(CONFIG_PATH)
    config = local_config
    
    # Force batch size 1
    for d in config.data:
        d.batch_size = 1
    logger.info("Configuration override: batch_size = 1")
    
    device = get_device(config.device)
    logger.info(f"Device: {device}")
    
    process_seed(config.seed)
    set_random_seeds(config.seed)
    
    logger.info("Loading datasets...")
    loader = prepare_data(config)
    neighbor_sampler = MultigraphNeighborSampler(num_neighbors=config.GNNNeighborhood)
     # Common params
    conditioning = config.Conditioning
    model_type = config.model
    vector_dim = config.embedding_dim
    n_embedding_layers = config.n_embedding_layers
    
    # Entity GNN params
    entity_message_func = config.entity_gnn.message_func
    entity_layer_norm = config.entity_gnn.layer_norm
    entity_skip_connection = config.entity_gnn.skip_connection
    entity_dropout = config.entity_gnn.dropout
    entity_graph_mode = "bipartite" if config.entity_gnn.model == "BipartiteGraphGNN" else "KG"

    # Relation GNN params
    relation_message_func = config.relation_gnn.message_func
    relation_layer_norm = config.relation_gnn.layer_norm
    relation_skip_connection = config.relation_gnn.skip_connection
    relation_dropout = config.relation_gnn.dropout
    relation_graph_mode = "bipartite" if config.relation_gnn.model == "BipartiteGraphGNN" else "RelationGraph"

    # Query GNN params
    query_gnn_model = config.query_gnn.model
    query_gnn_skip_connection = config.query_gnn.skip_connection
    query_gnn_layer_norm = config.query_gnn.layer_norm
    query_gnn_dropout = config.query_gnn.dropout
    
    
    model = FICE(
        vector_dim=vector_dim,
        n_embedding_layers=n_embedding_layers, 
        conditioning=conditioning,
        entity_graph_mode=entity_graph_mode,
        relation_graph_mode=relation_graph_mode,
        
        # Entity Params
        entity_message_func=entity_message_func,
        entity_layer_norm=entity_layer_norm,
        entity_skip_connection=entity_skip_connection,
        entity_dropout=entity_dropout,
        
        # Relation Params
        relation_message_func=relation_message_func,
        relation_layer_norm=relation_layer_norm,
        relation_skip_connection=relation_skip_connection,
        relation_dropout=relation_dropout,
        
        # Query Params
        query_gnn_type=query_gnn_model,
        query_layer_norm=query_gnn_layer_norm,
        query_skip_connection=query_gnn_skip_connection,
        query_gnn_dropout=query_gnn_dropout,
        laplacian_k=config.laplacian_k,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    results_path = Path(__file__).parent / 'inference_times.csv'
    # Initialize CSV with headers
    pd.DataFrame(columns=['dataset', 'query_type', 'true_cardinality', 'avg_inference_time']).to_csv(results_path, index=False)

    repeat = 30
    with torch.no_grad():
        for batch_idx, (original_full_graph, original_query_batch) in enumerate(loader):
            logger.info(f"Running inference on batch {batch_idx+1}/{len(loader)}")

            full_graph = original_full_graph.detach().clone()
            query_batch = original_query_batch.detach().clone()

            # full_graph = copy.deepcopy(original_full_graph)
            # query_batch = copy.deepcopy(original_query_batch)
            graph_size = torch.tensor([full_graph.num_edges], device=device)

            bipartite_graph = None
            bipartite_graph_gloal_to_local = None

            use_bipartite_entity = config.entity_gnn.model == "BipartiteGraphGNN"
            use_bipartite_relation = config.relation_gnn.model == "BipartiteGraphGNN"

                # generate a sample of the bipartite graph if needed
            if use_bipartite_entity or use_bipartite_relation:
                bipartite_graph, bipartite_graph_gloal_to_local = prepare_bipartite_graph(full_graph, query_batch, neighbor_sampler, device)

            if use_bipartite_relation:
                relation_graph = prepare_bipartite_relation_graph(full_graph, bipartite_graph, bipartite_graph_gloal_to_local, query_batch, device)
            else: 
                relation_graph = full_graph.relation_graph.to(device)


            # "KG" mode - Standard Logic
            num_neighbors = config.GNNNeighborhood   
            if num_neighbors is not None:
                # Reusable sampler path
                node_indices = query_batch.node_embedding_indices
                seeds_cpu = node_indices[node_indices >= 0].unique().detach().cpu()
                if seeds_cpu.numel() > 0:
                    # Get subgraph for the current query nodes and how global node ids map to node ids in the subgraph
                    # If a node is not in the subgraph, its mapped to -1
                    graph_sub, global_to_local = neighbor_sampler.sample(full_graph, seeds_cpu)
                    query_nodes = query_batch.node_embedding_indices.detach().cpu()
                    # Create a mapping that maps how nodes in the query graph map to nodes in the sampled subgraph
                    # This is because now we have a subgrpah instead of the full graph and need to update the node indices
                    mapped = torch.where(
                        query_nodes >= 0,
                        global_to_local[query_nodes],
                        torch.full_like(query_nodes, -1),
                    ).to(torch.long)
                    # Update the graph to the sampled subgraph
                    graph = graph_sub.to(device)
                    # Update the node indices to the mapped indices
                    query_batch.node_embedding_indices = mapped
                else:
                    graph = None
            else:
                graph = full_graph.to(device)

            relation_graph_inp = relation_graph
            log_cardinality_pred, avg_time = model(data=graph, relation_graph=relation_graph_inp, batch=query_batch.to(device), graph_size=graph_size, repeat=repeat)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            
            # Apply ReLU during evaluation (inference logic)
            log_cardinality_pred = torch.relu(log_cardinality_pred)

            true_cardinality = query_batch.y.to(torch.float32).to(device)
            log_true_cardinality = torch.log1p(true_cardinality)

            true_cardinality = true_cardinality.detach().cpu().numpy()
            # should only contain one element
            true_cardinality = true_cardinality[0]
            dataset_name = full_graph.dataset_name
            query_type = query_batch.query_set_name[0]
            logging.info(f"Dataset: {dataset_name}, Query Type: {query_type}, True Cardinality: {true_cardinality}, Avg Inference Time over {repeat} runs: {avg_time:.6f} seconds")
            
            # Write to CSV immediately
            pd.DataFrame([[dataset_name, query_type, true_cardinality, avg_time]], 
                        columns=['dataset', 'query_type', 'true_cardinality', 'avg_inference_time']).to_csv(results_path, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()