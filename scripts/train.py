# Standard library imports
import json
import logging
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Literal
import pprint

# Third-party imports
import matplotlib
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm



# Try to import the low-level NeighborSampler (newer API)
try:
    HAS_PYG_SAMPLER = True
except Exception:
    HAS_PYG_SAMPLER = False

# Configure matplotlib for server environments
matplotlib.use('Agg')

# Project imports
from fice.models import FICE, FiceBinary, FiceIndependent
from utils.checkpoint_helper import load_checkpoint, save_checkpoint
from utils.config import Config, load_config_yaml
from utils.data_helper import prepare_data
from utils.helper import (
    get_device,
    initialize_wandb,
    parse_arguments,
    process_seed,
    set_random_seeds,
    setup_logging,
    analyze_vector_similarity,
    wandb_log_artifact_with_retry,
    wandb_log_with_retry,
)
from utils.logging_helper import (
    TRAIN_METRICS_FILE_NAME,
    TEST_METRICS_FILE_NAME,
    TRAIN_METRICS_PER_GRAPH_FILE_NAME,
    TEST_METRICS_PER_GRAPH_FILE_NAME,
    TRAIN_METRICS_PER_QUERY_TYPE_FILE_NAME,
    TEST_METRICS_PER_QUERY_TYPE_FILE_NAME,
    TRAIN_METRICS_PER_QUERY_SET_NAME_FILE_NAME,
    TEST_METRICS_PER_QUERY_SET_NAME_FILE_NAME,
    PREDICTIONS_FILE_NAME,
    MetricLogger,
)
from utils.models import ModelComponents, TrainingContext, TrainingState
from utils.neighborhood_sampler import MultigraphNeighborSampler
from torch_geometric.utils import k_hop_subgraph
logger = logging.getLogger(__name__)

def sample_connected_subgraph(data, num_hops=3, max_nodes=500):
    """
    Extracts a connected patch of the graph to visualize structure.
    """
    g = torch.Generator()
    g.seed()

    # 1. Pick a random start node
    start_node = torch.randint(0, data.num_nodes, (1,), generator=g).item()
    
    # 2. Get k-hop neighbors (BFS expansion)
    # Increase num_hops until we have enough nodes
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=start_node, 
        num_hops=num_hops, 
        edge_index=data.edge_index, 
        relabel_nodes=True
    )
    
    # Safety: If too huge, trim it (though k-hop is usually safe)
    if subset.size(0) > max_nodes:
        # If too big, just take a smaller random subset of the valid connected subset
        # and induce the subgraph again to keep connections
        perm = torch.randperm(subset.size(0))[:max_nodes]
        subset = subset[perm]
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

    print(f"Sampled subgraph: {subset.size(0)} nodes.")
    
    # Create new Data object for the visualization function
    return Data(edge_index=sub_edge_index, num_nodes=subset.size(0))


def setup_directories(config):
    """Setup project directories, timestamps, and logging."""

    run_dir = setup_run_dir(config)

    log_dir = run_dir / "logs/"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints/"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, log_dir, checkpoint_dir


def initialize_model_components(context: TrainingContext, total_training_steps: int, model_state_dict: dict = None, optimizer_state_dict: dict = None, scheduler_state_dict: dict = None) -> ModelComponents:
    """Initialize the FICE model and optimizer."""
    config = context.config
    device = context.device
    
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
    
    
    if model_type == 'FiceBinary':
        logger.info(f"Initializing FiceBinary model (dim={vector_dim})")
        model = FiceBinary(vector_dim=vector_dim, skip_connection=query_gnn_skip_connection)
    elif model_type == 'FICE':
        logger.info(f"Initializing FICE model (dim={vector_dim}, layers={n_embedding_layers})")
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
    elif model_type == 'FiceIndependent':
        logger.info(f"Initializing FiceIndependent model (dim={vector_dim})")
        model = FiceIndependent(vector_dim=vector_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'FICE' or 'FiceBinary' or 'FiceIndependent'")
    
    if model_state_dict is not None:
        # TODO: this might not work if different models are used, model fo the state_dict and the current model must match
        model.load_state_dict(model_state_dict)

    model.to(device)

    # Initialize optimizer
    if config.weight_decay is not None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif model_type == 'FiceIndependent':
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # Initialize scheduler (CosineAnnealingLR)
    scheduler = None
    if optimizer is not None and config.scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_training_steps,
            eta_min=config.scheduler.min_lr
        )
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

    return ModelComponents(model=model, optimizer=optimizer, scheduler=scheduler)


def tensor_to_numpy(tensor):
    """Helper function to convert tensors to numpy."""
    return tensor.detach().cpu().numpy().tolist()


def log_gpu_memory_usage(step_name: str):
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        logger.debug(f"GPU Memory at {step_name}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        logger.debug(f"CPU-only mode at {step_name}")



def prepare_bipartite_graph(full_graph, query_batch, neighbor_sampler, device):
    full_bipartite_graph = full_graph.bipartite_graph.to(device)
    offset = full_graph.num_nodes
    seeds = torch.cat((query_batch.node_embedding_indices, query_batch.edge_embedding_indices + offset))
    seeds = seeds[seeds >= 0].unique().detach().cpu()
    if seeds.numel() > 0:
        # bipartite_graph_global_to_local of shape fullbipartite_graph.num_nodes
        bipartite_graph, bipartite_graph_gloal_to_local = neighbor_sampler.sample(full_bipartite_graph, seeds=seeds)
        # TODO: IN den sampler schreiben
        bipartite_graph.original_num_relations = full_bipartite_graph.original_num_relations
        bipartite_graph.original_num_nodes = full_bipartite_graph.original_num_nodes

        # TODO: in make_bipartite_graph schreiben
        min_relation_index = offset # inclusive
        max_relation_index = offset + full_bipartite_graph.original_num_relations # exclusive
        mapped_relation_indices = bipartite_graph_gloal_to_local[min_relation_index:max_relation_index] 


        min_entity_index = 0 # inclusive
        max_entity_index = full_bipartite_graph.original_num_nodes # exclusive
        mapped_entity_indices = bipartite_graph_gloal_to_local[min_entity_index:max_entity_index]

        bipartite_graph.mapped_relation_indices = mapped_relation_indices
        bipartite_graph.mapped_entity_indices = mapped_entity_indices
        bipartite_graph = bipartite_graph.to(device)

    return bipartite_graph, bipartite_graph_gloal_to_local



def prepare_bipartite_relation_graph(full_graph, bipartite_graph, bipartite_graph_gloal_to_local, query_batch, device):
    offset = full_graph.num_nodes
    # TODO: typo in edge_emebeddings => edge_embeddings
    edge_emebeddings = query_batch.edge_embedding_indices.detach().cpu()

    edge_emebeddings = torch.where(
        edge_emebeddings >= 0,
        bipartite_graph_gloal_to_local[edge_emebeddings + offset],
        torch.full_like(edge_emebeddings, -1),
    ).to(torch.long)
    query_batch.original_edge_embedding_indices = query_batch.edge_embedding_indices 
    query_batch.edge_embedding_indices = edge_emebeddings.to(device)

    return bipartite_graph


def train_epoch(state: TrainingState):
    """Train the model for one epoch."""
    # Extract components for easier access
    state.model_components.model.train()
    state.metric_logger.start_epoch(epoch=state.training_context.current_epoch, stage='train')
    
    epoch_train_loss = 0.0
    num_train_batches = 0
    device = state.training_context.device
    epochs = state.training_context.config.epochs

    loader = state.data_components.train_loader
    total_batches = len(loader)

    plot_embeddings = False

    progress_bar = tqdm(
        loader, desc=f"Epoch {state.training_context.current_epoch}/{epochs}", leave=False, total=total_batches
    )

    # Prepare reusable sampler once per epoch if enabled
    num_neighbors_cfg = state.training_context.config.GNNNeighborhood

    neighbor_sampler = state.data_components.neighbor_sampler

    if not isinstance(state.model_components.model, FiceBinary) and num_neighbors_cfg is not None:

        for batch_idx, (full_graph, query_batch) in enumerate(progress_bar):

            graph_size = torch.tensor([full_graph.num_edges], device=device)

            bipartite_graph = None
            bipartite_graph_gloal_to_local = None
            
            use_bipartite_entity = state.training_context.config.entity_gnn.model == "BipartiteGraphGNN"
            use_bipartite_relation = state.training_context.config.relation_gnn.model == "BipartiteGraphGNN"

            # generate a sample of the bipartite graph if needed
            if use_bipartite_entity or use_bipartite_relation:
                bipartite_graph, bipartite_graph_gloal_to_local = prepare_bipartite_graph(full_graph, query_batch, neighbor_sampler, device)


            
            # Choose correct graph for relation GNN
            if use_bipartite_relation:
                relation_graph = prepare_bipartite_relation_graph(full_graph, bipartite_graph, bipartite_graph_gloal_to_local, query_batch, device)
                
            else: 
                relation_graph = full_graph.relation_graph.to(device)


            if plot_embeddings:
                figure_dir = state.training_context.log_dir / "figures"
                with torch.no_grad():
                    if state.model_components.model.use_unified_bipartite_graph:
                        edge_embeddings = state.model_components.model.embedding_model(data=relation_graph)
                        edge_embeddings = edge_embeddings.detach().cpu().squeeze()
                        state.metric_logger.log_embedding_similarity(edge_embeddings, "Embeddings", state.training_context.current_epoch)
                        analyze_vector_similarity(edge_embeddings, "Embeddings Epoch " + str(state.training_context.current_epoch), output_dir=figure_dir)
                    else:
                        edge_embeddings = state.model_components.model.relation_model(data=relation_graph)
                        node_embeddings = state.model_components.model.entity_model(data=full_graph.to(device), relation_representations=edge_embeddings)
                        edge_embeddings = edge_embeddings.detach().cpu().squeeze()
                        node_embeddings = node_embeddings.detach().cpu().squeeze()
                        #state.metric_logger.log_embedding_similarity(node_embeddings, "Node_Embeddings", state.training_context.current_epoch)
                        #state.metric_logger.log_embedding_similarity(edge_embeddings, "Edge_Embeddings", state.training_context.current_epoch)
                        analyze_vector_similarity(node_embeddings, "Node_Embeddings Epoch " + str(state.training_context.current_epoch), output_dir=figure_dir)
                        analyze_vector_similarity(edge_embeddings, "Edge_Embeddings Epoch " + str(state.training_context.current_epoch), output_dir=figure_dir)
                    plot_embeddings = False


            # SELECT GRAPH FOR ENTITY GNN
            # Fast path for FiceBinary: skip neighborhood sampling & avoid moving big graphs
            if isinstance(state.model_components.model, FiceBinary) or isinstance(state.model_components.model, FiceIndependent):
                graph = None
                num_neighbors = False
            else:
                if use_bipartite_entity:
                    # Use the SAME bipartite graph as used for relations
                    if bipartite_graph is not None and bipartite_graph_gloal_to_local is not None:
                        graph = bipartite_graph
                        
                        # Map entity indices using the bipartite graph's mapping
                        query_nodes = query_batch.node_embedding_indices.detach().cpu()
                        
                        # Note: In bipartite graph, entity IDs [0..N-1] correspond to global IDs [0..N-1]
                        # rel_global_to_local maps global bipartite ID to local bipartite ID
                        mapped = torch.where(
                            query_nodes >= 0,
                            bipartite_graph_gloal_to_local[query_nodes],
                            torch.full_like(query_nodes, -1),
                        ).to(torch.long)
                        
                        query_batch.node_embedding_indices = mapped.to(device)
                    else:
                        graph = None
                else:
                    # "KG" mode - Standard Logic
                    num_neighbors = num_neighbors_cfg   
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

            # Forward pass
            if state.model_components.optimizer is not None:
                state.model_components.optimizer.zero_grad()

            relation_graph_inp = None if isinstance(state.model_components.model, FiceBinary) or isinstance(state.model_components.model, FiceIndependent) else relation_graph
            log_cardinality_pred = state.model_components.model(data=graph, relation_graph=relation_graph_inp, batch=query_batch.to(device), graph_size=graph_size)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)

            true_cardinality = query_batch.y.to(torch.float32).to(state.training_context.device)
            log_true_cardinality = torch.log1p(true_cardinality)
            
            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)

            # Backward pass
            if state.model_components.optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    state.model_components.model.parameters(), 
                    max_norm=state.training_context.config.gradient_clipping
                )
                state.model_components.optimizer.step()
                if state.model_components.scheduler is not None:
                    state.model_components.scheduler.step()
                    # logger.info(f"Learning Rate after step: {state.model_components.scheduler.get_last_lr()}")

            # Log batch
            state.metric_logger.log_batch(
                predictions_log=log_cardinality_pred,
                true_values_log=log_true_cardinality,
                loss=loss,
                dataset_name=full_graph.dataset_name,
                query_type=query_batch.type,
                query_set_name=query_batch.query_set_name
            )

            epoch_train_loss += loss.item()
            num_train_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_train_loss/num_train_batches:.4f}",
                }
            )
            # Clean up batch tensors to prevent memory accumulation
            del graph, relation_graph, log_cardinality_pred, loss
            if bipartite_graph is not None:
                del bipartite_graph, bipartite_graph_gloal_to_local
            
            # Periodic cache cleanup
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()



    progress_bar.close()
    # Clear GPU cache after training epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(state: TrainingState, mode: Literal['validation', 'test'] = 'validation'):   
    """Evaluate the model on test data."""
    # Extract components for easier access
    device = state.training_context.device
    use_wandb = state.training_context.use_wandb

    state.model_components.model.eval()
    stage = 'final_test' if mode == 'test' else 'test'
    state.metric_logger.start_epoch(epoch=state.training_context.current_epoch, stage=stage)

    epoch_test_loss = 0.0
    
    loader = state.data_components.validation_loader if mode == 'validation' else state.data_components.test_loader 
    total_batches = len(loader)

    with torch.no_grad():
        # Prepare reusable sampler once for evaluation
        num_neighbors_eval = getattr(state.training_context.config, 'GNNNeighborhood', None)
        neighbor_sampler = state.data_components.neighbor_sampler

        for batch_idx, (full_graph, query_batch) in tqdm(enumerate(loader), desc="Evaluating", leave=False, total=total_batches):

            graph_size = torch.tensor([full_graph.num_edges], device=device)

            bipartite_graph = None
            bipartite_graph_gloal_to_local = None

            use_bipartite_entity = state.training_context.config.entity_gnn.model == "BipartiteGraphGNN"
            use_bipartite_relation = state.training_context.config.relation_gnn.model == "BipartiteGraphGNN"

            # generate a sample of the bipartite graph if needed
            if use_bipartite_entity or use_bipartite_relation:
                bipartite_graph, bipartite_graph_gloal_to_local = prepare_bipartite_graph(full_graph, query_batch, neighbor_sampler, device)

            if use_bipartite_relation:
                relation_graph = prepare_bipartite_relation_graph(full_graph, bipartite_graph, bipartite_graph_gloal_to_local, query_batch, device)
            else: 
                relation_graph = full_graph.relation_graph.to(device)



            # SELECT GRAPH FOR ENTITY GNN
            # Fast path for FiceBinary: skip neighborhood sampling & avoid moving big graphs
            if isinstance(state.model_components.model, FiceBinary) or isinstance(state.model_components.model, FiceIndependent):
                graph = None
                num_neighbors = False
            else:
                if use_bipartite_entity:
                    # Use the SAME bipartite graph as used for relations
                    if bipartite_graph is not None and bipartite_graph_gloal_to_local is not None:
                        graph = bipartite_graph
                        
                        # Map entity indices using the bipartite graph's mapping
                        query_nodes = query_batch.node_embedding_indices.detach().cpu()
                        
                        # Note: In bipartite graph, entity IDs [0..N-1] correspond to global IDs [0..N-1]
                        # rel_global_to_local maps global bipartite ID to local bipartite ID
                        mapped = torch.where(
                            query_nodes >= 0,
                            bipartite_graph_gloal_to_local[query_nodes],
                            torch.full_like(query_nodes, -1),
                        ).to(torch.long)
                        
                        query_batch.node_embedding_indices = mapped.to(device)
                    else:
                        graph = None
                else:
                    # "KG" mode - Standard Logic
                    num_neighbors = state.training_context.config.GNNNeighborhood   
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

            # Forward pass
            relation_graph_inp = None if isinstance(state.model_components.model, FiceBinary) or isinstance(state.model_components.model, FiceIndependent) else relation_graph
            log_cardinality_pred = state.model_components.model(data=graph, relation_graph=relation_graph_inp, batch=query_batch.to(device), graph_size=graph_size)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            
            # Apply ReLU during evaluation (inference logic)
            log_cardinality_pred = torch.relu(log_cardinality_pred)

            true_cardinality = query_batch.y.to(torch.float32).to(device)
            log_true_cardinality = torch.log1p(true_cardinality)
            
            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)
            
            # Log batch
            state.metric_logger.log_batch(
                predictions_log=log_cardinality_pred,
                true_values_log=log_true_cardinality,
                loss=loss,
                dataset_name=full_graph.dataset_name,
                query_type=query_batch.type,
                query_set_name=query_batch.query_set_name
            )

            epoch_test_loss += loss.item()


    # Clear GPU cache after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_test_loss = epoch_test_loss / total_batches
    
    return avg_test_loss


def train_model(
    state: TrainingState 
):
    """Main training loop with progress tracking and evaluation."""

    epochs = state.training_context.config.epochs
    logger.info(f"Starting training for {epochs} epochs...")

    device = state.training_context.device
    if torch.cuda.is_available() and device.type != 'cuda':
        logger.warning(f"Using GPU: {torch.cuda.get_device_name(device)} even though cuda is available. (torch.cuda.is_available() = {torch.cuda.is_available()}).")

    # Training loop
    for epoch in range(max(state.training_context.current_epoch, 1), epochs + 1):
        state.training_context.current_epoch = epoch  # Update current epoch in context

        # Training phase
        train_epoch(state)
        
        # End training epoch
        should_save = (epoch % state.training_context.config.save_metrics_every == 0 or epoch == epochs)
        train_metrics = state.metric_logger.end_epoch(epoch, stage='train', save=should_save)
        epoch_avg_train_loss = train_metrics['loss']

        # Evaluation phase
        if (epoch) % state.training_context.config.eval_every == 0:
            # TODO start und end epoch entweder beide in evualte_model oder beide hier
            evaluate_model(state)
            
            # End test epoch
            test_metrics = state.metric_logger.end_epoch(epoch, stage='test', save=should_save)
            avg_test_loss = test_metrics['loss']

            # Log metrics
            logger.info(
                f"Epoch {state.training_context.current_epoch}/{state.training_context.config.epochs} - "
                f"Train Loss: {epoch_avg_train_loss:.6f}, "
                f"Test Loss: {avg_test_loss:.6f}"
            )

            # Save best model
            if avg_test_loss < state.best_test_loss:
                state.best_test_loss = avg_test_loss
                best_model_path = state.training_context.checkpoint_dir / "best_model.pt"

                if state.model_components.optimizer is not None:
                    save_checkpoint(state, best_model_path)
                    logger.info(
                        f"New best model saved with test loss: {state.best_test_loss:.6f}"
                    )
                
                # Log best model as wandb artifact
                if state.training_context.use_wandb and state.model_components.optimizer is not None:
                    artifact = wandb.Artifact(
                        name="best_model",
                        type="model",
                        description=f"Best model checkpoint at epoch {state.training_context.current_epoch} with test loss {state.best_test_loss:.6f}"
                    )
                    artifact.add_file(str(best_model_path))
                    wandb_log_artifact_with_retry(artifact)
                        
                    try:
                        # Also log the best test loss metric
                        wandb_log_with_retry({
                            "best_test_loss": state.best_test_loss,
                            "best_model_epoch": state.training_context.current_epoch
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log best test loss metrics: {e}")
            
        else:
            logger.info(
                f"Epoch {state.training_context.current_epoch}/{state.training_context.config['epochs']} - Train Loss: {epoch_avg_train_loss:.6f}"
            )

        # Save checkpoint periodically
        if (epoch + 1) % state.training_context.config.save_every == 0:
            checkpoint_path = state.training_context.checkpoint_dir / f"checkpoint_epoch_{state.training_context.current_epoch}.pt"
            save_checkpoint(state, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training completed!")
    
    history = state.metric_logger.history
        
    return {
        "best_test_loss": state.best_test_loss,
        "train_losses": [m['loss'] for m in history['train_metrics']],
        "test_losses": [m['loss'] for m in history['test_metrics']],
        "train_mae": [m['mae'] for m in history['train_metrics']],
        "test_mae": [m['mae'] for m in history['test_metrics']],
        "train_q_error": [m['q_error'] for m in history['train_metrics']],
        "test_q_error": [m['q_error'] for m in history['test_metrics']],
        "final_test_loss": None,
    }



def perform_final_test_evaluation(state: TrainingState) -> dict:
    """
    Perform final test evaluation on held-out test set after training.
    
    Args:
        state: TrainingState object containing model, data, and context
        
    Returns:
        Dictionary with test metrics and DataFrames
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON FINAL TEST SET")
    logger.info("=" * 60)
    
    # Run evaluation
    evaluate_model(state, mode='test')
    
    # Process and save results
    final_test_metrics = state.metric_logger.end_epoch(
        epoch=state.training_context.current_epoch, 
        stage='final_test', 
        save=True
    )
    
    logger.info(f"Final test results saved to {state.training_context.log_dir}")
    
    # Return results dictionary
    return {
        'final_test_loss': final_test_metrics['loss'],
        'final_test_mse': final_test_metrics['mse'],
        'final_test_mae': final_test_metrics['mae'],
        'final_test_rmse': final_test_metrics['rmse'],
        'final_test_q_error': final_test_metrics['q_error'],
        'final_test_metrics_df': None,
        'final_test_predictions_df': None,
        'final_test_per_graph_metrics_df': None,
    }


def print_final_summary(
    training_results, run_dir, checkpoint_dir, log_dir
):
    """Display final training results and saved artifacts."""
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("\nTraining Results:")
    logger.info(f"  Total epochs trained: {len(training_results['train_losses'])}")
    logger.info(f"  Best validation loss: {training_results['best_test_loss']:.6f}")
    logger.info(f"  Final train loss: {training_results['train_losses'][-1]:.6f}")
    logger.info(f"  Final validation loss: {training_results['test_losses'][-1]:.6f}")
    logger.info(f"  Final train MAE: {training_results['train_mae'][-1]:.6f}")
    logger.info(f"  Final validation MAE: {training_results['test_mae'][-1]:.6f}")
    logger.info(f"  Final train Q-Error: {training_results['train_q_error'][-1]:.6f}")
    logger.info(f"  Final validation Q-Error: {training_results['test_q_error'][-1]:.6f}")
    
    # Display final test results if available
    if training_results.get('final_test_loss') is not None:
        logger.info("\nFinal Test Results (on held-out test set):")
        logger.info(f"  Final test loss: {training_results['final_test_loss']:.6f}")
        logger.info(f"  Final test MSE: {training_results['final_test_mse']:.6f}")
        logger.info(f"  Final test MAE: {training_results['final_test_mae']:.6f}")
        logger.info(f"  Final test RMSE: {training_results['final_test_rmse']:.6f}")
        logger.info(f"  Final test Q-Error: {training_results['final_test_q_error']:.6f}")

    logger.info("\nSaved Artifacts:")
    logger.info(f"  Run directory: {run_dir}")
    logger.info(f"  Best model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  Final model: {checkpoint_dir / 'final_model.pt'}")
    logger.info(f"  Config files: {run_dir / 'config.yaml'}, {run_dir / 'config.json'}")
    logger.info(f"  Training metrics CSV: {log_dir / TRAIN_METRICS_FILE_NAME}")
    logger.info(f"  Validation metrics CSV: {log_dir / TEST_METRICS_FILE_NAME}")
    logger.info(f"  Predictions CSV (all stages): {log_dir / PREDICTIONS_FILE_NAME}")
    logger.info(f"  Training metrics per graph CSV: {log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME}")
    logger.info(f"  Validation metrics per graph CSV: {log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME}")
    logger.info(f"  Training metrics per query type CSV: {log_dir / TRAIN_METRICS_PER_QUERY_TYPE_FILE_NAME}")
    logger.info(f"  Validation metrics per query type CSV: {log_dir / TEST_METRICS_PER_QUERY_TYPE_FILE_NAME}")
    logger.info(f"  Training metrics per query set name CSV: {log_dir / TRAIN_METRICS_PER_QUERY_SET_NAME_FILE_NAME}")
    logger.info(f"  Validation metrics per query set name CSV: {log_dir / TEST_METRICS_PER_QUERY_SET_NAME_FILE_NAME}")
    if training_results.get('final_test_loss') is not None:
        logger.info(f"  Final test metrics CSV: {log_dir / 'final_test_metrics.csv'}")
        logger.info(f"  Final test metrics per graph CSV: {log_dir / 'final_test_metrics_per_graph.csv'}")
        logger.info(f"  Final test metrics per query type CSV: {log_dir / 'final_test_metrics_per_query_type.csv'}")
        logger.info(f"  Final test metrics per query set name CSV: {log_dir / 'final_test_metrics_per_query_set_name.csv'}")
    logger.info(f"  Log files: {log_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Training script execution completed!")
    logger.info("=" * 60)

def setup_run_dir(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(config.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    """Main function to orchestrate the training process."""
    # Parse arguments and load config
    args = parse_arguments()

    from_checkpoint = args.checkpoint is not None
    if from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        checkpoint = load_checkpoint(checkpoint_path=checkpoint_path)
        config = load_config_yaml(args.config) if args.config is not None else Config(**checkpoint['config'])
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
    else:
        default_config_path = "configs/train_local.yaml"
        config = load_config_yaml(args.config or default_config_path)
        model_state_dict = None
        optimizer_state_dict = None
        scheduler_state_dict = None


    run_dir, log_dir, checkpoint_dir = setup_directories(config)

    # Setup logging
    setup_logging(log_dir)
    logger.info("Starting training script...")
    logger.info("All imports successful!")
    
    # Log original configuration
    logger.info("Original Configuration from file:")
    pprint.pprint(config.model_dump())
    # for key, value in config.model_dump().items():
    #     logger.info(f"  {key}: {value}")
    

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Log initial GPU memory if available
    log_gpu_memory_usage("initialization")

    try:
        e = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
        logger.info(f"nvcc version: {e.stdout.strip()}")
    except FileNotFoundError as e:
        logger.error(f"Error occurred while checking CUDA version: {e}")

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Save config files for reproducibility
    config_dict = config.model_dump()
    config_yaml_path = run_dir / "config.yaml"
    config_json_path = run_dir / "config.json"
    
    with open(config_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Config saved to {config_yaml_path} and {config_json_path}")

    # Set random seeds for reproducibility
    processed_seed = process_seed(config.seed)
    set_random_seeds(processed_seed)
    logger.info(f"Random seed set to: {processed_seed}")

    # Setup device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Initialize Weights & Biases
    # TODO: fix
    use_wandb = config.wandb
    if use_wandb:
        initialize_wandb(config, run_dir)

    # Create training context
    context = TrainingContext(
        config=config,
        device=device,
        run_dir=run_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
    )
    
    # Initialize MetricLogger
    metric_logger = MetricLogger(log_dir=log_dir, use_wandb=use_wandb)

    # Load data
    data_component = prepare_data(config)
    data_component.neighbor_sampler = MultigraphNeighborSampler(num_neighbors=config.GNNNeighborhood)

    total_batches_per_epoch = len(data_component.train_loader)
    total_training_steps = config.epochs * total_batches_per_epoch

    # Log dataset info to wandb (updated for new prepare_data returning a single DataComponents)
    if use_wandb:
        total_train = total_test = total_nodes = total_edges = 0

        dc = data_component  # prepare_data now returns a single DataComponents

        # Resolve train/test collections (prefer explicit train_queries/test_queries, fall back to loader.dataset or loader itself)
        train_ds = getattr(dc, "train_queries", None) or getattr(getattr(dc, "train_loader", None), "dataset", None) or getattr(dc, "train_loader", None)
        test_ds = getattr(dc, "test_queries", None) or getattr(getattr(dc, "test_loader", None), "dataset", None) or getattr(dc, "test_loader", None)

        total_train += len(train_ds) if train_ds is not None else 0
        total_test += len(test_ds) if test_ds is not None else 0

        # dc.data is a list of graphs as produced by prepare_data
        for g in getattr(dc, "data", []) or []:
            total_nodes += getattr(g, "num_nodes", 0) or 0
            total_edges += getattr(g, "num_edges", 0) or 0

        wandb_log_with_retry({
            "dataset_total_queries": total_train + total_test,
            "dataset_train_queries": total_train,
            "dataset_test_queries": total_test,
            "dataset_num_nodes": total_nodes,
            "dataset_num_edges": total_edges,
        })


    # Initialize model and optimizer
    logger.info("Initializing model...")
    model_components = initialize_model_components(
        context,
        total_training_steps=total_training_steps,
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict
    )

    model_components.model = model_components.model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model_components.model.parameters())
    trainable_params = sum(p.numel() for p in model_components.model.parameters() if p.requires_grad)

    logger.info("Model initialized successfully!")
    logger.info(f"Model: {model_components.model.__class__.__name__}")
    logger.info(f"{model_components.model}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    if model_components.optimizer is not None:
        logger.info(f"Model device: {next(model_components.model.parameters()).device}")

    if model_components.optimizer is not None:
        logger.info(f"Optimizer: {model_components.optimizer.__class__.__name__}")
        logger.info(f"Learning rate: {model_components.optimizer.param_groups[0]['lr']}")
        logger.info(f"Weight decay: {model_components.optimizer.param_groups[0]['weight_decay']}")
    else:
        logger.info("No optimizer found")

    if use_wandb:
        try:
            wandb.watch(model_components.model, log="all", log_freq=100)
        except Exception as e:
            logger.warning(f"Failed to setup wandb.watch: {e}")

    # Log model information to wandb
    if use_wandb:
        total_params = sum(p.numel() for p in model_components.model.parameters())
        trainable_params = sum(p.numel() for p in model_components.model.parameters() if p.requires_grad)
        wandb_log_with_retry({
            "model_total_parameters": total_params,
            "model_trainable_parameters": trainable_params,
            "model_name": model_components.model.__class__.__name__
        })
        
    state = TrainingState(
        best_test_loss=float('inf'),
        metric_logger=metric_logger,
        training_context=context,
        model_components=model_components,
        data_components=data_component,
    )

    # Train the model
    training_results = train_model(state)

    # Perform final test evaluation
    final_test_results = perform_final_test_evaluation(state)
    training_results.update(final_test_results)

    # Print final summary
    print_final_summary(
        training_results, 
        state.training_context.run_dir, 
        state.training_context.checkpoint_dir, 
        state.training_context.log_dir
    )

    # Finalize wandb run
    # Save final model state using save_checkpoint
    final_model_path = state.training_context.checkpoint_dir / "final_model.pt"
    save_checkpoint(state, final_model_path)
    if state.training_context.use_wandb:
        # Log final metrics summary
        final_metrics = {
            "final_train_loss": training_results['train_losses'][-1],
            "final_validation_loss": training_results['test_losses'][-1],
            "final_train_mae": training_results['train_mae'][-1],
            "final_validation_mae": training_results['test_mae'][-1],
            "final_train_q_error": training_results['train_q_error'][-1],
            "final_validation_q_error": training_results['test_q_error'][-1],
            "total_epochs": len(training_results['train_losses'])
        }
        
        # Add final test metrics if available
        if training_results.get('final_test_loss') is not None:
            final_metrics.update({
                "final_test_loss": training_results['final_test_loss'],
                "final_test_mse": training_results['final_test_mse'],
                "final_test_mae": training_results['final_test_mae'],
                "final_test_rmse": training_results['final_test_rmse'],
                "final_test_q_error": training_results['final_test_q_error'],
            })
            
            # Log per-graph final test metrics
            if training_results['final_test_per_graph_metrics_df'] is not None:
                for _, row in training_results['final_test_per_graph_metrics_df'].iterrows():
                    dataset_name = row['dataset_name']
                    final_metrics.update({
                        f"final_test_loss_{dataset_name}": row['loss'],
                        f"final_test_mse_{dataset_name}": row['mse'],
                        f"final_test_mae_{dataset_name}": row['mae'],
                        f"final_test_rmse_{dataset_name}": row['rmse'],
                        f"final_test_q_error_{dataset_name}": row['q_error'],
                    })
        
        wandb_log_with_retry(final_metrics)
        
        # Save training data
        artifact = wandb.Artifact("training_data", type="dataset")
        
        # Add metrics files
        if (state.training_context.log_dir / TRAIN_METRICS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_FILE_NAME))
        if (state.training_context.log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME))
        if (state.training_context.log_dir / TRAIN_METRICS_PER_QUERY_TYPE_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_PER_QUERY_TYPE_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_PER_QUERY_TYPE_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_PER_QUERY_TYPE_FILE_NAME))
        if (state.training_context.log_dir / TRAIN_METRICS_PER_QUERY_SET_NAME_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_PER_QUERY_SET_NAME_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_PER_QUERY_SET_NAME_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_PER_QUERY_SET_NAME_FILE_NAME))
        
        # Add unified predictions file
        if (state.training_context.log_dir / PREDICTIONS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / PREDICTIONS_FILE_NAME))
        
        # Add config files
        if (state.training_context.run_dir / 'config.yaml').exists():
            artifact.add_file(str(state.training_context.run_dir / 'config.yaml'))
        if (state.training_context.run_dir / 'config.json').exists():
            artifact.add_file(str(state.training_context.run_dir / 'config.json'))
        
        # Add final test results to artifact if available
        if (state.training_context.log_dir / 'final_test_metrics.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_metrics.csv'))
        if (state.training_context.log_dir / 'final_test_metrics_per_graph.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_metrics_per_graph.csv'))
        if (state.training_context.log_dir / 'final_test_metrics_per_query_type.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_metrics_per_query_type.csv'))
        if (state.training_context.log_dir / 'final_test_metrics_per_query_set_name.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_metrics_per_query_set_name.csv'))
        
        wandb_log_artifact_with_retry(artifact)
        
        
        wandb.finish()
        logger.info("Weights & Biases run finished")


if __name__ == "__main__":
    main()
