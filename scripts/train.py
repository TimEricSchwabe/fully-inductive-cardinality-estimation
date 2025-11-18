# Standard library imports
import json
import logging
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal

# Third-party imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data
from utils.multigraph import MultigraphLoader
from utils.neighborhood_sampler import MultigraphNeighborSampler
from tqdm import tqdm

# Configure matplotlib for server environments
matplotlib.use('Agg')

# Project imports
import utils.metrics as metrics
from fice.models import FICE, FiceBinary
from utils.checkpoint_helper import load_checkpoint, save_checkpoint
from utils.config import DataConfig, load_config_yaml, Config
from utils.data_helper import load_data, train_test_split
from utils.helper import get_device, initialize_wandb, parse_arguments, process_seed, set_random_seeds, setup_logging
from utils.models import TrainingContext, ModelComponents, DataComponents, TrainingState
from torch_geometric.utils import subgraph

# Try to import the low-level NeighborSampler (newer API). If unavailable, we'll fallback.
try:
    from torch_geometric.sampler import NeighborSampler as PYGNeighborSampler  # type: ignore
    from torch_geometric.sampler import NodeSamplerInput  # type: ignore
    HAS_PYG_SAMPLER = True
except Exception:
    HAS_PYG_SAMPLER = False

logger = logging.getLogger(__name__)


TRAIN_METRICS_FILE_NAME = "train_training_metrics.csv"
TEST_METRICS_FILE_NAME = "test_training_metrics.csv"
TRAIN_METRICS_PER_GRAPH_FILE_NAME = "train_metrics_per_graph.csv"
TEST_METRICS_PER_GRAPH_FILE_NAME = "test_metrics_per_graph.csv"
TRAIN_VALUES_PER_GRAPH_FILE_NAME = "train_values_per_graph.csv"
TEST_VALUES_PER_GRAPH_FILE_NAME = "test_values_per_graph.csv"

sampler_per_graph: Dict[str, PYGNeighborSampler] = {}

def create_reusable_sampler(data_graph, num_neighbors):
    """Create a reusable sampler if the low-level sampler API is available.

    Returns sampler or None.
    """
    if not HAS_PYG_SAMPLER:
        return None
    # Newer API takes a Data object / graph store
    sampler = PYGNeighborSampler(
        data_graph,
        num_neighbors=num_neighbors,
        replace=False,
    )
    return sampler


def sample_with_sampler(sampler, data_graph, seeds: torch.Tensor):
    # 1) Sample neighborhood
    # Wrap seeds in NodeSamplerInput
    inputs = NodeSamplerInput(input_id=None, node=seeds)
    # get output with fields
    # node: sampled node ids (global ids)
    # row: row indices of the sampled edges (local indices)
    # col: column indices of the sampled edges (local indices)
    # edge: edge indices of the sampled edges (original edge ids from full graph)
    out = sampler.sample_from_nodes(inputs)

    # 2) Get sampled global node IDs
    n_id = getattr(out, 'node', None)
    if n_id is None:
        n_id = getattr(out, 'n_id', None)
    if n_id is None:
        raise RuntimeError("Sampler output missing node ids (node / n_id)")

    # 3) Local edge_index from sampler (row/col are local indices)
    edge_index_local = torch.stack([out.row, out.col], dim=0)

    # 4) Build subgraph Data directly
    sub_data = Data()
    sub_data.edge_index = edge_index_local
    sub_data.num_nodes = n_id.numel()

    # carry edge types via original edge IDs
    # we put this as optional but should always be there
    if hasattr(data_graph, 'edge_type') and hasattr(out, 'edge'):
        sub_data.edge_type = data_graph.edge_type[out.edge]

    # 5) Global->local mapping (global ID -> local [0..N-1])
    # i.e. which id do all the global ids from the full graph have (if they dont have one, they map to -1)
    global_to_local = torch.full((data_graph.num_nodes,), -1, dtype=torch.long, device=n_id.device)
    global_to_local[n_id] = torch.arange(n_id.numel(), device=n_id.device)

    return sub_data, global_to_local.detach().cpu()


def setup_directories(config):
    """Setup project directories, timestamps, and logging."""

    run_dir = setup_run_dir(config)

    log_dir = run_dir / "logs/"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints/"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, log_dir, checkpoint_dir


def initialize_model_components(context: TrainingContext, model_state_dict: dict = None, optimizer_state_dict: dict = None) -> ModelComponents:
    """Initialize the FICE model and optimizer."""
    config = context.config
    device = context.device
    
    # Get message functions from config if available
    entity_message_func = getattr(config, 'EntityGNNMessageFunc', 'rotate')
    relation_message_func = getattr(config, 'RelationGNNMessageFunc', 'rotate')
    layer_norm = getattr(config, 'layer_norm', True)
    conditioning = getattr(config, 'Conditioning', 'all')
    
    model_type = getattr(config, 'model', 'FICE')
    
    if model_type == 'FiceBinary':
        logger.info("Initializing FiceBinary model (binary embedding ablation)")
        model = FiceBinary(vector_dim=128)
    elif model_type == 'FICE':
        logger.info("Initializing FICE model")
        model = FICE(
            entity_message_func=entity_message_func,
            relation_message_func=relation_message_func,
            conditioning=conditioning,
            layer_norm=layer_norm
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'FICE' or 'FiceBinary'")
    
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
        logger.info("Model state loaded from checkpoint.")

    model.to(device)

    # Initialize optimizer
    if config.weight_decay is not None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        logger.info("Optimizer state loaded from checkpoint.")

    return ModelComponents(model=model, optimizer=optimizer)


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


def log_predictions_to_wandb(predictions, true_values, epoch, split_name, dataset_name, use_wandb, max_points=1000):
    """
    Log predictions and true values as raw data to wandb for scatter plot visualization.
    
    Args:
        predictions: List or array of predicted values (in original cardinality space)
        true_values: List or array of true values (in original cardinality space)
        epoch: Current epoch number
        split_name: 'train' or 'test'
        dataset_name: Name of the dataset (use 'overall' for combined data)
        use_wandb: Whether wandb logging is enabled
        max_points: Maximum number of points to log (default 1000)
    """
    if not use_wandb or len(predictions) == 0:
        return
    
    try:
        # Convert to numpy arrays and clean data
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Remove infinite values and NaNs
        valid_mask = np.isfinite(predictions) & np.isfinite(true_values)
        predictions_clean = predictions[valid_mask]
        true_values_clean = true_values[valid_mask]
        
        if len(predictions_clean) == 0:
            logger.warning(f"No valid predictions for {split_name}/{dataset_name} at epoch {epoch}")
            return
        
        # Sample data if too many points
        if len(predictions_clean) > max_points:
            indices = np.random.choice(len(predictions_clean), max_points, replace=False)
            predictions_sampled = predictions_clean[indices]
            true_values_sampled = true_values_clean[indices]
        else:
            predictions_sampled = predictions_clean
            true_values_sampled = true_values_clean
        
        # Create a WandB table for scatter plot
        data = [[float(pred), float(true)] for pred, true in zip(predictions_sampled, true_values_sampled)]
        table = wandb.Table(data=data, columns=["prediction", "true_value"])
        
        # Log the table with a descriptive key and create a scatter plot
        table_key = f"{split_name}_predictions_{dataset_name}_epoch_{epoch}"
        
        # Create scatter plot directly
        wandb.log({
            table_key: table,
            "epoch": epoch
        })
        
        logger.debug(f"Successfully logged {len(predictions_sampled)} predictions for {split_name}/{dataset_name} at epoch {epoch}")
        
    except Exception as e:
        logger.error(f"Error logging predictions to wandb for {split_name}/{dataset_name} at epoch {epoch}: {e}")


def train_epoch(state: TrainingState):
    """Train the model for one epoch."""
    # Extract components for easier access
    state.model_components.model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0
    device = state.training_context.device
    epochs = state.training_context.config.epochs

    loader = state.data_components.train_loader
    total_batches = len(loader)

    epoch_train_predictions_log = []    
    epoch_train_true_values_log = []
    
    # Per-graph data collection
    per_graph_data = {}  # {dataset_name: {'predictions_log': [], 'true_values_log': [], 'loss_sum': 0.0, 'batch_count': 0}}

    progress_bar = tqdm(
        loader, desc=f"Epoch {state.training_context.current_epoch}/{epochs}", leave=False, total=total_batches
    )

    # Prepare reusable sampler once per epoch if enabled
    num_neighbors_cfg = state.training_context.config.EntityGNNNeighborhood
    # sampler = None
    # sampler_per_graph: Dict[str, PYGNeighborSampler] = {}
    neighbor_sampler = state.data_components.neighbor_sampler

    if not isinstance(state.model_components.model, FiceBinary) and num_neighbors_cfg is not None:

        for batch_idx, (full_graph, query_batch) in enumerate(progress_bar):
            # sampler = sampler_per_graph.get(full_graph.dataset_name, None)
            # if sampler is None:
            #     sampler = create_reusable_sampler(full_graph, num_neighbors_cfg)
            #     sampler_per_graph[full_graph.dataset_name] = sampler

            relation_graph = full_graph.relation_graph.to(device)
            graph_size = torch.tensor([full_graph.num_edges], device=device)


            state.model_components.optimizer.zero_grad()

            # Fast path for FiceBinary: skip neighborhood sampling & avoid moving big graphs
            if isinstance(state.model_components.model, FiceBinary):
                graph = None
                num_neighbors = False
            else:
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

            relation_graph_inp = None if isinstance(state.model_components.model, FiceBinary) else relation_graph
            log_cardinality_pred = state.model_components.model(data=graph, relation_graph=relation_graph_inp, batch=query_batch.to(device), graph_size=graph_size)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)

            true_cardinality = query_batch.y.to(torch.float32).to(state.training_context.device)
            log_true_cardinality = torch.log(true_cardinality)
            

            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)

            # Backward pass
            loss.backward()
            state.model_components.optimizer.step()

            # Move predictions to CPU for logging to save GPU memory
            epoch_train_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
            epoch_train_true_values_log.extend(tensor_to_numpy(log_true_cardinality))
            
            # Collect per-graph data
            dataset_name = full_graph.dataset_name
            if dataset_name not in per_graph_data:
                per_graph_data[dataset_name] = {
                    'predictions_log': [],
                    'true_values_log': [],
                    'loss_sum': 0.0,
                    'batch_count': 0
                }
            per_graph_data[dataset_name]['predictions_log'].extend(tensor_to_numpy(log_cardinality_pred))
            per_graph_data[dataset_name]['true_values_log'].extend(tensor_to_numpy(log_true_cardinality))
            per_graph_data[dataset_name]['loss_sum'] += loss.item()
            per_graph_data[dataset_name]['batch_count'] += 1

            epoch_train_loss += loss.item()
            num_train_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_train_loss/num_train_batches:.4f}",
                }
            )

        
            # Log batch-level metrics to wandb if enabled
            if state.training_context.use_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid spam
                wandb.log({
                    f"train_loss_epoch_{state.training_context.current_epoch}": loss.item(),
                    f"train_loss_epoch_{state.training_context.current_epoch}_{full_graph.dataset_name}": loss.item(),
                })

            # Clean up batch tensors to free GPU memory
            del log_cardinality_pred, true_cardinality, log_true_cardinality, loss
            if graph is not None:
                del graph

    progress_bar.close()
    # Clear GPU cache after training epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    epoch_avg_train_loss = epoch_train_loss / num_train_batches
    
    # Convert overall predictions to original cardinality space for logging
    epoch_train_predictions = np.exp(epoch_train_predictions_log)
    epoch_train_true_values = np.exp(epoch_train_true_values_log)
    
    # Calculate per-graph metrics
    per_graph_metrics = {}
    for dataset_name, data in per_graph_data.items():
        predictions_log = np.array(data['predictions_log'])
        true_values_log = np.array(data['true_values_log'])
        
        # Convert to original cardinality space for metrics
        predictions = np.exp(predictions_log)
        true_values = np.exp(true_values_log)
        
        avg_loss = data['loss_sum'] / data['batch_count']
        mse = metrics.calculate_mse(predictions, true_values)
        mae = metrics.calculate_mae(predictions, true_values)
        rmse = np.sqrt(mse)
        q_error = metrics.calculate_q_error(predictions, true_values)
        
        per_graph_metrics[dataset_name] = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'q_error': q_error,
            'predictions_log': predictions_log,
            'true_values_log': true_values_log,
            'predictions': predictions,
            'true_values': true_values
        }

    # Log predictions to wandb for overall training data
    log_predictions_to_wandb(
        predictions=epoch_train_predictions_log,
        true_values=epoch_train_true_values_log,
        epoch=state.training_context.current_epoch,
        split_name="train",
        dataset_name="overall",
        use_wandb=state.training_context.use_wandb
    )
    
    # Log predictions to wandb for each graph
    for dataset_name, graph_metrics in per_graph_metrics.items():
        log_predictions_to_wandb(
            predictions=graph_metrics['predictions_log'],
            true_values=graph_metrics['true_values_log'],
            epoch=state.training_context.current_epoch,
            split_name="train",
            dataset_name=dataset_name,
            use_wandb=state.training_context.use_wandb
        )

    return (
        epoch_avg_train_loss,
        epoch_train_predictions_log,
        epoch_train_true_values_log,
        per_graph_metrics,
    )


def evaluate_model(state: TrainingState, mode: Literal['validation', 'test'] = 'validation'):   
    """Evaluate the model on test data."""
    # Extract components for easier access
    device = state.training_context.device
    use_wandb = state.training_context.use_wandb

    state.model_components.model.eval()

    epoch_test_loss = 0.0
    epoch_test_predictions_log = []
    epoch_test_true_values_log = []
    
    # Per-graph data collection
    per_graph_data = {}  # {dataset_name: {'predictions_log': [], 'true_values_log': [], 'loss_sum': 0.0, 'batch_count': 0}}

    loader = state.data_components.validation_loader if mode == 'validation' else state.data_components.test_loader 
    total_batches = len(loader)

    with torch.no_grad():
        # Prepare reusable sampler once for evaluation
        num_neighbors_eval = getattr(state.training_context.config, 'EntityGNNNeighborhood', None)
        neighbor_sampler = state.data_components.neighbor_sampler

        for batch_idx, (full_graph, query_batch) in tqdm(enumerate(loader), desc="Evaluating", leave=False, total=total_batches):
            relation_graph = full_graph.relation_graph.to(device)
            graph_size = torch.tensor([full_graph.num_edges], device=device)

            if isinstance(state.model_components.model, FiceBinary):
                graph = None
                relation_graph_inp = None
            else:
                if num_neighbors_eval is not None:
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

            # Forward pass
            log_cardinality_pred = state.model_components.model(data=graph, relation_graph=relation_graph_inp, batch=query_batch.to(device), graph_size=graph_size)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            
            true_cardinality = query_batch.y.to(torch.float32).to(device)
            log_true_cardinality = torch.log(true_cardinality)
            
            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)
            
            # Move to CPU immediately for logging to save GPU memory
            epoch_test_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
            epoch_test_true_values_log.extend(tensor_to_numpy(log_true_cardinality))
            
            # Collect per-graph data
            dataset_name = full_graph.dataset_name
            if dataset_name not in per_graph_data:
                per_graph_data[dataset_name] = {
                    'predictions_log': [],
                    'true_values_log': [],
                    'loss_sum': 0.0,
                    'batch_count': 0
                }
            per_graph_data[dataset_name]['predictions_log'].extend(tensor_to_numpy(log_cardinality_pred))
            per_graph_data[dataset_name]['true_values_log'].extend(tensor_to_numpy(log_true_cardinality))
            per_graph_data[dataset_name]['loss_sum'] += loss.item()
            per_graph_data[dataset_name]['batch_count'] += 1

            epoch_test_loss += loss.item()

            # Log batch-level metrics to wandb if enabled
            if use_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid spam
                wandb.log({
                    f"test_loss_epoch_{state.training_context.current_epoch}": loss.item(),
                    f"test_loss_epoch_{state.training_context.current_epoch}_{full_graph.dataset_name}": loss.item(),
                })

            # Clean up batch tensors to free GPU memory
            del log_cardinality_pred, true_cardinality, log_true_cardinality, loss
            if graph is not None:
                del graph



    # Clear GPU cache after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_test_loss = epoch_test_loss / total_batches
    
    # Calculate per-graph metrics
    per_graph_metrics = {}
    for dataset_name, data in per_graph_data.items():
        predictions_log = np.array(data['predictions_log'])
        true_values_log = np.array(data['true_values_log'])
        
        # Convert to original cardinality space for metrics
        predictions = np.exp(predictions_log)
        true_values = np.exp(true_values_log)
        
        avg_loss = data['loss_sum'] / data['batch_count']
        mse = metrics.calculate_mse(predictions, true_values)
        mae = metrics.calculate_mae(predictions, true_values)
        rmse = np.sqrt(mse)
        q_error = metrics.calculate_q_error(predictions, true_values)
        
        per_graph_metrics[dataset_name] = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'q_error': q_error,
            'predictions_log': predictions_log,
            'true_values_log': true_values_log,
            'predictions': predictions,
            'true_values': true_values
        }

    # Log predictions to wandb for overall test data
    epoch_test_predictions = np.exp(epoch_test_predictions_log)
    epoch_test_true_values = np.exp(epoch_test_true_values_log)
    log_predictions_to_wandb(
        predictions=epoch_test_predictions_log,
        true_values=epoch_test_true_values_log,
        epoch=state.training_context.current_epoch,
        split_name="test",
        dataset_name="overall",
        use_wandb=use_wandb
    )
    
    # Log predictions to wandb for each graph
    for dataset_name, graph_metrics in per_graph_metrics.items():
        log_predictions_to_wandb(
            predictions=graph_metrics['predictions_log'],
            true_values=graph_metrics['true_values_log'],
            epoch=state.training_context.current_epoch,
            split_name="test",
            dataset_name=dataset_name,
            use_wandb=use_wandb
        )

    return (
        avg_test_loss,
        epoch_test_predictions_log,
        epoch_test_true_values_log,
        per_graph_metrics,
    )


def train_model(
    state: TrainingState 
):
    """Main training loop with progress tracking and evaluation."""

    # Initialize lists for efficient data accumulation (avoid repeated DataFrame concatenation)
    train_metrics_data = []
    test_metrics_data = []
    train_metrics_per_graph_data = []
    test_metrics_per_graph_data = []
    train_predictions_per_graph_data = []
    test_predictions_per_graph_data = []
    epochs = state.training_context.config.epochs
    logger.info("Data accumulation lists initialized for efficient metric tracking!")
    logger.info(f"Starting training for {epochs} epochs...")

    device = state.training_context.device
    if torch.cuda.is_available() and device.type != 'cuda':
        logger.warning(f"Using GPU: {torch.cuda.get_device_name(device)} even though cuda is available. (torch.cuda.is_available() = {torch.cuda.is_available()}).")



    # Training loop
    for epoch in range(max(state.training_context.current_epoch, 1), epochs + 1):
        state.training_context.current_epoch = epoch  # Update current epoch in context

        # Training phase
        (
            epoch_avg_train_loss,
            epoch_train_predictions_log,
            epoch_train_true_values_log,
            train_per_graph_metrics,
        ) = train_epoch(state)

        # Calculate training metrics
        # Use original cardinality space for all metrics

        epoch_train_predictions = np.exp(epoch_train_predictions_log)
        epoch_train_true_values = np.exp(epoch_train_true_values_log)
        mse = metrics.calculate_mse(epoch_train_predictions, epoch_train_true_values)
        mae = metrics.calculate_mae(epoch_train_predictions, epoch_train_true_values)
        rmse = np.sqrt(mse)

        q_error = metrics.calculate_q_error(
            epoch_train_predictions, epoch_train_true_values
        )

        # Append training metrics to list (much more efficient than DataFrame concatenation)
        train_metrics_data.append({
            'epoch': state.training_context.current_epoch,
            'loss': epoch_avg_train_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'q_error': q_error
        })
        
        # Store per-graph metrics
        for dataset_name, graph_metrics in train_per_graph_metrics.items():
            train_metrics_per_graph_data.append({
                'epoch': state.training_context.current_epoch,
                'dataset_name': dataset_name,
                'loss': graph_metrics['loss'],
                'mse': graph_metrics['mse'],
                'mae': graph_metrics['mae'],
                'rmse': graph_metrics['rmse'],
                'q_error': graph_metrics['q_error']
            })
            
            # Store per-graph predictions
            for i, (pred_log, true_log, pred, true) in enumerate(zip(
                graph_metrics['predictions_log'],
                graph_metrics['true_values_log'],
                graph_metrics['predictions'],
                graph_metrics['true_values']
            )):
                train_predictions_per_graph_data.append({
                    'epoch': state.training_context.current_epoch,
                    'dataset_name': dataset_name,
                    'batch_idx': i,
                    'prediction_log': pred_log,
                    'true_value_log': true_log,
                    'prediction': pred,
                    'true_value': true
                })
        
        # Save DataFrames to CSV periodically (every 10 epochs) to avoid excessive I/O
        if (state.training_context.current_epoch) % 10 == 0 or (state.training_context.current_epoch) == epochs:
            train_metrics_df = pd.DataFrame(train_metrics_data)
            train_predictions_per_graph_df = pd.DataFrame(train_predictions_per_graph_data)
            train_metrics_per_graph_df = pd.DataFrame(train_metrics_per_graph_data)
            
            state.train_predictions = train_predictions_per_graph_df
            state.train_metrics = train_metrics_df
            
            train_metrics_csv_path = state.training_context.log_dir / TRAIN_METRICS_FILE_NAME
            train_predictions_per_graph_csv_path = state.training_context.log_dir / TRAIN_VALUES_PER_GRAPH_FILE_NAME
            train_metrics_per_graph_csv_path = state.training_context.log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME
            
            train_metrics_df.to_csv(train_metrics_csv_path, index=False)
            train_predictions_per_graph_df.to_csv(train_predictions_per_graph_csv_path, index=False)
            train_metrics_per_graph_df.to_csv(train_metrics_per_graph_csv_path, index=False)

        # Log training metrics to wandb
        if state.training_context.use_wandb:
            wandb.log({
                "train_loss": epoch_avg_train_loss,
                "train_mse": mse,
                "train_mae": mae,
                "train_rmse": rmse,
                "train_q_error": q_error,
                "epoch": state.training_context.current_epoch
            })
            
            # Log per-graph training metrics
            for dataset_name, graph_metrics in train_per_graph_metrics.items():
                wandb.log({
                    f"train_loss_{dataset_name}": graph_metrics['loss'],
                    f"train_mse_{dataset_name}": graph_metrics['mse'],
                    f"train_mae_{dataset_name}": graph_metrics['mae'],
                    f"train_rmse_{dataset_name}": graph_metrics['rmse'],
                    f"train_q_error_{dataset_name}": graph_metrics['q_error'],
                    "epoch": state.training_context.current_epoch
                })

        # Evaluation phase
        if (epoch) % state.training_context.config.eval_every == 0:
            (
                avg_test_loss,
                epoch_test_predictions_log,
                epoch_test_true_values_log,
                test_per_graph_metrics,
            ) = evaluate_model(state)
            # Use original cardinality space for all metrics
            epoch_test_predictions = np.exp(epoch_test_predictions_log)
            epoch_test_true_values = np.exp(epoch_test_true_values_log)
            # Calculate test metrics
            # Use original cardinality space for MSE, MAE, RMSE
            mse = metrics.calculate_mse(epoch_test_predictions, epoch_test_true_values)
            mae = metrics.calculate_mae(epoch_test_predictions, epoch_test_true_values)
            rmse = np.sqrt(mse)
            
            q_error = metrics.calculate_q_error(
                epoch_test_predictions, epoch_test_true_values
            )

            # Append test metrics to list (much more efficient than DataFrame concatenation)
            test_metrics_data.append({
                'epoch': state.training_context.current_epoch,
                'loss': avg_test_loss,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'q_error': q_error
            })
            
            # Store per-graph test metrics
            for dataset_name, graph_metrics in test_per_graph_metrics.items():
                test_metrics_per_graph_data.append({
                    'epoch': state.training_context.current_epoch,
                    'dataset_name': dataset_name,
                    'loss': graph_metrics['loss'],
                    'mse': graph_metrics['mse'],
                    'mae': graph_metrics['mae'],
                    'rmse': graph_metrics['rmse'],
                    'q_error': graph_metrics['q_error']
                })
                
                # Store per-graph test predictions
                for i, (pred_log, true_log, pred, true) in enumerate(zip(
                    graph_metrics['predictions_log'],
                    graph_metrics['true_values_log'],
                    graph_metrics['predictions'],
                    graph_metrics['true_values']
                )):
                    test_predictions_per_graph_data.append({
                        'epoch': state.training_context.current_epoch,
                        'dataset_name': dataset_name,
                        'batch_idx': i,
                        'prediction_log': pred_log,
                        'true_value_log': true_log,
                        'prediction': pred,
                        'true_value': true
                    })
            
            # Save DataFrames to CSV periodically (every 10 epochs) to avoid excessive I/O
            if (state.training_context.current_epoch) % 10 == 0 or (state.training_context.current_epoch) == state.training_context.config.epochs:
                test_metrics_df = pd.DataFrame(test_metrics_data)
                test_predictions_per_graph_df = pd.DataFrame(test_predictions_per_graph_data)
                test_metrics_per_graph_df = pd.DataFrame(test_metrics_per_graph_data)
                
                state.test_predictions = test_predictions_per_graph_df
                state.test_metrics = test_metrics_df

                test_metrics_csv_path = state.training_context.log_dir / TEST_METRICS_FILE_NAME
                test_predictions_per_graph_csv_path = state.training_context.log_dir / TEST_VALUES_PER_GRAPH_FILE_NAME
                test_metrics_per_graph_csv_path = state.training_context.log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME
                
                test_metrics_df.to_csv(test_metrics_csv_path, index=False)
                test_predictions_per_graph_df.to_csv(test_predictions_per_graph_csv_path, index=False)
                test_metrics_per_graph_df.to_csv(test_metrics_per_graph_csv_path, index=False)

            # Log test metrics to wandb
            if state.training_context.use_wandb:
                wandb.log({
                    "test_loss": avg_test_loss,
                    "test_mse": mse,
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "test_q_error": q_error,
                    "epoch": state.training_context.current_epoch
                })
                
                # Log per-graph test metrics
                for dataset_name, graph_metrics in test_per_graph_metrics.items():
                    wandb.log({
                        f"test_loss_{dataset_name}": graph_metrics['loss'],
                        f"test_mse_{dataset_name}": graph_metrics['mse'],
                        f"test_mae_{dataset_name}": graph_metrics['mae'],
                        f"test_rmse_{dataset_name}": graph_metrics['rmse'],
                        f"test_q_error_{dataset_name}": graph_metrics['q_error'],
                        "epoch": state.training_context.current_epoch
                    })

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

                save_checkpoint(state, best_model_path)
                logger.info(
                    f"New best model saved with test loss: {state.best_test_loss:.6f}"
                )
                
                # Log best model as wandb artifact
                if state.training_context.use_wandb:
                    artifact = wandb.Artifact(
                        name="best_model",
                        type="model",
                        description=f"Best model checkpoint at epoch {state.training_context.current_epoch} with test loss {state.best_test_loss:.6f}"
                    )
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact)
                    
                    # Also log the best test loss metric
                    wandb.log({
                        "best_test_loss": state.best_test_loss,
                        "best_model_epoch": state.training_context.current_epoch
                    })
        else:
            logger.info(
                f"Epoch {state.training_context.current_epoch}/{state.training_context.config['epochs']} - Train Loss: {epoch_avg_train_loss:.6f}"
            )

        # Save checkpoint periodically
        if (epoch + 1) % state.training_context.config.save_every == 0:
            # Get latest test loss from test data list
            latest_test_loss = test_metrics_data[-1]['loss'] if len(test_metrics_data) > 0 else None

            checkpoint_path = state.training_context.checkpoint_dir / f"checkpoint_epoch_{state.training_context.current_epoch}.pt"
            save_checkpoint(state, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training completed!")

    # Create final DataFrames from accumulated data
    final_train_metrics_df = pd.DataFrame(train_metrics_data)
    final_test_metrics_df = pd.DataFrame(test_metrics_data)
    final_train_metrics_per_graph_df = pd.DataFrame(train_metrics_per_graph_data)
    final_test_metrics_per_graph_df = pd.DataFrame(test_metrics_per_graph_data)
    final_train_predictions_per_graph_df = pd.DataFrame(train_predictions_per_graph_data)
    final_test_predictions_per_graph_df = pd.DataFrame(test_predictions_per_graph_data)
    
    # Save final DataFrames to CSV
    train_metrics_csv_path = state.training_context.log_dir / TRAIN_METRICS_FILE_NAME
    test_metrics_csv_path = state.training_context.log_dir / TEST_METRICS_FILE_NAME
    train_metrics_per_graph_csv_path = state.training_context.log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME
    train_predictions_per_graph_csv_path = state.training_context.log_dir / TRAIN_VALUES_PER_GRAPH_FILE_NAME
    test_metrics_per_graph_csv_path = state.training_context.log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME
    test_predictions_per_graph_csv_path = state.training_context.log_dir / TEST_VALUES_PER_GRAPH_FILE_NAME

    final_train_metrics_df.to_csv(train_metrics_csv_path, index=False)
    final_test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    final_train_metrics_per_graph_df.to_csv(train_metrics_per_graph_csv_path, index=False)
    final_train_predictions_per_graph_df.to_csv(train_predictions_per_graph_csv_path, index=False)
    final_test_metrics_per_graph_df.to_csv(test_metrics_per_graph_csv_path, index=False)
    final_test_predictions_per_graph_df.to_csv(test_predictions_per_graph_csv_path, index=False)

    return {
        "best_test_loss": state.best_test_loss,
        "train_losses": final_train_metrics_df['loss'].tolist() if len(final_train_metrics_df) > 0 else [],
        "train_mse": final_train_metrics_df['mse'].tolist() if len(final_train_metrics_df) > 0 else [],
        "train_mae": final_train_metrics_df['mae'].tolist() if len(final_train_metrics_df) > 0 else [],
        "train_rmse": final_train_metrics_df['rmse'].tolist() if len(final_train_metrics_df) > 0 else [],
        "train_q_error": final_train_metrics_df['q_error'].tolist() if len(final_train_metrics_df) > 0 else [],
        "test_losses": final_test_metrics_df['loss'].tolist() if len(final_test_metrics_df) > 0 else [],
        "test_mse": final_test_metrics_df['mse'].tolist() if len(final_test_metrics_df) > 0 else [],
        "test_mae": final_test_metrics_df['mae'].tolist() if len(final_test_metrics_df) > 0 else [],
        "test_rmse": final_test_metrics_df['rmse'].tolist() if len(final_test_metrics_df) > 0 else [],
        "test_q_error": final_test_metrics_df['q_error'].tolist() if len(final_test_metrics_df) > 0 else [],
        "train_metrics_df": final_train_metrics_df,
        "test_metrics_df": final_test_metrics_df,
        "train_predictions_df": final_train_predictions_per_graph_df,
        "test_predictions_df": final_test_predictions_per_graph_df,
        "train_metrics_per_graph_df": final_train_metrics_per_graph_df,
        "test_metrics_per_graph_df": final_test_metrics_per_graph_df,
        "train_predictions_per_graph_df": final_train_predictions_per_graph_df,
        "test_predictions_per_graph_df": final_test_predictions_per_graph_df,
        # Placeholder for final test metrics (will be populated in main)
        "final_test_loss": None,
        "final_test_mse": None,
        "final_test_mae": None,
        "final_test_rmse": None,
        "final_test_q_error": None,
        "final_test_metrics_df": None,
        "final_test_predictions_df": None,
        "final_test_per_graph_metrics_df": None,
    }



def prepare_data(config: Union[Config, List[DataConfig]]) -> DataComponents:

    data_configs: List[DataConfig]

    if isinstance(config, Config):
        data_configs = config.data
    else:
        data_configs = config

    train_sets = [] 
    validation_sets = [] 
    train_validation_data = []
    train_batch_size = []

    test_dataset = []
    test_sets = []


    for data_config in data_configs:
        dataset_name = data_config.dataset
        queries_name = data_config.queries 
        dataset, _, queries = load_data(dataset_name=dataset_name, queries_name=queries_name)

        if data_config.type == 'train':
            train_validation_data.append(dataset)
            train_batch_size.append(data_config.batch_size)

            train_q, test_q = train_test_split(queries=queries, train_ratio=config.train_ratio, num_queries=data_config.num_queries, min_cardinality=config.min_cardinality)
            train_sets.append(train_q)
            validation_sets.append(test_q)
        elif data_config.type == 'test':
            # TODO: Filter richtig implementeiren
            test_dataset.append(dataset)
            test_sets.append(queries[:data_config.num_queries])
    

    train_loader = MultigraphLoader(
        data=train_validation_data,
        queries=train_sets,
        batch_size=train_batch_size,
        shuffle=True,
    )
    validation_loader = MultigraphLoader(
        data=train_validation_data,
        queries=validation_sets,
        batch_size=1,
        shuffle=True,
    )
    test_loader = MultigraphLoader(
        data=test_dataset,
        queries=test_sets,
        batch_size=1,
        shuffle=True,
    )

    data_component = DataComponents(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


    return data_component



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
    (
        final_test_loss,
        final_test_predictions_log,
        final_test_true_values_log,
        final_test_per_graph_metrics,
    ) = evaluate_model(state, mode='test')
    
    # Calculate final test metrics in original cardinality space
    final_test_predictions = np.exp(final_test_predictions_log)
    final_test_true_values = np.exp(final_test_true_values_log)
    
    final_test_mse = metrics.calculate_mse(final_test_predictions, final_test_true_values)
    final_test_mae = metrics.calculate_mae(final_test_predictions, final_test_true_values)
    final_test_rmse = np.sqrt(final_test_mse)
    final_test_q_error = metrics.calculate_q_error(final_test_predictions, final_test_true_values)
    
    # Create DataFrames for final test results
    final_test_metrics_df = pd.DataFrame([{
        'loss': final_test_loss,
        'mse': final_test_mse,
        'mae': final_test_mae,
        'rmse': final_test_rmse,
        'q_error': final_test_q_error
    }])
    
    # Create per-graph predictions DataFrame
    final_test_predictions_per_graph_data = []
    final_test_metrics_per_graph_data = []
    
    for dataset_name, graph_metrics in final_test_per_graph_metrics.items():
        # Store per-graph metrics
        final_test_metrics_per_graph_data.append({
            'dataset_name': dataset_name,
            'loss': graph_metrics['loss'],
            'mse': graph_metrics['mse'],
            'mae': graph_metrics['mae'],
            'rmse': graph_metrics['rmse'],
            'q_error': graph_metrics['q_error']
        })
        
        # Store per-graph predictions
        for i, (pred_log, true_log, pred, true) in enumerate(zip(
            graph_metrics['predictions_log'],
            graph_metrics['true_values_log'],
            graph_metrics['predictions'],
            graph_metrics['true_values']
        )):
            final_test_predictions_per_graph_data.append({
                'dataset_name': dataset_name,
                'batch_idx': i,
                'prediction_log': pred_log,
                'true_value_log': true_log,
                'prediction': pred,
                'true_value': true
            })
    
    final_test_predictions_per_graph_df = pd.DataFrame(final_test_predictions_per_graph_data)
    final_test_metrics_per_graph_df = pd.DataFrame(final_test_metrics_per_graph_data)
    
    # Save final test results to CSV
    final_test_metrics_csv_path = state.training_context.log_dir / 'final_test_metrics.csv'
    final_test_predictions_per_graph_csv_path = state.training_context.log_dir / 'final_test_predictions_per_graph.csv'
    final_test_metrics_per_graph_csv_path = state.training_context.log_dir / 'final_test_metrics_per_graph.csv'
    
    final_test_metrics_df.to_csv(final_test_metrics_csv_path, index=False)
    final_test_predictions_per_graph_df.to_csv(final_test_predictions_per_graph_csv_path, index=False)
    final_test_metrics_per_graph_df.to_csv(final_test_metrics_per_graph_csv_path, index=False)
    
    logger.info(f"Final test results saved to {state.training_context.log_dir}")
    
    # Return results dictionary
    return {
        'final_test_loss': final_test_loss,
        'final_test_mse': final_test_mse,
        'final_test_mae': final_test_mae,
        'final_test_rmse': final_test_rmse,
        'final_test_q_error': final_test_q_error,
        'final_test_metrics_df': final_test_metrics_df,
        'final_test_predictions_df': final_test_predictions_per_graph_df,
        'final_test_per_graph_metrics_df': final_test_metrics_per_graph_df,
    }


def print_final_summary(
    training_results, run_dir, checkpoint_dir, log_dir
):
    """Display final training results and saved artifacts."""
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"\nTraining Results:")
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
        logger.info(f"\nFinal Test Results (on held-out test set):")
        logger.info(f"  Final test loss: {training_results['final_test_loss']:.6f}")
        logger.info(f"  Final test MSE: {training_results['final_test_mse']:.6f}")
        logger.info(f"  Final test MAE: {training_results['final_test_mae']:.6f}")
        logger.info(f"  Final test RMSE: {training_results['final_test_rmse']:.6f}")
        logger.info(f"  Final test Q-Error: {training_results['final_test_q_error']:.6f}")

    logger.info(f"\nSaved Artifacts:")
    logger.info(f"  Run directory: {run_dir}")
    logger.info(f"  Best model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  Final model: {checkpoint_dir / 'final_model.pt'}")
    logger.info(f"  Training metrics CSV: {log_dir / TRAIN_METRICS_FILE_NAME}")
    logger.info(f"  Validation metrics CSV: {log_dir / TEST_METRICS_FILE_NAME}")
    logger.info(f"  Training predictions per graph CSV: {log_dir / TRAIN_VALUES_PER_GRAPH_FILE_NAME}")
    logger.info(f"  Validation predictions per graph CSV: {log_dir / TEST_VALUES_PER_GRAPH_FILE_NAME}")
    logger.info(f"  Training metrics per graph CSV: {log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME}")
    logger.info(f"  Validation metrics per graph CSV: {log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME}")
    if training_results.get('final_test_loss') is not None:
        logger.info(f"  Final test metrics CSV: {log_dir / 'final_test_metrics.csv'}")
        logger.info(f"  Final test predictions per graph CSV: {log_dir / 'final_test_predictions_per_graph.csv'}")
        logger.info(f"  Final test metrics per graph CSV: {log_dir / 'final_test_metrics_per_graph.csv'}")
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
    else:
        default_config_path = "configs/train_local.yaml"
        config = load_config_yaml(args.config or default_config_path)
        model_state_dict = None
        optimizer_state_dict = None


    run_dir, log_dir, checkpoint_dir = setup_directories(config)

    # Setup logging
    setup_logging(log_dir)
    logger.info("Starting training script...")
    logger.info("All imports successful!")
    
    # Log original configuration
    logger.info("Original Configuration from file:")
    for key, value in config.model_dump().items():
        logger.info(f"  {key}: {value}")
    

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

    # Log final configuration (after merging with command-line args)
    logger.info("Final Configuration (after command-line overrides):")
    for key, value in config.model_dump().items():
        logger.info(f"  {key}: {value}")

    # Set random seeds for reproducibility
    processed_seed = process_seed(config.seed)
    set_random_seeds(processed_seed)
    logger.info(f"Random seed set to: {processed_seed}")

    # Setup device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Initialize Weights & Biases
    use_wandb = args.wandb
    #use_wandb = False # todo
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

    # Load data
    data_component = prepare_data(config)
    data_component.neighbor_sampler = MultigraphNeighborSampler(num_neighbors=config.EntityGNNNeighborhood)


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

        wandb.log({
            "dataset_total_queries": total_train + total_test,
            "dataset_train_queries": total_train,
            "dataset_test_queries": total_test,
            "dataset_num_nodes": total_nodes,
            "dataset_num_edges": total_edges,
        })


    # Initialize model and optimizer
    logger.info("Initializing model...")
    model_components = initialize_model_components(context, model_state_dict=model_state_dict, optimizer_state_dict=optimizer_state_dict)

    model_components.model = model_components.model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model_components.model.parameters())
    trainable_params = sum(p.numel() for p in model_components.model.parameters() if p.requires_grad)

    logger.info(f"Model initialized successfully!")
    logger.info(f"Model: {model_components.model.__class__.__name__}")
    logger.info(f"{model_components.model}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model device: {next(model_components.model.parameters()).device}")

    logger.info(f"Optimizer: {model_components.optimizer.__class__.__name__}")

    logger.info(f"Learning rate: {model_components.optimizer.param_groups[0]['lr']}")
    logger.info(f"Weight decay: {model_components.optimizer.param_groups[0]['weight_decay']}")

    if use_wandb:
        wandb.watch(model_components.model, log="all", log_freq=100)

    # Log model information to wandb
    if use_wandb:
        total_params = sum(p.numel() for p in model_components.model.parameters())
        trainable_params = sum(p.numel() for p in model_components.model.parameters() if p.requires_grad)
        wandb.log({
            "model_total_parameters": total_params,
            "model_trainable_parameters": trainable_params,
            "model_name": model_components.model.__class__.__name__
        })
        
    state = TrainingState(
        best_test_loss=float('inf'),
        train_metrics = None,
        test_metrics = None,
        train_predictions = None,
        test_predictions = None,
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
        
        wandb.log(final_metrics)
        
        # Save training data
        artifact = wandb.Artifact("training_data", type="dataset")
        if (state.training_context.log_dir / TRAIN_METRICS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_FILE_NAME))
        if (state.training_context.log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_PER_GRAPH_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_PER_GRAPH_FILE_NAME))
        if (state.training_context.log_dir / TRAIN_VALUES_PER_GRAPH_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_VALUES_PER_GRAPH_FILE_NAME))
        if (state.training_context.log_dir / TEST_VALUES_PER_GRAPH_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_VALUES_PER_GRAPH_FILE_NAME))
        
        # Add final test results to artifact if available
        if (state.training_context.log_dir / 'final_test_metrics.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_metrics.csv'))
        if (state.training_context.log_dir / 'final_test_predictions_per_graph.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_predictions_per_graph.csv'))
        if (state.training_context.log_dir / 'final_test_metrics_per_graph.csv').exists():
            artifact.add_file(str(state.training_context.log_dir / 'final_test_metrics_per_graph.csv'))
        
        wandb.log_artifact(artifact)
        
        
        wandb.finish()
        logger.info("Weights & Biases run finished")


if __name__ == "__main__":
    main()
