# Standard library imports
import json
import logging
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path

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
from tqdm import tqdm

# Configure matplotlib for server environments
matplotlib.use('Agg')

# Project imports
import utils.metrics as metrics
from fice.models import FICE, FiceBinary
from utils.checkpoint_helper import load_checkpoint, save_checkpoint
from utils.config import load_config_yaml, Config
from utils.data_helper import load_dataset, load_occurrences, load_queries, train_test_split
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
TRAIN_VALUES_FILE_NAME = "train_values.csv"
TEST_VALUES_FILE_NAME = "test_values.csv"


def map_global_to_neighborhood(data_n_id, node_indices):
    """
    Maps node indices from a batch to their corresponding positions in data.n_id.
    
    Uses torch.searchsorted to efficiently find the positions of node_indices values
    within the potentially unsorted data.n_id tensor.
    
    Args:
        data_n_id: Tensor containing node IDs from the data object (potentially unsorted)
        node_indices: Tensor containing node indices to map
        
    Returns:
        Tensor of indices corresponding to positions in data.n_id
        
    Example:
        data_n_id = tensor([1, 3, 5, 2, 6])
        node_indices = tensor([5, 6, 1])
        result = tensor([2, 4, 0])  # positions of [5, 6, 1] in data_n_id
    """
    # Sort data.n_id and keep track of original indices
    sorted_n_id, sort_indices = torch.sort(data_n_id)
    
    # Find positions in sorted array
    sorted_positions = torch.searchsorted(sorted_n_id, node_indices)
    
    # Map back to original indices
    original_indices = sort_indices[sorted_positions]
    
    return original_indices


def build_neighborhood_subgraph(
    data_graph,
    query_batch,
    device: torch.device = None,
    num_neighbors=[-1],
    disjoint: bool = False,
    shuffle: bool = False,
    loader_num_workers: int = 0,
    loader_pin_memory: bool = False,
    loader_persistent_workers: bool = False,
):
    """
    Sample a neighborhood subgraph for the node indices found in `query_batch` and map
    the original `query_batch.node_embedding_indices` into the subgraph's index space.

    Args:
        data_graph: torch_geometric.data.Data (full graph)
        query_batch: batch object that contains `node_embedding_indices` (tensor with -1 placeholders)
        device: torch.device to move the sampled subgraph to
        num_neighbors: list passed to NeighborLoader (default [-1])
        disjoint: pass-through to NeighborLoader
        shuffle: pass-through to NeighborLoader

    Returns:
        (subgraph_or_none, mapped_node_indices)
        - subgraph_or_none: Data moved to `device`, or None if no positive input nodes
        - mapped_node_indices: LongTensor same shape as query_batch.node_embedding_indices where
          entries < 0 remain -1, and entries >=0 are replaced with indices into subgraph.n_id.
          The returned mapped tensor is on CPU. If a global node id is not present in the
          sampled subgraph, it will be mapped to -1.
    """
    # Extract the node indices tensor and work on CPU for mapping
    node_indices = query_batch.node_embedding_indices
    node_indices_cpu = node_indices.detach().cpu()

    positive_mask = node_indices_cpu >= 0
    positive_nodes = node_indices_cpu[positive_mask].unique()

    if positive_nodes.numel() == 0:
        # Nothing to sample; return None and a copy of the original indices
        return None, node_indices_cpu.clone()

    neighbor_loader = NeighborLoader(
        data=data_graph,
        num_neighbors=num_neighbors,
        batch_size=positive_nodes.size(0),
        shuffle=shuffle,
        disjoint=disjoint,
        input_nodes=positive_nodes,
        num_workers=loader_num_workers,
        pin_memory=loader_pin_memory,
        persistent_workers=loader_persistent_workers if loader_num_workers > 0 else False,
    )

    # There should be a single batch corresponding to the requested input_nodes
    subgraph = next(iter(neighbor_loader))

    if device is not None:
        subgraph = subgraph.to(device)

    # Use existing helper to map global ids to neighborhood positions (on CPU)
    # Note: map_global_to_neighborhood expects (data_n_id, node_indices)
    mapped = map_global_to_neighborhood(subgraph.n_id.detach().cpu(), node_indices_cpu)
    # Ensure placeholder negative indices remain -1
    mapped = torch.where(node_indices_cpu >= 0, mapped, torch.full_like(mapped, -1))
    mapped = mapped.to(torch.long)

    return subgraph, mapped


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
    inputs = NodeSamplerInput(input_id=None, node=seeds)
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

    # Optional: carry edge types via original edge IDs
    if hasattr(data_graph, 'edge_type') and hasattr(out, 'edge'):
        sub_data.edge_type = data_graph.edge_type[out.edge]

    # 5) Global->local mapping (global ID -> local [0..N-1])
    global_to_local = torch.full((data_graph.num_nodes,), -1, dtype=torch.long, device=n_id.device)
    global_to_local[n_id] = torch.arange(n_id.numel(), device=n_id.device)

    return sub_data, global_to_local.detach().cpu()


# TODO: logging
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
            conditioning=conditioning
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


def log_predictions_scatter_plot(predictions, true_values, epoch, split_name, use_wandb):
    """
    Create and log a scatter plot of predictions vs true values to wandb.
    
    Args:
        predictions: List of predicted values
        true_values: List of true values
        epoch: Current epoch number
        split_name: 'train' or 'test'
        use_wandb: Whether wandb logging is enabled
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
            logger.warning(f"No valid predictions for {split_name} scatter plot at epoch {epoch}")
            return
        
        # Sample data if too many points for better visualization and performance
        max_points = 1000
        if len(predictions_clean) > max_points:
            indices = np.random.choice(len(predictions_clean), max_points, replace=False)
            predictions_clean = predictions_clean[indices]
            true_values_clean = true_values_clean[indices]
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(true_values_clean, predictions_clean, alpha=0.6, s=20, c='blue')
        
        # Add perfect prediction line (y=x)
        min_val = min(np.min(true_values_clean), np.min(predictions_clean))
        max_val = max(np.max(true_values_clean), np.max(predictions_clean))
        
        # Ensure we have a valid range
        if min_val == max_val:
            min_val = min_val * 0.9 if min_val > 0 else min_val - 1
            max_val = max_val * 1.1 if max_val > 0 else max_val + 1
        
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('True Cardinality')
        ax.set_ylabel('Predicted Cardinality')
        ax.set_title(f'{split_name.capitalize()} Data: Predictions vs True Values (Epoch {epoch})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale if the range is large and positive
        if min_val > 0 and max_val / min_val > 100:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{split_name.capitalize()} Data: Predictions vs True Values (Epoch {epoch} - Log Scale)')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({
            f"{split_name}_predictions_scatter_epoch_{epoch}": wandb.Image(fig),
        })
        
        # Close the figure to free memory
        plt.close(fig)
        
        logger.debug(f"Successfully logged {split_name} scatter plot for epoch {epoch}")
        
    except Exception as e:
        logger.error(f"Error creating scatter plot for {split_name} at epoch {epoch}: {e}")
        # Close any open figures
        plt.close('all')


def train_epoch(state: TrainingState):
    """Train the model for one epoch."""
    # Extract components for easier access
    state.model_components.model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0
    device = state.training_context.device
    epochs = state.training_context.config.epochs

    original_graph = state.data_components.data
    relation_graph = state.data_components.data.relation_graph
    assert original_graph.edge_index.device == torch.device('cpu')
    # assert relation_graph.edge_index.device == torch.device('cpu')

    # Move relation_graph to device once (it's static and reused every batch)
    relation_graph = relation_graph.to(device)

    graph_size = torch.tensor([original_graph.num_edges], device=device)

    epoch_train_predictions_log = []    
    epoch_train_true_values_log = []
    progress_bar = tqdm(
        state.data_components.train_loader, desc=f"Epoch {state.training_context.current_epoch}/{epochs}", leave=False
    )

    # Prepare reusable sampler once per epoch if enabled
    num_neighbors_cfg = getattr(state.training_context.config, 'EntityGNNNeighborhood', False)
    sampler = None
    if not isinstance(state.model_components.model, FiceBinary) and num_neighbors_cfg is not False:
        sampler = create_reusable_sampler(original_graph, num_neighbors_cfg)


    #with torch.profiler.profile(
    #    activities=[torch.profiler.ProfilerActivity.CPU],
    #    record_shapes=True,
    #) as prof:

        for batch_idx, query_batch in enumerate(progress_bar):


            state.model_components.optimizer.zero_grad()

            # Fast path for FiceBinary: skip neighborhood sampling & avoid moving big graphs
            if isinstance(state.model_components.model, FiceBinary):
                graph = None
                num_neighbors = False
            else:
                num_neighbors = num_neighbors_cfg
                if num_neighbors is not False and sampler is not None:
                    # Reusable sampler path
                    node_indices = query_batch.node_embedding_indices
                    seeds_cpu = node_indices[node_indices >= 0].unique().detach().cpu()
                    if seeds_cpu.numel() > 0:
                        graph_sub, global_to_local = sample_with_sampler(sampler, original_graph, seeds_cpu)
                        query_nodes = query_batch.node_embedding_indices.detach().cpu()
                        mapped = torch.where(
                            query_nodes >= 0,
                            global_to_local[query_nodes],
                            torch.full_like(query_nodes, -1),
                        ).to(torch.long)
                        graph = graph_sub.to(device)
                        query_batch.node_embedding_indices = mapped
                    else:
                        graph = None
                elif num_neighbors is not False:
                    raise Exception("NeighborSampler is not supported for training")
                    # Fallback: NeighborLoader per-batch
                    loader_workers = int(getattr(state.training_context.config, 'SamplingNumWorkers', 0) or 0)
                    pin_memory = bool(getattr(state.training_context.config, 'SamplingPinMemory', False))
                    persistent = bool(getattr(state.training_context.config, 'SamplingPersistentWorkers', False))
                    graph, mapped_indices = build_neighborhood_subgraph(
                        data_graph=original_graph,
                        query_batch=query_batch,
                        num_neighbors=num_neighbors,
                        disjoint=False,
                        shuffle=False,
                        loader_num_workers=loader_workers,
                        loader_pin_memory=pin_memory,
                        loader_persistent_workers=persistent,
                    )
                    if graph is not None:
                        graph = graph.to(device)
                        query_batch.node_embedding_indices = mapped_indices
                else:
                    graph = original_graph.to(device)

            relation_graph_inp = None if isinstance(state.model_components.model, FiceBinary) else relation_graph
            log_cardinality_pred = state.model_components.model(data=graph, relation_graph=relation_graph_inp, batch=query_batch.to(device), graph_size=graph_size)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)

            true_cardinality = query_batch.y.to(torch.float32).to(state.training_context.device)
            log_true_cardinality = torch.log(true_cardinality)

            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)

            # Backward pass
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(state.model_components.model.parameters(), max_norm=1.0)  # Gradient clipping
            state.model_components.optimizer.step()

            # Move predictions to CPU for logging to save GPU memory
            epoch_train_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
            epoch_train_true_values_log.extend(tensor_to_numpy(log_true_cardinality))

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
                })

            # Clean up batch tensors to free GPU memory
            del log_cardinality_pred, true_cardinality, log_true_cardinality, loss
            if graph is not None:
                del graph

    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))

    # Clear GPU cache after training epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    epoch_avg_train_loss = epoch_train_loss / num_train_batches

    # Log scatter plot for training predictions vs true values
    log_predictions_scatter_plot(
        predictions=epoch_train_predictions_log,
        true_values=epoch_train_true_values_log,
        epoch=state.training_context.current_epoch,
        split_name="train",
        use_wandb=state.training_context.use_wandb
    )

    return (
        epoch_avg_train_loss,
        epoch_train_predictions_log,
        epoch_train_true_values_log,
    )


def evaluate_model(state: TrainingState):   
    """Evaluate the model on test data."""
    # Extract components for easier access
    device = state.training_context.device
    use_wandb = state.training_context.use_wandb
    original_graph = state.data_components.data
    relation_graph = state.data_components.data.relation_graph
    
    assert original_graph.edge_index.device == torch.device('cpu')
    # assert relation_graph.edge_index.device == torch.device('cpu')

    # Move relation_graph to device once (it's static and reused every batch)
    relation_graph = relation_graph.to(device)
    graph_size = torch.tensor([original_graph.num_edges], device=device)

    state.model_components.model.eval()

    epoch_test_loss = 0.0
    epoch_test_predictions_log = []
    epoch_test_true_values_log = []

    with torch.no_grad():
        # Prepare reusable sampler once for evaluation
        num_neighbors_eval = getattr(state.training_context.config, 'EntityGNNNeighborhood', False)
        sampler = None
        if not isinstance(state.model_components.model, FiceBinary) and num_neighbors_eval is not False:
            sampler = create_reusable_sampler(original_graph, num_neighbors_eval)

        for batch_idx, query_batch in tqdm(enumerate(state.data_components.test_loader), desc="Evaluating", leave=False):
            if isinstance(state.model_components.model, FiceBinary):
                graph = None
                relation_graph_inp = None
            else:
                if num_neighbors_eval is not False and sampler is not None:
                    # Reusable sampler path
                    node_indices = query_batch.node_embedding_indices
                    seeds_cpu = node_indices[node_indices >= 0].unique().detach().cpu()
                    if seeds_cpu.numel() > 0:
                        graph_sub, global_to_local = sample_with_sampler(sampler, original_graph, seeds_cpu)
                        query_nodes = query_batch.node_embedding_indices.detach().cpu()
                        mapped = torch.where(
                            query_nodes >= 0,
                            global_to_local[query_nodes],
                            torch.full_like(query_nodes, -1),
                        ).to(torch.long)
                        graph = graph_sub.to(device)
                        query_batch.node_embedding_indices = mapped
                    else:
                        graph = None
                elif num_neighbors_eval is not False:
                    raise Exception("NeighborSampler is not supported for training")
                    # Fallback: NeighborLoader per-batch
                    loader_workers = int(getattr(state.training_context.config, 'SamplingNumWorkers', 0) or 0)
                    pin_memory = bool(getattr(state.training_context.config, 'SamplingPinMemory', False))
                    persistent = bool(getattr(state.training_context.config, 'SamplingPersistentWorkers', False))
                    graph, mapped_indices = build_neighborhood_subgraph(
                        data_graph=original_graph,
                        query_batch=query_batch,
                        num_neighbors=num_neighbors_eval if isinstance(num_neighbors_eval, list) else [-1],
                        disjoint=False,
                        shuffle=False,
                        loader_num_workers=loader_workers,
                        loader_pin_memory=pin_memory,
                        loader_persistent_workers=persistent,
                    )
                    if graph is not None:
                        graph = graph.to(device)
                        query_batch.node_embedding_indices = mapped_indices
                else:
                    graph = original_graph.to(device)
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

            epoch_test_loss += loss.item()

            # Store predictions for analysis - move to CPU immediately
            # Log batch-level metrics to wandb if enabled
            if use_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid spam
                wandb.log({
                    f"test_loss_epoch_{state.training_context.current_epoch}": loss.item(),
                })

            # Clean up batch tensors to free GPU memory
            del log_cardinality_pred, true_cardinality, log_true_cardinality, loss
            if graph is not None:
                del graph



    # Clear GPU cache after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_test_loss = epoch_test_loss / len(state.data_components.test_loader)

    # Log scatter plot for test predictions vs true values
    log_predictions_scatter_plot(
        predictions=epoch_test_predictions_log,
        true_values=epoch_test_true_values_log,
        epoch=state.training_context.current_epoch,
        split_name="test",
        use_wandb=use_wandb
    )

    return (
        avg_test_loss,
        epoch_test_predictions_log,
        epoch_test_true_values_log,
    )


def train_model(
    state: TrainingState 
):
    """Main training loop with progress tracking and evaluation."""

    # Initialize lists for efficient data accumulation (avoid repeated DataFrame concatenation)
    train_metrics_data = []
    test_metrics_data = []
    train_predictions_data = []
    test_predictions_data = []
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
        
        # Save training predictions to list
        for i, (pred_log, true_log, pred, true) in enumerate(zip(epoch_train_predictions_log, epoch_train_true_values_log, epoch_train_predictions, epoch_train_true_values)):
            train_predictions_data.append({
                'epoch': state.training_context.current_epoch,
                'batch_idx': i,
                'prediction_log': pred_log,
                'true_value_log': true_log,
                'prediction': pred,
                'true_value': true
            })
        
        # Save DataFrames to CSV periodically (every 10 epochs) to avoid excessive I/O
        if (state.training_context.current_epoch) % 10 == 0 or (state.training_context.current_epoch) == epochs:
            train_metrics_df = pd.DataFrame(train_metrics_data)
            train_predictions_df = pd.DataFrame(train_predictions_data)
            state.train_predictions = train_predictions_df
            state.train_metrics = train_metrics_df
            
            train_metrics_csv_path = state.training_context.log_dir / TRAIN_METRICS_FILE_NAME
            train_predictions_csv_path = state.training_context.log_dir / TRAIN_VALUES_FILE_NAME
            train_metrics_df.to_csv(train_metrics_csv_path, index=False)
            train_predictions_df.to_csv(train_predictions_csv_path, index=False)

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

        # Evaluation phase
        if (epoch) % state.training_context.config.eval_every == 0:
            (
                avg_test_loss,
                epoch_test_predictions_log,
                epoch_test_true_values_log,
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
            
            # Save test predictions to list
            for i, (pred_log, true_log, pred, true) in enumerate(zip(epoch_test_predictions_log, epoch_test_true_values_log, epoch_test_predictions, epoch_test_true_values)):
                test_predictions_data.append({
                    'epoch': state.training_context.current_epoch,
                    'batch_idx': i,
                    'prediction_log': pred_log,
                    'true_value_log': true_log,
                    'prediction': pred,
                    'true_value': true
                })
            
            # Save DataFrames to CSV periodically (every 10 epochs) to avoid excessive I/O
            if (state.training_context.current_epoch) % 10 == 0 or (state.training_context.current_epoch) == state.training_context.config.epochs:
                test_metrics_df = pd.DataFrame(test_metrics_data)
                test_predictions_df = pd.DataFrame(test_predictions_data)
                state.test_predictions
                state.test_metrics = test_metrics_df

                test_metrics_csv_path = state.training_context.log_dir / TEST_METRICS_FILE_NAME
                test_predictions_csv_path = state.training_context.log_dir / TEST_VALUES_FILE_NAME
                test_metrics_df.to_csv(test_metrics_csv_path, index=False)
                test_predictions_df.to_csv(test_predictions_csv_path, index=False)

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
    final_train_predictions_df = pd.DataFrame(train_predictions_data)
    final_test_predictions_df = pd.DataFrame(test_predictions_data)
    
    # Save final DataFrames to CSV
    train_metrics_csv_path = state.training_context.log_dir / TRAIN_METRICS_FILE_NAME
    train_predictions_csv_path = state.training_context.log_dir / TRAIN_VALUES_FILE_NAME
    test_metrics_csv_path = state.training_context.log_dir / TEST_METRICS_FILE_NAME
    test_predictions_csv_path = state.training_context.log_dir / TEST_VALUES_FILE_NAME

    final_train_metrics_df.to_csv(train_metrics_csv_path, index=False)
    final_train_predictions_df.to_csv(train_predictions_csv_path, index=False)
    final_test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    final_test_predictions_df.to_csv(test_predictions_csv_path, index=False)

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
        "train_predictions_df": final_train_predictions_df,
        "test_predictions_df": final_test_predictions_df,
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
    logger.info(f"  Best test loss: {training_results['best_test_loss']:.6f}")
    logger.info(f"  Final train loss: {training_results['train_losses'][-1]:.6f}")
    logger.info(f"  Final test loss: {training_results['test_losses'][-1]:.6f}")
    logger.info(f"  Final train MAE: {training_results['train_mae'][-1]:.6f}")
    logger.info(f"  Final test MAE: {training_results['test_mae'][-1]:.6f}")
    logger.info(f"  Final train Q-Error: {training_results['train_q_error'][-1]:.6f}")
    logger.info(f"  Final test Q-Error: {training_results['test_q_error'][-1]:.6f}")

    logger.info(f"\nSaved Artifacts:")
    logger.info(f"  Run directory: {run_dir}")
    logger.info(f"  Best model: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"  Final model: {checkpoint_dir / 'final_model.pt'}")
    logger.info(f"  Training metrics CSV: {log_dir / TRAIN_METRICS_FILE_NAME}")
    logger.info(f"  Test metrics CSV: {log_dir / TEST_METRICS_FILE_NAME}")
    logger.info(f"  Training predictions CSV: {log_dir / TRAIN_VALUES_FILE_NAME}")
    logger.info(f"  Test predictions CSV: {log_dir / TEST_VALUES_FILE_NAME}")
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
    for key, value in config.dict().items():
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
    data = load_dataset(context)
    assert data.edge_index.device == torch.device('cpu')
    assert data.relation_graph.edge_index.device == torch.device('cpu')

    (entity_occurrences, relation_occurrences, entity_mapping, relation_mapping) = (
        load_occurrences(context)
    )
    queries = load_queries(
        context,
        entity_occurrences,
        relation_occurrences,
        entity_mapping,
        relation_mapping,
    )


    assert queries[0].x.device == torch.device('cpu')
    logger.info(f"Queries loaded successfully on device: {queries[0].x.device}!")

    train_queries, test_queries = train_test_split(context, queries)
    logger.info(f"Dataset split:")
    logger.info(f"  Total queries: {len(queries)}")
    logger.info(
        f"  Train queries: {len(train_queries)} ({len(train_queries)/len(queries)*100:.1f}%)"
    )
    logger.info(
        f"  Test queries: {len(test_queries)} ({len(test_queries)/len(queries)*100:.1f}%)"
    )
    logger.info(f"  Train ratio: {config.train_ratio}")

    # Log dataset info to wandb
    if use_wandb:
        wandb.log({
            "dataset_total_queries": len(queries),
            "dataset_train_queries": len(train_queries),
            "dataset_test_queries": len(test_queries),
            "dataset_num_nodes": data.num_nodes,
            "dataset_num_edges": data.num_edges,
        })

    train_loader = DataLoader(
        dataset=train_queries, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_queries,
        batch_size=1,  # Use batch_size=1 for evaluation
        shuffle=False,
    )

    # Create data components
    data_components = DataComponents(
        data=data,
        train_loader=train_loader,
        test_loader=test_loader,
        train_queries=train_queries,
        test_queries=test_queries,

    )

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
        data_components=data_components,
    )

    # Train the model
    training_results = train_model(state)

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
            "final_test_loss": training_results['test_losses'][-1],
            "final_train_mae": training_results['train_mae'][-1],
            "final_test_mae": training_results['test_mae'][-1],
            "final_train_q_error": training_results['train_q_error'][-1],
            "final_test_q_error": training_results['test_q_error'][-1],
            "total_epochs": len(training_results['train_losses'])
        }
        wandb.log(final_metrics)
        
        # Save training data
        artifact = wandb.Artifact("training_data", type="dataset")
        if (state.training_context.log_dir / TRAIN_METRICS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_METRICS_FILE_NAME))
        if (state.training_context.log_dir / TEST_METRICS_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_METRICS_FILE_NAME))
        if (state.training_context.log_dir / TRAIN_VALUES_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TRAIN_VALUES_FILE_NAME))
        if (state.training_context.log_dir / TEST_VALUES_FILE_NAME).exists():
            artifact.add_file(str(state.training_context.log_dir / TEST_VALUES_FILE_NAME))
        wandb.log_artifact(artifact)
        
        
        wandb.finish()
        logger.info("Weights & Biases run finished")


if __name__ == "__main__":
    main()
