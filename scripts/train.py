import torch
import json
import random
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import subprocess
import wandb
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Import project modules
from utils.helper import parse_arguments, merge_config_with_args, setup_logging, set_random_seeds, get_device, initialize_wandb, process_seed
from utils.config import load_config
import utils.metrics as metrics
from fice import datasets
from fice.models import FICE
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


TRAIN_METRICS_FILE_NAME = "train_training_metrics.csv"
TEST_METRICS_FILE_NAME = "test_training_metrics.csv"
TRAIN_VALUES_FILE_NAME = "train_values.csv"
TEST_VALUES_FILE_NAME = "test_values.csv"


@dataclass
class TrainingContext:
    """Container for training configuration and environment."""
    config: dict
    device: torch.device
    run_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    use_wandb: bool = False
    current_epoch: int = 0


@dataclass
class ModelComponents:
    """Container for model, optimizer, and scheduler."""
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None


@dataclass
class DataComponents:
    """Container for data and data loaders."""
    data: torch.Tensor
    train_loader: DataLoader
    test_loader: DataLoader
    train_queries: list
    test_queries: list

# TODO: logging
def setup_directories(config):
    """Setup project directories, timestamps, and logging."""

    run_dir = setup_run_dir(config)

    log_dir = run_dir / "logs/"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints/"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, log_dir, checkpoint_dir


def load_dataset(context: TrainingContext):
    """Load the graph dataset and verify its properties."""
    config = context.config
    device = context.device
    
    data_root = Path(config["data_root"])
    queries_root = Path(config["queries_root"])

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
    graph = graph[0].to(device)

    logger.info(f"Graph loaded successfully!")
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
    device = context.device
    
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
        device=device,
    ) #TODO: will change hopefully

    logger.info(f"Queries loaded. Total queries: {len(queries)}")
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


def initialize_model_and_optimizer(context: TrainingContext, steps_per_epoch: int) -> ModelComponents:
    """Initialize the FICE model and optimizer."""
    config = context.config
    device = context.device
    
    logger.info("Initializing model...")
    model = FICE()
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model initialized successfully!")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"{model}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model device: {next(model.parameters()).device}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Initialize scheduler based on configuration
    scheduler = None
    if config.get("use_scheduler", False):
        max_lr = config.get("scheduler_max_lr", 0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,  
            steps_per_epoch=steps_per_epoch, 
            epochs=config["epochs"]
        )
        logger.info(f"Scheduler: OneCycleLR with max_lr={max_lr}")
    else:
        logger.info("Scheduler: Disabled")

    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Weight decay: {config['weight_decay']}")

    return ModelComponents(model=model, optimizer=optimizer, scheduler=scheduler)


def tensor_to_numpy(tensor):
    """Helper function to convert tensors to numpy."""
    return tensor.detach().cpu().numpy().tolist()


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


def train_epoch(context: TrainingContext, model_components: ModelComponents, data_components: DataComponents):
    """Train the model for one epoch."""
    # Extract components for easier access
    config = context.config
    device = context.device
    use_wandb = context.use_wandb
    epoch_num = context.current_epoch
    
    model = model_components.model
    optimizer = model_components.optimizer
    scheduler = model_components.scheduler
    
    data = data_components.data
    train_loader = data_components.train_loader
    model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0

    epoch_train_predictions = []
    epoch_train_true_values = []
    epoch_train_predictions_log = []
    epoch_train_true_values_log = []
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch_num+1}/{config['epochs']}", leave=False
    )

    for batch_idx, query_batch in enumerate(progress_bar):
        optimizer.zero_grad()

        # Forward pass
        query_batch = query_batch.to(device)
        log_cardinality_pred = model(data=data, batch=query_batch)
        log_cardinality_pred = log_cardinality_pred.to(torch.float32).to(device)

        true_cardinality = query_batch.y.to(torch.float32).to(device)
        log_true_cardinality = torch.log(true_cardinality).to(device)

        epoch_train_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
        epoch_train_true_values_log.extend(tensor_to_numpy(log_true_cardinality))

        # Compute loss
        loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)


        # Backward pass
        loss.backward()
        optimizer.step()
        # Step scheduler once per epoch (for OneCycleLR)
        if scheduler:
            scheduler.step()

        epoch_train_loss += loss.item()
        num_train_batches += 1

        # Store predictions for analysis
        prediction = torch.exp(log_cardinality_pred)
        

        epoch_train_predictions.extend(tensor_to_numpy(prediction))
        epoch_train_true_values.extend(tensor_to_numpy(true_cardinality))

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_train_loss/num_train_batches:.4f}",
            }
        )
        
        # Log batch-level metrics to wandb if enabled
        if use_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid spam
            wandb.log({
                f"train_loss_epoch_{context.current_epoch + 1}": loss.item(),
            })

    epoch_avg_train_loss = epoch_train_loss / num_train_batches

    # Log scatter plot for training predictions vs true values
    log_predictions_scatter_plot(
        predictions=epoch_train_predictions_log,
        true_values=epoch_train_true_values_log,
        epoch=epoch_num + 1,
        split_name="train",
        use_wandb=use_wandb
    )

    return (
        epoch_avg_train_loss,
        epoch_train_predictions,
        epoch_train_true_values,
        epoch_train_predictions_log,
        epoch_train_true_values_log,
    )


def evaluate_model(context: TrainingContext, model_components: ModelComponents, data_components: DataComponents):
    """Evaluate the model on test data."""
    # Extract components for easier access
    device = context.device
    use_wandb = context.use_wandb
    epoch = context.current_epoch
    
    model = model_components.model
    data = data_components.data
    test_loader = data_components.test_loader
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_predictions = []
    epoch_test_true_values = []

    epoch_test_predictions_log = []
    epoch_test_true_values_log = []

    with torch.no_grad():
        for batch_idx, query_batch in tqdm(enumerate(test_loader), desc="Evaluating", leave=False):
            # Forward pass
            query_batch = query_batch.to(device)

            log_cardinality_pred = model(data=data, batch=query_batch)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32).to(device)
            log_true_cardinality = torch.log(query_batch.y.to(torch.float32).to(device))
            true_cardinality = query_batch.y.to(torch.float32).to(device)
            epoch_test_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
            epoch_test_true_values_log.extend(tensor_to_numpy(log_true_cardinality))

            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)
            epoch_test_loss += loss.item()

            # Store predictions for analysis
            prediction = torch.exp(log_cardinality_pred)

            epoch_test_predictions.extend(tensor_to_numpy(prediction))
            epoch_test_true_values.extend(tensor_to_numpy(true_cardinality))
            # Log batch-level metrics to wandb if enabled
            if use_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid spam
                wandb.log({
                    f"test_loss_epoch_{context.current_epoch + 1}": loss.item(),
                })

    avg_test_loss = epoch_test_loss / len(test_loader)

    # Log scatter plot for test predictions vs true values
    log_predictions_scatter_plot(
        predictions=epoch_test_predictions_log,
        true_values=epoch_test_true_values_log,
        epoch=context.current_epoch + 1,
        split_name="test",
        use_wandb=use_wandb
    )

    return (
        avg_test_loss,
        epoch_test_predictions,
        epoch_test_true_values,
        epoch_test_predictions_log,
        epoch_test_true_values_log,
    )


def train_model(
    context: TrainingContext,
    model_components: ModelComponents,
    data_components: DataComponents
):
    """Main training loop with progress tracking and evaluation."""
    # Extract components for easier access
    config = context.config
    device = context.device
    log_dir = context.log_dir
    checkpoint_dir = context.checkpoint_dir
    use_wandb = context.use_wandb
    
    model = model_components.model
    optimizer = model_components.optimizer
    scheduler = model_components.scheduler
    
    data = data_components.data
    train_loader = data_components.train_loader
    test_loader = data_components.test_loader
    
    best_test_loss = float("inf")

    # Initialize lists for efficient data accumulation (avoid repeated DataFrame concatenation)
    train_metrics_data = []
    test_metrics_data = []
    train_predictions_data = []
    test_predictions_data = []

    logger.info("Data accumulation lists initialized for efficient metric tracking!")
    logger.info(f"Starting training for {config['epochs']} epochs...")

    

    # Training loop
    for epoch in range(config["epochs"]):
        context.current_epoch = epoch
        
        # Training phase
        (
            epoch_avg_train_loss,
            epoch_train_predictions,
            epoch_train_true_values,
            epoch_train_predictions_log,
            epoch_train_true_values_log,
        ) = train_epoch(context, model_components, data_components)

        # Calculate training metrics
        mse = metrics.calculate_mse(epoch_train_predictions, epoch_train_true_values)
        mae = metrics.calculate_mae(epoch_train_predictions, epoch_train_true_values)
        rmse = np.sqrt(mse)
        q_error = metrics.calculate_q_error(
            epoch_train_predictions, epoch_train_true_values
        )

        # Append training metrics to list (much more efficient than DataFrame concatenation)
        train_metrics_data.append({
            'epoch': context.current_epoch + 1,
            'loss': epoch_avg_train_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'q_error': q_error
        })
        
        # Save training predictions to list
        for i, (pred_log, true_log, pred, true) in enumerate(zip(epoch_train_predictions_log, epoch_train_true_values_log, epoch_train_predictions, epoch_train_true_values)):
            train_predictions_data.append({
                'epoch': context.current_epoch + 1,
                'batch_idx': i,
                'prediction_log': pred_log,
                'true_value_log': true_log,
                'prediction': pred,
                'true_value': true
            })
        
        # Save DataFrames to CSV periodically (every 10 epochs) to avoid excessive I/O
        if (context.current_epoch + 1) % 10 == 0 or (context.current_epoch + 1) == config['epochs']:
            train_metrics_df = pd.DataFrame(train_metrics_data)
            train_predictions_df = pd.DataFrame(train_predictions_data)
            
            train_metrics_csv_path = log_dir / TRAIN_METRICS_FILE_NAME
            train_predictions_csv_path = log_dir / TRAIN_VALUES_FILE_NAME
            train_metrics_df.to_csv(train_metrics_csv_path, index=False)
            train_predictions_df.to_csv(train_predictions_csv_path, index=False)

        # Log training metrics to wandb
        if use_wandb:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            wandb.log({
                "train_loss": epoch_avg_train_loss,
                "train_mse": mse,
                "train_mae": mae,
                "train_rmse": rmse,
                "train_q_error": q_error,
                "learning_rate": current_lr,
                "epoch": context.current_epoch + 1
            })

        # Evaluation phase
        if (epoch + 1) % config["eval_every"] == 0:
            (
                avg_test_loss,
                epoch_test_predictions,
                epoch_test_true_values,
                epoch_test_predictions_log,
                epoch_test_true_values_log,
            ) = evaluate_model(context, model_components, data_components)

            # Calculate test metrics
            mse = metrics.calculate_mse(epoch_test_predictions, epoch_test_true_values)
            mae = metrics.calculate_mae(epoch_test_predictions, epoch_test_true_values)
            rmse = np.sqrt(mse)
            q_error = metrics.calculate_q_error(
                epoch_test_predictions, epoch_test_true_values
            )

            # Append test metrics to list (much more efficient than DataFrame concatenation)
            test_metrics_data.append({
                'epoch': context.current_epoch + 1,
                'loss': avg_test_loss,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'q_error': q_error
            })
            
            # Save test predictions to list
            for i, (pred_log, true_log, pred, true) in enumerate(zip(epoch_test_predictions_log, epoch_test_true_values_log, epoch_test_predictions, epoch_test_true_values)):
                test_predictions_data.append({
                    'epoch': context.current_epoch + 1,
                    'batch_idx': i,
                    'prediction_log': pred_log,
                    'true_value_log': true_log,
                    'prediction': pred,
                    'true_value': true
                })
            
            # Save DataFrames to CSV periodically (every 10 epochs) to avoid excessive I/O
            if (context.current_epoch + 1) % 10 == 0 or (context.current_epoch + 1) == config['epochs']:
                test_metrics_df = pd.DataFrame(test_metrics_data)
                test_predictions_df = pd.DataFrame(test_predictions_data)
                
                test_metrics_csv_path = log_dir / TEST_METRICS_FILE_NAME
                test_predictions_csv_path = log_dir / TEST_VALUES_FILE_NAME
                test_metrics_df.to_csv(test_metrics_csv_path, index=False)
                test_predictions_df.to_csv(test_predictions_csv_path, index=False)

            # Log test metrics to wandb
            if use_wandb:
                wandb.log({
                    "test_loss": avg_test_loss,
                    "test_mse": mse,
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "test_q_error": q_error,
                    "epoch": context.current_epoch + 1
                })

            # Log metrics
            logger.info(
                f"Epoch {context.current_epoch+1}/{config['epochs']} - "
                f"Train Loss: {epoch_avg_train_loss:.6f}, "
                f"Test Loss: {avg_test_loss:.6f}"
            )

            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_path = checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": context.current_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "train_loss": epoch_avg_train_loss,
                        "test_loss": avg_test_loss,
                        "config": config,
                    },
                    best_model_path,
                )
                logger.info(
                    f"New best model saved with test loss: {best_test_loss:.6f}"
                )
                
                # Log best model as wandb artifact
                if use_wandb:
                    artifact = wandb.Artifact(
                        name="best_model",
                        type="model",
                        description=f"Best model checkpoint at epoch {context.current_epoch+1} with test loss {best_test_loss:.6f}"
                    )
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact)
                    
                    # Also log the best test loss metric
                    wandb.log({
                        "best_test_loss": best_test_loss,
                        "best_model_epoch": context.current_epoch + 1
                    })
        else:
            logger.info(
                f"Epoch {context.current_epoch+1}/{config['epochs']} - Train Loss: {epoch_avg_train_loss:.6f}"
            )

        # Save checkpoint periodically
        if (epoch + 1) % config["save_every"] == 0:
            # Get latest test loss from test data list
            latest_test_loss = test_metrics_data[-1]['loss'] if len(test_metrics_data) > 0 else None
            
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{context.current_epoch+1}.pt"
            torch.save(
                {
                    "epoch": context.current_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "train_loss": epoch_avg_train_loss,
                    "test_loss": latest_test_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training completed!")

    # Create final DataFrames from accumulated data
    final_train_metrics_df = pd.DataFrame(train_metrics_data)
    final_test_metrics_df = pd.DataFrame(test_metrics_data)
    final_train_predictions_df = pd.DataFrame(train_predictions_data)
    final_test_predictions_df = pd.DataFrame(test_predictions_data)
    
    # Save final DataFrames to CSV
    train_metrics_csv_path = log_dir / TRAIN_METRICS_FILE_NAME
    train_predictions_csv_path = log_dir / TRAIN_VALUES_FILE_NAME
    test_metrics_csv_path = log_dir / TEST_METRICS_FILE_NAME
    test_predictions_csv_path = log_dir / TEST_VALUES_FILE_NAME
    
    final_train_metrics_df.to_csv(train_metrics_csv_path, index=False)
    final_train_predictions_df.to_csv(train_predictions_csv_path, index=False)
    final_test_metrics_df.to_csv(test_metrics_csv_path, index=False)
    final_test_predictions_df.to_csv(test_predictions_csv_path, index=False)

    return {
        "best_test_loss": best_test_loss,
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
    run_dir = Path(config.get("output_dir")) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    """Main function to orchestrate the training process."""
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)

    # Merge command-line arguments with config (overwrite config values with cli values if passed)
    config = merge_config_with_args(config, args)
    run_dir, log_dir, checkpoint_dir = setup_directories(config)

    # Setup logging
    setup_logging(log_dir)
    logger.info("Starting training script...")
    logger.info("All imports successful!")
    
    # Log original configuration
    logger.info("Original Configuration from file:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
   
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")

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
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Set random seeds for reproducibility
    processed_seed = process_seed(config["seed"])
    set_random_seeds(processed_seed)
    logger.info(f"Random seed set to: {processed_seed}")

    # Setup device
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb_run = initialize_wandb(config, run_dir)
    use_wandb = wandb_run is not None and args.wandb

    # Create training context
    context = TrainingContext(
        config=config,
        device=device,
        run_dir=run_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb
    )

    # Load data
    data = load_dataset(context)
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

    train_queries, test_queries = train_test_split(context, queries)
    logger.info(f"Dataset split:")
    logger.info(f"  Total queries: {len(queries)}")
    logger.info(
        f"  Train queries: {len(train_queries)} ({len(train_queries)/len(queries)*100:.1f}%)"
    )
    logger.info(
        f"  Test queries: {len(test_queries)} ({len(test_queries)/len(queries)*100:.1f}%)"
    )
    logger.info(f"  Train ratio: {config['train_ratio']}")

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
        dataset=train_queries, batch_size=config["batch_size"], shuffle=True
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
        test_queries=test_queries
    )

    # Initialize model and optimizer
    model_components = initialize_model_and_optimizer(context, steps_per_epoch=len(train_loader))

    # Log model information to wandb
    if use_wandb:
        total_params = sum(p.numel() for p in model_components.model.parameters())
        trainable_params = sum(p.numel() for p in model_components.model.parameters() if p.requires_grad)
        wandb.log({
            "model_total_parameters": total_params,
            "model_trainable_parameters": trainable_params,
            "model_name": model_components.model.__class__.__name__
        })
        
        # Watch the model for gradient and parameter tracking
        wandb.watch(model_components.model, log="all", log_freq=100)

    # Train the model
    training_results = train_model(context, model_components, data_components)

    # Print final summary
    print_final_summary(
        training_results, run_dir, checkpoint_dir, log_dir
    )

    # Finalize wandb run
    if use_wandb:
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
        if (log_dir / TRAIN_METRICS_FILE_NAME).exists():
            artifact.add_file(str(log_dir / TRAIN_METRICS_FILE_NAME))
        if (log_dir / TEST_METRICS_FILE_NAME).exists():
            artifact.add_file(str(log_dir / TEST_METRICS_FILE_NAME))
        if (log_dir / TRAIN_VALUES_FILE_NAME).exists():
            artifact.add_file(str(log_dir / TRAIN_VALUES_FILE_NAME))
        if (log_dir / TEST_VALUES_FILE_NAME).exists():
            artifact.add_file(str(log_dir / TEST_VALUES_FILE_NAME))
        wandb.log_artifact(artifact)
        
        # Save final model state as artifact
        final_model_path = checkpoint_dir / "final_model.pt"
        torch.save({
            "model_state_dict": model_components.model.state_dict(),
            "optimizer_state_dict": model_components.optimizer.state_dict(),
            "scheduler_state_dict": model_components.scheduler.state_dict() if model_components.scheduler else None,
            "training_results": training_results,
            "config": config,
            "final_epoch": len(training_results['train_losses'])
        }, final_model_path)
        
        final_model_artifact = wandb.Artifact(
            name="final_model",
            type="model",
            description=f"Final model state after {len(training_results['train_losses'])} epochs of training"
        )
        final_model_artifact.add_file(str(final_model_path))
        wandb.log_artifact(final_model_artifact)
        
        wandb.finish()
        logger.info("Weights & Biases run finished")


if __name__ == "__main__":
    main()
