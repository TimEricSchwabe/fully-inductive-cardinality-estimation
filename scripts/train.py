import torch
import json
import random
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import wandb

# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.append("/Users/lukas/Workspace/Uni/BachelorsThesis/FICE")

# Import project modules
from utils.helper import parse_arguments, merge_config_with_args, setup_logging, set_random_seeds, get_device
from utils.config import load_config
import utils.metrics as metrics
import utils.plotting as plotting
from fice import datasets
from fice.models import FICE
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# Weights & Biases configuration
WANDB_CONFIG = {
    'enabled': True,
    'project': 'fice',
    'entity': "fice",  # Set to your wandb username/team or leave None for default
    'run_name': None,  # Will be auto-generated if None
    'tags': [],
    'notes': ''
}


def initialize_wandb(config, run_dir):
    """Initialize Weights & Biases logging if enabled."""
    # Check if wandb is enabled: command-line flag takes precedence over WANDB_CONFIG
    wandb_enabled = config.get('wandb', WANDB_CONFIG['enabled'])
    if not wandb_enabled:
        logger.info("Weights & Biases logging is disabled")
        return None

    # Prepare training config for wandb
    training_config = {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "train_ratio": config["train_ratio"],
        "seed": config["seed"],
        "device": config["device"],
        "data_root": config["data_root"],
        "queries_root": config["queries_root"],
        "eval_every": config["eval_every"],
        "save_every": config["save_every"],
    }
    
    # Add debug config if present
    if config.get("DEBUG_num_queries", -1) > 0:
        training_config["DEBUG_num_queries"] = config["DEBUG_num_queries"]

    # Initialize wandb using WANDB_CONFIG defaults
    run = wandb.init(
        project=WANDB_CONFIG['project'],
        entity=WANDB_CONFIG['entity'],
        name=WANDB_CONFIG['run_name'],
        config=training_config,
        tags=WANDB_CONFIG['tags'],
        notes=WANDB_CONFIG['notes'],
        dir=str(run_dir)
    )
    
    logger.info(f"Weights & Biases initialized: {run.url}")
    return run


# TODO: logging
def setup_directories(config):
    """Setup project directories, timestamps, and logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(config.get("output_dir"))
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "logs/"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints/"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, log_dir, checkpoint_dir


def load_dataset(config, device):
    """Load the graph dataset and verify its properties."""
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


def load_occurrences(config):
    """Load entity and relation mappings and occurrence data."""
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
    from utils.helper import parse_arguments, merge_config_with_args
    logger.info(f"Entities with occurrences: {len(entity_occurrences)}")
    logger.info(f"Relations with occurrences: {len(relation_occurrences)}")
    logger.info(f"Entity mapping size: {len(entity_mapping)}")
    logger.info(f"Relation mapping size: {len(relation_mapping)}")

    return entity_occurrences, relation_occurrences, entity_mapping, relation_mapping
    logger.info("All imports successful!")

def load_queries(
    config,
    entity_occurrences,
    relation_occurrences,
    entity_mapping,
    relation_mapping,
    device,
):
    """Load queries and create train/test split."""
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

def train_test_split(config, queries):
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


def initialize_model_and_optimizer(config, device):
    """Initialize the FICE model and optimizer."""
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

    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Weight decay: {config['weight_decay']}")

    return model, optimizer


def tensor_to_numpy(tensor):
    """Helper function to convert tensors to numpy."""
    return tensor.detach().cpu().numpy().tolist()


def train_epoch(model, data, train_loader, optimizer, device, epoch_num, config, use_wandb=False):
    """Train the model for one epoch."""
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

        log_true_cardinality = torch.log(query_batch.y.to(torch.float32).to(device))

        epoch_train_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
        epoch_train_true_values_log.extend(tensor_to_numpy(log_true_cardinality))

        # Compute loss
        loss = F.mse_loss(log_cardinality_pred, log_true_cardinality).to(device)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        num_train_batches += 1

        # Store predictions for analysis
        prediction = torch.exp(log_cardinality_pred)
        true_value = query_batch.y.to(torch.float32)

        epoch_train_predictions.extend(tensor_to_numpy(prediction))
        epoch_train_true_values.extend(tensor_to_numpy(true_value))

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
                f"train_loss_epoch_{epoch_num + 1}": loss.item(),
            })

    epoch_avg_train_loss = epoch_train_loss / num_train_batches

    return (
        epoch_avg_train_loss,
        epoch_train_predictions,
        epoch_train_true_values,
        epoch_train_predictions_log,
        epoch_train_true_values_log,
    )


def evaluate_model(model, data, test_loader, device):
    """Evaluate the model on test data."""
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_predictions = []
    epoch_test_true_values = []

    epoch_test_predictions_log = []
    epoch_test_true_values_log = []

    with torch.no_grad():
        for query_batch in tqdm(test_loader, desc="Evaluating", leave=False):
            # Forward pass
            query_batch = query_batch.to(device)

            log_cardinality_pred = model(data=data, batch=query_batch)
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            log_true_cardinality = torch.log(query_batch.y.to(torch.float32))

            epoch_test_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
            epoch_test_true_values_log.extend(tensor_to_numpy(log_true_cardinality))

            # Compute loss
            loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)
            epoch_test_loss += loss.item()

            # Store predictions for analysis
            prediction = torch.exp(log_cardinality_pred)
            true_value = query_batch.y.to(torch.float32)

            epoch_test_predictions.extend(tensor_to_numpy(prediction))
            epoch_test_true_values.extend(tensor_to_numpy(true_value))

    avg_test_loss = epoch_test_loss / len(test_loader)

    return (
        avg_test_loss,
        epoch_test_predictions,
        epoch_test_true_values,
        epoch_test_predictions_log,
        epoch_test_true_values_log,
    )


def train_model(
    config, model, optimizer, data, train_loader, test_loader, checkpoint_dir, device, use_wandb=False
):
    """Main training loop with progress tracking and evaluation."""
    best_test_loss = float("inf")

    # Training metrics tracking
    train_losses = []
    train_mse = []
    train_mae = []
    train_rmse = []
    train_q_error = []
    train_predictions_log = []
    train_true_values_log = []

    test_losses = []
    test_mse = []
    test_mae = []
    test_rmse = []
    test_q_error = []
    test_predictions_log = []
    test_true_values_log = []

    logger.info("Training tracking variables initialized!")
    logger.info(f"Starting training for {config['epochs']} epochs...")

    # Training loop
    for epoch in range(config["epochs"]):
        # Training phase
        (
            epoch_avg_train_loss,
            epoch_train_predictions,
            epoch_train_true_values,
            epoch_train_predictions_log,
            epoch_train_true_values_log,
        ) = train_epoch(model, data, train_loader, optimizer, device, epoch, config, use_wandb)

        # Calculate training metrics
        mse = metrics.calculate_mse(epoch_train_predictions, epoch_train_true_values)
        mae = metrics.calculate_mae(epoch_train_predictions, epoch_train_true_values)
        rmse = np.sqrt(mse)
        q_error = metrics.calculate_q_error(
            epoch_train_predictions, epoch_train_true_values
        )

        train_losses.append(epoch_avg_train_loss)
        train_mse.append(mse)
        train_mae.append(mae)
        train_rmse.append(rmse)
        train_q_error.append(q_error)
        train_predictions_log.append(epoch_train_predictions_log)
        train_true_values_log.append(epoch_train_true_values_log)

        # Log training metrics to wandb
        if use_wandb:
            wandb.log({
                "train_loss": epoch_avg_train_loss,
                "train_mse": mse,
                "train_mae": mae,
                "train_rmse": rmse,
                "train_q_error": q_error,
                "epoch": epoch + 1
            })

        # Evaluation phase
        if (epoch + 1) % config["eval_every"] == 0:
            (
                avg_test_loss,
                epoch_test_predictions,
                epoch_test_true_values,
                epoch_test_predictions_log,
                epoch_test_true_values_log,
            ) = evaluate_model(model, data, test_loader, device)

            # Calculate test metrics
            mse = metrics.calculate_mse(epoch_test_predictions, epoch_test_true_values)
            mae = metrics.calculate_mae(epoch_test_predictions, epoch_test_true_values)
            rmse = np.sqrt(mse)
            q_error = metrics.calculate_q_error(
                epoch_test_predictions, epoch_test_true_values
            )

            test_losses.append(avg_test_loss)
            test_mse.append(mse)
            test_mae.append(mae)
            test_rmse.append(rmse)
            test_q_error.append(q_error)
            test_predictions_log.append(epoch_test_predictions_log)
            test_true_values_log.append(epoch_test_true_values_log)

            # Log test metrics to wandb
            if use_wandb:
                wandb.log({
                    "test_loss": avg_test_loss,
                    "test_mse": mse,
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "test_q_error": q_error,
                    "epoch": epoch + 1
                })

            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {epoch_avg_train_loss:.6f}, "
                f"Test Loss: {avg_test_loss:.6f}"
            )

            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_path = checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
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
                        description=f"Best model checkpoint at epoch {epoch+1} with test loss {best_test_loss:.6f}"
                    )
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact)
                    
                    # Also log the best test loss metric
                    wandb.log({
                        "best_test_loss": best_test_loss,
                        "best_model_epoch": epoch + 1
                    })
        else:
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {epoch_avg_train_loss:.6f}"
            )

        # Save checkpoint periodically
        if (epoch + 1) % config["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_avg_train_loss,
                    "test_loss": test_losses[-1] if test_losses else None,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("Training completed!")

    return {
        "best_test_loss": best_test_loss,
        "train_losses": train_losses,
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_q_error": train_q_error,
        "train_predictions_log": train_predictions_log,
        "train_true_values_log": train_true_values_log,
        "test_losses": test_losses,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_q_error": test_q_error,
        "test_predictions_log": test_predictions_log,
        "test_true_values_log": test_true_values_log,
    }


def post_training_gradient_analysis(model):
    """Analyze gradient flow and model performance."""
    logger.info("\nGradient Flow Analysis:")
    has_grad_entity = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.entity_model.parameters()
    )
    logger.info(f"Entity model has gradients: {has_grad_entity}")

    has_grad_relation = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.relation_model.parameters()
    )
    logger.info(f"Relation model has gradients: {has_grad_relation}")



def save_training_metrics(training_results, log_dir):
    """Save training data and metrics to CSV files."""
    # Save prediction logs
    pd.DataFrame(training_results["train_predictions_log"]).to_csv(
        log_dir / "train_prediction_log.csv", index=False
    )
    pd.DataFrame(training_results["train_true_values_log"]).to_csv(
        log_dir / "train_true_values_log.csv", index=False
    )
    pd.DataFrame(training_results["test_predictions_log"]).to_csv(
        log_dir / "test_prediction_log.csv", index=False
    )
    pd.DataFrame(training_results["test_true_values_log"]).to_csv(
        log_dir / "test_true_values_log.csv", index=False
    )

    # Prepare data for saving
    epochs_range = list(range(1, len(training_results["train_losses"]) + 1))
    

    # Create DataFrame with training metrics
    df = pd.DataFrame(
        {
            "epoch": epochs_range,
            "train_loss": training_results["train_losses"],
            "train_mse": training_results["train_mse"],
            "train_mae": training_results["train_mae"],
            "train_rmse": training_results["train_rmse"],
            "train_q_error": training_results["train_q_error"],
        }
    )

    # Add test metrics (aligned with evaluation epochs)
    test_df = pd.DataFrame(
        {
            "epoch": epochs_range,
            "test_loss": training_results["test_losses"],
            "test_mse": training_results["test_mse"],
            "test_mae": training_results["test_mae"],
            "test_rmse": training_results["test_rmse"],
            "test_q_error": training_results["test_q_error"],
        }
    )

    # Merge dataframes
    full_df = df.merge(test_df, on="epoch", how="left")

    # Save to CSV
    csv_path = log_dir / "training_curves.csv"
    full_df.to_csv(csv_path, index=False)
    logger.info(f"Training metrics saved to: {csv_path}")

    # Display first few rows
    logger.info("\nTraining metrics summary:")
    logger.info(full_df.head())

    return full_df


def create_visualizations(training_results, full_df, log_dir):
    """Create training metrics and predictions visualizations."""
    # Training metrics visualization
    metrics_plot_path = log_dir / "training_metrics.png"
    fig = plotting.plot_training_metrics(
        df=full_df,
        figsize=(18, 10),
        use_log_scale=False,
        save_path=metrics_plot_path,
        show_plot=True,
    )
    logger.info(f"Training metrics plot saved to: {metrics_plot_path}")
    logger.info("Training curves and metrics plotted!")

    # Predictions vs True Values visualization
    predictions_plot_path = log_dir / "predictions_vs_true_values_all_epochs.png"
    fig = plotting.plot_predictions_vs_true_values(
        train_predictions_log=training_results["train_predictions_log"],
        train_true_values_log=training_results["train_true_values_log"],
        test_predictions_log=training_results["test_predictions_log"],
        test_true_values_log=training_results["test_true_values_log"],
        figsize=(16, 8),
        max_points_per_epoch=50,
        save_path=predictions_plot_path,
        show_plot=True,
    )
    logger.info(f"Predictions vs True Values plot saved to: {predictions_plot_path}")
    logger.info("Predictions vs True Values across all epochs plotted!")


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
    logger.info(f"  Training metrics CSV: {log_dir / 'training_curves.csv'}")
    logger.info(f"  Training metrics plot: {log_dir / 'training_metrics.png'}")
    logger.info(
        f"  Predictions plot: {log_dir / 'predictions_vs_true_values_all_epochs.png'}"
    )
    logger.info(f"  Log files: {log_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Training script execution completed!")
    logger.info("=" * 60)


def main():
    """Main function to orchestrate the training process."""
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)

     # Merge command-line arguments with config
    config = merge_config_with_args(config, args)
    

    # Setup logging
    run_dir, log_dir, checkpoint_dir = setup_directories(config)

    logger = setup_logging(log_dir)
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
    set_random_seeds(config["seed"])
    logger.info(f"Random seed: {config['seed']}")

    # Setup device
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb_run = initialize_wandb(config, run_dir)
    use_wandb = wandb_run is not None

    # Load data
    data = load_dataset(config, device=device)
    (entity_occurrences, relation_occurrences, entity_mapping, relation_mapping) = (
        load_occurrences(config)
    )
    queries = load_queries(
        config,
        entity_occurrences,
        relation_occurrences,
        entity_mapping,
        relation_mapping,
        device,
    )

    train_queries, test_queries = train_test_split(config, queries)
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



    # Initialize model and optimizer
    model, optimizer = initialize_model_and_optimizer(config, device)

    # Log model information to wandb
    if use_wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model_total_parameters": total_params,
            "model_trainable_parameters": trainable_params,
            "model_name": model.__class__.__name__
        })

    # Train the model
    training_results = train_model(
        config,
        model,
        optimizer,
        data,
        train_loader,
        test_loader,
        checkpoint_dir,
        device,
        use_wandb
    )

    # Post-training analysis
    post_training_gradient_analysis(model)

    # Save metrics and create visualizations
    full_df = save_training_metrics(training_results, log_dir)
    create_visualizations(training_results, full_df, log_dir)

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
        
        # Save training plots as artifacts
        if (log_dir / "training_metrics.png").exists():
            artifact = wandb.Artifact("training_plots", type="plots")
            artifact.add_file(str(log_dir / "training_metrics.png"))
            artifact.add_file(str(log_dir / "predictions_vs_true_values_all_epochs.png"))
            wandb.log_artifact(artifact)
        
        # Save training data
        if (log_dir / "training_curves.csv").exists():
            artifact = wandb.Artifact("training_data", type="dataset")
            artifact.add_file(str(log_dir / "training_curves.csv"))
            wandb.log_artifact(artifact)
        
        wandb.finish()
        logger.info("Weights & Biases run finished")


if __name__ == "__main__":
    main()
