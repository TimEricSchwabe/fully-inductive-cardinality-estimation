#!/usr/bin/env python3
"""
FICE Training Script

A comprehensive training script for the Fully Inductive Cardinality Estimation (FICE) model.
"""

import torch
import json
import random
import numpy as np
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fice import datasets
from fice.models import FICE
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# Configuration
CONFIG = {
    # Data settings
    "data_root": "data/swdf/",
    "queries_root": "data/swdf_queries",
    
    # Training settings
    "epochs": 200,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "train_ratio": 0.8,
    
    # Logging and saving settings
    "log_dir": "logs/",
    "checkpoint_dir": "checkpoints/",
    "save_every": 5,
    "log_level": "INFO",
    
    # Other settings
    "seed": 42,
    "device": "auto",  # "auto", "cpu", or "cuda"
    "eval_every": 1,
}


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('fice_training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"training_{timestamp}.log")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)


if __name__ == "__main__":
    # Use config dictionary instead of argument parsing
    config = CONFIG.copy()  # Make a copy so we can modify if needed
    
    # Set random seeds for reproducibility
    set_random_seeds(config["seed"])
    
    # Setup device
    device = get_device(config["device"])
    
    # Setup logging
    log_dir = Path(config["log_dir"])
    logger = setup_logging(log_dir, config["log_level"])
    
    logger.info(f"Starting FICE training with config: {config}")
    logger.info(f"Using device: {device}")
    
    try:
        # Setup directories
        root_project_folder = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        data_root = root_project_folder / config["data_root"]
        queries_root = root_project_folder / config["queries_root"]
        checkpoint_dir = Path(config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info("Loading dataset...")
        nt_path = data_root / "raw/swdf.nt"
        
        if not nt_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {nt_path}")
        
        graph = datasets.NTDatasets(root=data_root, nt_path=nt_path)
        graph = graph[0]
        logger.info(f"Dataset loaded. Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")

        # Load metadata
        logger.info("Loading metadata...")
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
        logger.info("Metadata loaded successfully")

        # Load queries
        logger.info("Loading queries...")
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
            device=device
        )
        logger.info(f"Queries loaded. Total queries: {len(queries)}")

        # Train-test split
        queries_list = list(queries)
        train_size = int(len(queries_list) * config["train_ratio"])
        test_size = len(queries_list) - train_size

        # Shuffle queries for random split
        random.shuffle(queries_list)
        train_queries = queries_list[:train_size]
        test_queries = queries_list[train_size:]

        logger.info(f"Data split - Train: {len(train_queries)}, Test: {len(test_queries)}")

        # Create data loaders
        train_loader = DataLoader(
            dataset=train_queries, 
            batch_size=config["batch_size"], 
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_queries, 
            batch_size=1,  # Use batch_size=1 for evaluation
            shuffle=False
        )

        # Initialize model
        logger.info("Initializing model...")
        model = FICE(data=graph)
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created. Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Training metrics
        best_test_loss = float('inf')
        train_losses = []
        test_losses = []

        # Training loop
        logger.info(f"Starting training for {config['epochs']} epochs...")
        
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{config['epochs']}",
                leave=False
            )
            
            for batch_idx, query_batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                # Forward pass
                log_cardinality_pred = model(batch=query_batch)
                log_cardinality_pred = log_cardinality_pred.to(torch.float32)
                true_cardinality = query_batch.y.to(torch.float32)
                
                # Compute loss
                loss = F.mse_loss(log_cardinality_pred, true_cardinality)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_train_loss/num_train_batches:.4f}"
                })
            
            avg_train_loss = epoch_train_loss / num_train_batches
            train_losses.append(avg_train_loss)
            
            # Evaluation phase
            if (epoch + 1) % config["eval_every"] == 0:
                model.eval()
                epoch_test_loss = 0.0
                test_predictions = []
                test_true_values = []
                
                with torch.no_grad():
                    for query_batch in tqdm(test_loader, desc="Evaluating", leave=False):
                        # Forward pass
                        log_cardinality_pred = model(batch=query_batch)
                        log_cardinality_pred = log_cardinality_pred.to(torch.float32)
                        true_cardinality = query_batch.y.to(torch.float32)
                        
                        # Compute loss
                        loss = F.mse_loss(log_cardinality_pred, true_cardinality)
                        epoch_test_loss += loss.item()
                        
                        # Store predictions for analysis
                        test_predictions.extend(log_cardinality_pred.cpu().numpy().tolist())
                        test_true_values.extend(true_cardinality.cpu().numpy().tolist())
                
                avg_test_loss = epoch_test_loss / len(test_loader)
                test_losses.append(avg_test_loss)
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch+1}/{config['epochs']} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Test Loss: {avg_test_loss:.6f}"
                )
                
                # Save best model
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_model_path = checkpoint_dir / "best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'test_loss': avg_test_loss,
                        'config': config
                    }, best_model_path)
                    logger.info(f"New best model saved with test loss: {best_test_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.6f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % config["save_every"] == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'test_loss': test_losses[-1] if test_losses else None,
                    'config': config
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        logger.info("Training completed!")
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        model.eval()
        final_test_loss = 0.0
        final_predictions = []
        final_true_values = []
        
        with torch.no_grad():
            for query_batch in tqdm(test_loader, desc="Final evaluation"):
                log_cardinality_pred = model(batch=query_batch)
                log_cardinality_pred = log_cardinality_pred.to(torch.float32)
                true_cardinality = query_batch.y.to(torch.float32)
                
                loss = F.mse_loss(log_cardinality_pred, true_cardinality)
                final_test_loss += loss.item()
                
                final_predictions.extend(log_cardinality_pred.cpu().numpy().tolist())
                final_true_values.extend(true_cardinality.cpu().numpy().tolist())
        
        final_avg_test_loss = final_test_loss / len(test_loader)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(np.array(final_predictions) - np.array(final_true_values)))
        mse = np.mean((np.array(final_predictions) - np.array(final_true_values))**2)
        rmse = np.sqrt(mse)
        
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Final Test Loss (MSE): {final_avg_test_loss:.6f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.6f}")
        logger.info(f"Root Mean Square Error (RMSE): {rmse:.6f}")
        logger.info(f"Best Test Loss: {best_test_loss:.6f}")
        logger.info(f"Sample predictions: {final_predictions[:5]}")
        logger.info(f"Sample true values: {final_true_values[:5]}")
        
        # Gradient flow analysis
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
        
        logger.info("Training script completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

    # root_project_folder = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # # 1. Load dataset
    # # Using FB15k-237 for demonstration. It will be downloaded if not present.
    # print("Loading dataset...")
    # # dataset = datasets.FB15k237(root="data/fb15k237")
    # # train_data = dataset[0]
    # root = root_project_folder / "data/swdf/"
    # nt_path = root / "raw/swdf.nt"

    # graph = datasets.NTDatasets(root=root, nt_path=nt_path)
    # graph = graph[0]
    # print("Dataset loaded.")

    # print("Loading occurrences...")
    # with open(f"{root}/processed/swdf_entity_occurrences.json", "r") as f:
    #     entity_occurrences = json.load(f)
    # with open(f"{root}/processed/swdf_relation_occurrences.json", "r") as f:
    #     relation_occurrences = json.load(f)
    # print("Occurrences loaded.")

    # print("Loading mappings...")
    # with open(f"{root}/processed/swdf_entity_to_id.json", "r") as f:
    #     entity_mapping = json.load(f)
    # with open(f"{root}/processed/swdf_relation_to_id.json", "r") as f:
    #     relation_mapping = json.load(f)
    # print("Mappings loaded.")

    # print("Load queries...")
    # query_root = root_project_folder / "data/swdf_queries"
    # query_path = query_root / "raw/Joined_Queries.json"
    # # ...existing code...
    # queries = datasets.NTQueries(
    #     root=query_root,
    #     queries_path=query_path,
    #     entity_occurrences=entity_occurrences,
    #     relation_occurrences=relation_occurrences,
    #     global_entity_mapping=entity_mapping,
    #     global_relation_mapping=relation_mapping,
    #     device=device
    # )
    # print("Queries loaded.")

    # # Convert queries to list for splitting
    # queries_list = list(queries)
    # print(f"Total queries: {len(queries_list)}")
    # # Train-test split
    
    # train_ratio = 0.8 
    # train_size = int(len(queries_list) * train_ratio)
    # test_size = len(queries_list) - train_size

    # # Shuffle queries for random split
    # import random
    # random.seed(420)  # For reproducibility
    # random.shuffle(queries_list)

    # train_queries = queries_list[:train_size]
    # test_queries = queries_list[train_size:]

    # print(f"Train queries: {len(train_queries)}")
    # print(f"Test queries: {len(test_queries)}")
    # batch_size = 16
    # loader = DataLoader(dataset=train_queries, batch_size=batch_size, shuffle=False)


    # # You can replace FICE with your own model class
    # model = FICE(data=graph)
    # model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # print("\nModel created:")
    # print(model)

    # # Training loop
    # num_epochs = 3  # You can adjust this
    # print(f"\nStarting training for {num_epochs} epochs...")

    # for epoch in range(num_epochs):
    #     epoch_loss = 0.0
    #     num_queries = 0

    #     progress_bar = tqdm(total=len(train_queries), desc=f"Training Epoch {epoch+1}")
    #     for query_batch in loader:
    #         num_queries += query_batch.num_graphs
    #         progress_bar.update(query_batch.num_graphs)

    #         graph_size = torch.tensor(
    #             [graph.num_edges], device=device
    #         ) 

    #         # Getting a cardinality prediction from the combined model
    #         log_cardinality_pred = model(
    #             batch=query_batch,
    #         )
    #         log_cardinality_pred = log_cardinality_pred.to(torch.float32)
    #         true_cardinality = query_batch.y.to(torch.float32)

    #         loss = F.mse_loss(log_cardinality_pred, true_cardinality)
            
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         epoch_loss += loss.item()
        
    #     avg_epoch_loss = epoch_loss / num_queries
    #     print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_epoch_loss:.6f}")

    # print("Training completed!")

    # # Evaluation on test set
    # print("\nEvaluating on test set...")
    # test_loss = 0.0
    # test_predictions = []
    # test_true_values = []

    # with torch.no_grad():
    #     loader = DataLoader(dataset=test_queries, batch_size=1, shuffle=False)
    #     for query in tqdm(loader, desc="Evaluating test queries"):
    #         graph_size = torch.tensor(
    #             [graph.num_edges], device=device
    #         ) 

    #         log_cardinality_pred = model(
    #             data=graph,
    #             batch=query,
    #             graph_size=graph_size
    #         )
    #         true_cardinality = torch.tensor([0.3], device=device)
            
    #         loss = F.mse_loss(log_cardinality_pred, true_cardinality)
    #         test_loss += loss.item()
            
    #         test_predictions.append(log_cardinality_pred.item())
    #         test_true_values.append(true_cardinality.item())
    
    # avg_test_loss = test_loss / len(test_queries)
    # print(f"Test Loss: {avg_test_loss:.6f}")
    # print(f"Sample predictions: {test_predictions[:5]}")
    # print(f"True values: {test_true_values[:5]}")

    # # Check gradients (this part remains the same)
    # # Check if we can backprop the cardinality error to the ULTRA node GNN
    # has_grad_ultra = any(
    #     p.grad is not None and p.grad.abs().sum() > 0
    #     for p in model.entity_model.parameters()
    # )
    # print(
    #     f"Grad->ULTRA: {has_grad_ultra}"
    # )

    # # Check if at least one ULTRA parameter has no gradient
    # no_grad_ultra = any(
    #     p.grad is None or p.grad.abs().sum() == 0
    #     for p in model.entity_model.parameters()
    # )
    # print(f"At least one parameter has no grad in the Entity GNN: {no_grad_ultra}")

    # # Check if at least one ULTRA parameter has no gradient
    # no_grad_ultra_relation = any(
    #     p.grad is None or p.grad.abs().sum() == 0
    #     for p in model.relation_model.parameters()
    # )
    # print(
    #     f"At least one parameter has no grad in the Relation GNN: {no_grad_ultra_relation}"
    # )
