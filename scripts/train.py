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
import utils.helper
import utils.metrics
import math
import pandas as pd
import matplotlib.pyplot as plt

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
    "epochs": 5,
    "batch_size": 1,
    "learning_rate": 1e-5,
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
    'DEBUG_num_queries': 10,  # Set to None to use all queries
}


if __name__ == "__main__":
    # Use config dictionary instead of argument parsing
    config = CONFIG.copy()  # Make a copy so we can modify if needed

    root_project_folder = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = root_project_folder / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config["data_root"] = root_project_folder / "data/swdf/"
    config["queries_root"] = root_project_folder / "data/swdf_queries"
    config["log_dir"] = run_dir / "logs/"
    config["checkpoint_dir"] = run_dir / "checkpoints/"

    # Set random seeds for reproducibility
    utils.helper.set_random_seeds(config["seed"])

    # Setup device
    device = utils.helper.get_device(config["device"])

    # Setup logging
    log_dir = Path(config["log_dir"])
    logger = utils.helper.setup_logging(log_dir, config["log_level"])


    logger.info(f"Starting FICE training with config: {config}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Using device: {device}")

    logger.info(f"Random seed: {config['seed']}")
    logger.info(f"Log directory: {log_dir}")

   
    try:
        # Setup directories
        data_root = Path(config["data_root"])
        queries_root = Path(config["queries_root"])
        checkpoint_dir = Path(config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Project root: {root_project_folder}")
        logger.info(f"Data root: {data_root}")
        logger.info(f"Queries root: {queries_root}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")

        
        # Load dataset
        logger.info("Loading dataset...")
        nt_path = data_root / "raw/swdf.nt"
        
        if not nt_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {nt_path}")
        
        graph = datasets.NTDatasets(root=data_root, nt_path=nt_path)
        graph = graph[0]

        logger.info(f"Graph loaded successfully!")
        logger.info(f"Number of nodes: {graph.num_nodes}")
        logger.info(f"Number of edges: {graph.num_edges}")
        logger.info(f"Node features shape: {graph.x.shape if hasattr(graph, 'x') and graph.x is not None else 'None'}")
        logger.info(f"Edge indices shape: {graph.edge_index.shape}")
        logger.info(f"Edge types shape: {graph.edge_type.shape if hasattr(graph, 'edge_type') and graph.edge_type is not None else 'None'}")

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

        logger.info("Metadata loaded successfully!")
        logger.info(f"Entities with occurrences: {len(entity_occurrences)}")
        logger.info(f"Relations with occurrences: {len(relation_occurrences)}")
        logger.info(f"Entity mapping size: {len(entity_mapping)}")
        logger.info(f"Relation mapping size: {len(relation_mapping)}")



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
        queries_list = [q for q in queries_list if q.y.item() <= 30]
        random.shuffle(queries_list)

        if config['DEBUG_num_queries'] is not None:
            queries_list = queries_list[:config['DEBUG_num_queries']]
        

        train_size = int(len(queries_list) * config["train_ratio"])
        test_size = len(queries_list) - train_size

        # Shuffle queries for random split
        train_queries = queries_list[:train_size]
        test_queries = queries_list[train_size:]

        logger.info(f"Data split - Train: {len(train_queries)}, Test: {len(test_queries)}")

        logger.info(f"Dataset split:")
        logger.info(f"  Total queries: {len(queries_list)}")
        logger.info(f"  Train queries: {len(train_queries)} ({len(train_queries)/len(queries_list)*100:.1f}%)")
        logger.info(f"  Test queries: {len(test_queries)} ({len(test_queries)/len(queries_list)*100:.1f}%)")
        logger.info(f"  Train ratio: {config['train_ratio']}")

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

        logger.info(f"Model initialized successfully!")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Optimizer: {optimizer.__class__.__name__}")
        logger.info(f"Learning rate: {config['learning_rate']}")
        logger.info(f"Weight decay: {config['weight_decay']}")
        
        # Training metrics
        best_test_loss = float('inf')

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



        # Training loop
        logger.info(f"Starting training for {config['epochs']} epochs...")

        def tensor_to_numpy(tensor):
            return tensor.detach().cpu().numpy().tolist()


        # Training loop
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

            epoch_train_predictions = []
            epoch_train_true_values = []

            epoch_train_predictions_log = []
            epoch_train_true_values_log = []
            for batch_idx, query_batch in enumerate(progress_bar):
                optimizer.zero_grad()

                # Forward pass
                query_batch = query_batch.to(device)
                log_cardinality_pred = model(batch=query_batch)
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
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_train_loss/num_train_batches:.4f}"
                })
            
 
            epoch_avg_train_loss = epoch_train_loss / num_train_batches

            mse = utils.metrics.calculate_mse(epoch_train_predictions, epoch_train_true_values)
            mae = utils.metrics.calculate_mae(epoch_train_predictions, epoch_train_true_values)
            rmse = np.sqrt(mse)
            q_error = utils.metrics.calculate_q_error(epoch_train_predictions, epoch_train_true_values)

            train_losses.append(epoch_avg_train_loss)
            train_mse.append(mse)
            train_mae.append(mae)
            train_rmse.append(rmse)
            train_q_error.append(q_error)
            train_predictions_log.append(epoch_train_predictions_log)
            train_true_values_log.append(epoch_train_true_values_log)


            # Evaluation phase
            if (epoch + 1) % config["eval_every"] == 0:
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

                        log_cardinality_pred = model(batch=query_batch)
                        log_cardinality_pred = log_cardinality_pred.to(torch.float32)
                        log_true_cardinality = torch.log(query_batch.y.to(torch.float32))

                        epoch_test_predictions_log.extend(tensor_to_numpy(log_cardinality_pred))
                        epoch_test_true_values_log.extend(tensor_to_numpy(log_true_cardinality))



                        # Compute loss
                        loss = F.mse_loss(log_cardinality_pred, log_true_cardinality)
                        epoch_test_loss += loss.item()

                        # Store predictions for analysis
                        # Store predictions for analysis
                        prediction = torch.exp(log_cardinality_pred)
                        true_value = query_batch.y.to(torch.float32)

                        epoch_test_predictions.extend(tensor_to_numpy(prediction))
                        epoch_test_true_values.extend(tensor_to_numpy(true_value))

                avg_test_loss = epoch_test_loss / len(test_loader)

                mse = utils.metrics.calculate_mse(epoch_test_predictions, epoch_test_true_values)
                mae = utils.metrics.calculate_mae(epoch_test_predictions, epoch_test_true_values)
                rmse = np.sqrt(mse)
                q_error = utils.metrics.calculate_q_error(epoch_test_predictions, epoch_test_true_values)

                test_losses.append(avg_test_loss)
                test_mse.append(mse)
                test_mae.append(mae)
                test_rmse.append(rmse)
                test_q_error.append(q_error)
                test_predictions_log.append(epoch_test_predictions_log)
                test_true_values_log.append(epoch_test_true_values_log)



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
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_avg_train_loss,
                        'test_loss': avg_test_loss,
                        'config': config
                    }, best_model_path)
                    logger.info(f"New best model saved with test loss: {best_test_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {epoch_avg_train_loss:.6f}")

            # Save checkpoint periodically
            if (epoch + 1) % config["save_every"] == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_avg_train_loss,
                    'test_loss': test_losses[-1] if test_losses else None,
                    'config': config
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        logger.info("Training completed!")

        
        # # Final evaluation
        # logger.info("Performing final evaluation...")
        # model.eval()
        # final_test_loss = 0.0
        # final_predictions = []
        # final_true_values = []
        
        # with torch.no_grad():
        #     for query_batch in tqdm(test_loader, desc="Final evaluation"):
        #         log_cardinality_pred = model(batch=query_batch)
        #         log_cardinality_pred = log_cardinality_pred.to(torch.float32)
        #         true_cardinality = query_batch.y.to(torch.float32)
                
        #         loss = F.mse_loss(log_cardinality_pred, true_cardinality)
        #         final_test_loss += loss.item()
                
        #         final_predictions.extend(log_cardinality_pred.cpu().numpy().tolist())
        #         final_true_values.extend(true_cardinality.cpu().numpy().tolist())
        
        # final_avg_test_loss = final_test_loss / len(test_loader)
        
        # # Calculate additional metrics
        # mae = np.mean(np.abs(np.array(final_predictions) - np.array(final_true_values)))
        # mse = np.mean((np.array(final_predictions) - np.array(final_true_values))**2)
        # rmse = np.sqrt(mse)
        
        # logger.info("=" * 60)
        # logger.info("FINAL RESULTS")
        # logger.info("=" * 60)
        # logger.info(f"Final Test Loss (MSE): {final_avg_test_loss:.6f}")
        # logger.info(f"Mean Absolute Error (MAE): {mae:.6f}")
        # logger.info(f"Root Mean Square Error (RMSE): {rmse:.6f}")
        # logger.info(f"Best Test Loss: {best_test_loss:.6f}")
        # logger.info(f"Sample predictions: {final_predictions[:20]}")
        # logger.info(f"Sample true values: {final_true_values[:20]}")
        
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
        


        epochs_range = range(1, len(train_losses) + 1)
        eval_epochs = range(1, len(train_losses) + 1, config["eval_every"])
        df = pd.DataFrame({
            'epochs_range': epochs_range,
            'eval_range': eval_epochs,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_q_error': train_q_error,
            'test_q_error': test_q_error
        })

        df.to_csv(log_dir / 'training_curves.csv', index=False)

        plt.figure(figsize=(18, 10)) # Adjust figure size for better visibility

        # Plot training and test losses
        plt.subplot(2, 3, 1) # 2 rows, 3 columns, 1st plot
        plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')

        # Plot test losses at evaluation intervals
        print(train_losses)
        print(len(eval_epochs))
        print(len(test_losses))

        plt.plot(eval_epochs, test_losses, 'r-', label='Test Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)

        # # Plot predictions vs true values (final)
        # plt.subplot(2, 3, 2) # 2 rows, 3 columns, 2nd plot
        # plt.scatter(final_true_values, final_predictions, alpha=0.5)
        # plt.plot([min(final_true_values), max(final_true_values)],
        #         [min(final_true_values), max(final_true_values)], 'r--')
        # plt.xlabel('True Cardinality')
        # plt.ylabel('Predicted Cardinality')
        # plt.title('Final Predictions vs True Values')
        # plt.grid(True)

        # Plot MAE
        plt.subplot(2, 3, 3) # 2 rows, 3 columns, 3rd plot
        plt.plot(epochs_range, train_mae, 'b-', label='Train MAE')
        plt.plot(eval_epochs, test_mae, 'r-', label='Test MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True)

        # Plot RMSE
        plt.subplot(2, 3, 4) # 2 rows, 3 columns, 4th plot
        plt.plot(epochs_range, train_rmse, 'b-', label='Train RMSE')
        plt.plot(eval_epochs, test_rmse, 'r-', label='Test RMSE')

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Root Mean Square Error')
        plt.legend()
        plt.grid(True)

        # Plot Q-Error
        plt.subplot(2, 3, 5) # 2 rows, 3 columns, 5th plot
        plt.plot(epochs_range, train_q_error, 'b-', label='Train Q-ERROR')
        plt.plot(eval_epochs, test_q_error, 'r-', label='Test Q-ERROR')

        plt.xlabel('Epoch')
        plt.ylabel('Q-Error')
        plt.title('Average Q-Error')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        
        # Save the training metrics plot
        metrics_plot_path = log_dir / 'training_metrics.png'
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training metrics plot saved to: {metrics_plot_path}")
        
        plt.show()

        print("Training curves and metrics plotted!")

        # Create separate figure for predictions vs true values across all epochs
        plt.figure(figsize=(16, 8))

        # Randomly select indices to plot from each epoch to avoid overcrowding
        max_points_per_epoch = 10  # Adjust this number as needed
        np.random.seed(config["seed"])  # For reproducible random selection

        # Helper function to randomly sample indices
        def get_random_indices(data_length, max_points):
            if data_length <= max_points:
                return list(range(data_length))
            return np.random.choice(data_length, max_points, replace=False).tolist()

        idx = get_random_indices(len(train_predictions_log[0]), max_points_per_epoch)
        test_idx = get_random_indices(len(test_predictions_log[0]), max_points_per_epoch)
        train_predictions_log = [[epoch_preds[i] for i in idx] for epoch_preds in train_predictions_log]
        train_true_values_log = [[epoch_trues[i] for i in idx] for epoch_trues in train_true_values_log]
        test_predictions_log = [[epoch_preds[i] for i in test_idx] for epoch_preds in test_predictions_log]
        test_true_values_log = [[epoch_trues[i] for i in test_idx] for epoch_trues in test_true_values_log]
        # Plot train predictions vs true values for all epochs
        plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
        if train_predictions_log and train_true_values_log:
            for epoch_idx, (preds, trues) in enumerate(zip(train_predictions_log, train_true_values_log)):
                if len(preds) > 0 and len(trues) > 0:
                    plt.scatter(trues, preds, alpha=0.3, 
                               s=20, label=f'Epoch {epoch_idx+1}' if epoch_idx % 5 == 0 else "")
        
        # Add perfect prediction line
        if train_true_values_log:
            all_train_trues = [val for epoch_trues in train_true_values_log for val in epoch_trues]
            if all_train_trues:
                min_val, max_val = min(all_train_trues), max(all_train_trues)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        plt.xlabel('True Cardinality')
        plt.ylabel('Predicted Cardinality')
        plt.title('Train Predictions vs True Values (All Epochs)')
        plt.grid(True)
        # Only show legend for every 10th epoch to avoid clutter
        if len(train_predictions_log) <= 10:
            plt.legend(fontsize=8)

        # Plot test predictions vs true values for all epochs
        plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
        if test_predictions_log and test_true_values_log:
            test_colors = plt.cm.plasma(np.linspace(0, 1, len(test_predictions_log)))
            for epoch_idx, (preds, trues) in enumerate(zip(test_predictions_log, test_true_values_log)):
                if len(preds) > 0 and len(trues) > 0:
                    plt.scatter(trues, preds, alpha=0.3, color=test_colors[epoch_idx], 
                               s=20, label=f'Epoch {epoch_idx+1}' if epoch_idx % 5 == 0 else "")
        # Add perfect prediction line
        if test_true_values_log:
            all_test_trues = [val for epoch_trues in test_true_values_log for val in epoch_trues]
            if all_test_trues:
                min_val, max_val = min(all_test_trues), max(all_test_trues)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        all_test_trues = [val for epoch_trues in test_true_values_log for val in epoch_trues]
        min_val, max_val = min(all_test_trues), max(all_test_trues)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('True Cardinality')
        plt.ylabel('Predicted Cardinality')
        plt.title('Test Predictions vs True Values (All Epochs)')
        plt.grid(True)
        # Only show legend for every 5th evaluation epoch to avoid clutter
        if len(test_predictions_log) <= 10:
            plt.legend(fontsize=8)

        plt.tight_layout()
        
        # Save the predictions plot
        predictions_plot_path = log_dir / 'predictions_vs_true_values_all_epochs.png'
        plt.savefig(predictions_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Predictions vs True Values plot saved to: {predictions_plot_path}")
        
        plt.show()

        print("Predictions vs True Values across all epochs plotted!")


        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise   



