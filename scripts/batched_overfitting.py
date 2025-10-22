import torch
import json
import random
import numpy as np

import os
import sys

# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fice import datasets
from fice.models import FICE
from pathlib import Path

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_project_folder = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # 1. Load dataset
    # Using FB15k-237 for demonstration. It will be downloaded if not present.
    print("Loading dataset...")
    # dataset = datasets.FB15k237(root="data/fb15k237")
    # train_data = dataset[0]
    root = root_project_folder / "data/swdf/"
    nt_path = root / "raw/swdf.nt"

    graph = datasets.NTDatasets(root=root, nt_path=nt_path)
    graph = graph[0]
    print("Dataset loaded.")

    print("Loading occurrences...")
    with open(f"{root}/processed/entity_occurrences.json", "r") as f:
        entity_occurrences = json.load(f)
    with open(f"{root}/processed/relation_occurrences.json", "r") as f:
        relation_occurrences = json.load(f)
    print("Occurrences loaded.")

    print("Loading mappings...")
    with open(f"{root}/processed/entity_to_id.json", "r") as f:
        entity_mapping = json.load(f)
    with open(f"{root}/processed/relation_to_id.json", "r") as f:
        relation_mapping = json.load(f)
    print("Mappings loaded.")

    print("Load queries...")
    query_root = root_project_folder / "data/swdf_queries"
    query_path = query_root / "raw/Joined_Queries.json"
    # ...existing code...
    queries = datasets.NTQueries(
        root=query_root,
        queries_path=query_path,
        entity_occurrences=entity_occurrences,
        relation_occurrences=relation_occurrences,
        global_entity_mapping=entity_mapping,
        global_relation_mapping=relation_mapping,
        device=device
    )
    print("Queries loaded.")

    # Convert queries to list for splitting
    queries_list = list(queries)
    print(f"Total queries: {len(queries_list)}")

    # Batch overfitting experiment configuration
    batch_size = 2  # Configurable batch size for batch training
    num_training_steps_per_batch = 200  # Number of training steps per batch
    tolerance = 0.01  # Tolerance for considering prediction as "correct"
    learning_rate = 1e-3  # Learning rate for the optimizer
    
    # Create data loader with batch size
    loader = DataLoader(dataset=queries_list, batch_size=batch_size, shuffle=False)
    
    print(f"\nStarting batch overfitting experiment...")
    print(f"Training mode: Batch (queries trained in batches, evaluated individually)")
    print(f"Training steps per batch: {num_training_steps_per_batch}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Tolerance for correct prediction: ±{tolerance}")
    
    correct_predictions = 0
    total_queries = 0
    
    # Process each batch
    progress_bar = tqdm(total=len(queries_list), desc="Processing queries")
    
    total_batches = 0
    batch_progress = tqdm(loader, desc="Processing batches")
    for batch_idx, query_batch in enumerate(batch_progress):
        if batch_idx == 0:
            continue
        total_batches += 1
        batch_queries = query_batch.num_graphs
        progress_bar.update(batch_queries)
        
        # Reset model parameters for each batch
        model = FICE(data=graph)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        print(f"\n--- Batch {batch_idx + 1}/{len(loader)} (contains {batch_queries} queries) ---")
        
        graph_size = torch.tensor([graph.num_edges], device=device)
        true_cardinalities = query_batch.y.to(torch.float32)
        
        print(f"True cardinalities: {[f'{val:.6f}' for val in true_cardinalities]}")
        
        # Train on the entire batch
        model.train()
        batch_training_progress = tqdm(range(num_training_steps_per_batch), 
                                     desc=f"Training Batch {batch_idx + 1}", 
                                     leave=False)
        for step in batch_training_progress:
            optimizer.zero_grad()
            query_batch = query_batch.detach().clone().to(device)
            true_cardinalities = true_cardinalities.detach().clone().to(device)
            # Forward pass on entire batch
            log_cardinality_pred = model(
                batch=query_batch,
            )
            assert log_cardinality_pred.shape[0] == batch_queries, "Prediction shape does not match number of queries in batch"
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            
            loss = F.mse_loss(log_cardinality_pred, true_cardinalities)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Print detailed information for each iteration
            pred_errors = torch.abs(log_cardinality_pred - true_cardinalities)
            correct_in_batch = (pred_errors <= tolerance).sum().item()
            
            print(f"\n  Step {step + 1}/{num_training_steps_per_batch}:")
            print(f"    Loss: {loss.item():.6f}")
            print(f"    Predictions: {[f'{val:.6f}' for val in log_cardinality_pred.cpu()]}")
            print(f"    True values: {[f'{val:.6f}' for val in true_cardinalities.cpu()]}")
            print(f"    Errors: {[f'{val:.6f}' for val in pred_errors.cpu()]}")
            print(f"    Correct predictions: {correct_in_batch}/{batch_queries}")
            print(f"    Batch accuracy: {correct_in_batch/batch_queries:.2%}")
            
            # Update progress bar with current loss
            batch_training_progress.set_postfix(loss=f"{loss.item():.4f}", correct=f"{correct_in_batch}/{batch_queries}")
        
        # Evaluate the entire batch at once
        model.eval()
        with torch.no_grad():
            print(f"\n--- Batch Evaluation (all queries at once) ---")
            batch_pred = model(batch=query_batch)
            batch_pred = batch_pred.to(torch.float32)
            batch_loss = F.mse_loss(batch_pred, true_cardinalities)
            
            # Calculate batch-level metrics
            batch_errors = torch.abs(batch_pred - true_cardinalities)
            batch_correct = (batch_errors <= tolerance).sum().item()
            
            print(f"Batch predictions: {[f'{val:.6f}' for val in batch_pred]}")
            print(f"True cardinalities: {[f'{val:.6f}' for val in true_cardinalities]}")
            print(f"Prediction errors: {[f'{val:.6f}' for val in batch_errors]}")
            print(f"Batch loss: {batch_loss.item():.6f}")
            print(f"Correct predictions in batch: {batch_correct}/{batch_queries}")
            print(f"Batch accuracy: {batch_correct/batch_queries:.2%}")

        # Evaluate each query in the batch individually
        with torch.no_grad():
            print(f"\n--- Individual Query Evaluation ---")
            test_loader = DataLoader(dataset=query_batch, batch_size=1, shuffle=False)
            eval_progress = tqdm(test_loader, 
                               desc=f"Evaluating Batch {batch_idx + 1}", 
                               leave=False)
            for test_query_idx, test_query in enumerate(eval_progress):
                total_queries += 1
                
                # Extract single query from batch
                single_query = test_query
                true_cardinality = test_query.y.unsqueeze(0)
                
                # Evaluate single query
                final_pred = model(
                    batch=single_query,
                )
                final_pred = final_pred.to(torch.float32)
                final_loss = F.mse_loss(final_pred, true_cardinality)
                
                prediction_error = abs(final_pred.item() - true_cardinality.item())
                is_correct = prediction_error <= tolerance
                
                if is_correct:
                    correct_predictions += 1
                
                print(f"\n  Query {test_query_idx + 1} in batch:")
                print(f"    Final prediction: {final_pred.item():.6f}")
                print(f"    True cardinality: {true_cardinality.item():.6f}")
                print(f"    Prediction error: {prediction_error:.6f}")
                print(f"    Correct (within tolerance): {is_correct}")
                print(f"    Final loss: {final_loss.item():.6f}")
    
    progress_bar.close()
    
    # Summary statistics
    accuracy = correct_predictions / total_queries
    print(f"\n=== BATCH OVERFITTING EXPERIMENT RESULTS ===")
    print(f"Total queries processed: {total_queries}")
    print(f"Correct predictions (within ±{tolerance}): {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Training steps per batch: {num_training_steps_per_batch}")
    print(f"Batch size: {batch_size}")
    print(f"Queries were trained in batches of {batch_size}, then evaluated individually.")
    
    if accuracy > 0.8:
        print("✅ Model shows good overfitting capability!")
    elif accuracy > 0.5:
        print("⚠️  Model shows moderate overfitting capability.")
    else:
        print("❌ Model struggles to overfit to batched examples.")

    # Check gradients
    # Check if we can backprop the cardinality error to the ULTRA node GNN
    has_grad_ultra = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.entity_model.parameters()
    )
    print(
        f"Grad->ULTRA: {has_grad_ultra}"
    )

    # Check if at least one ULTRA parameter has no gradient
    no_grad_ultra = any(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model.entity_model.parameters()
    )
    print(f"At least one parameter has no grad in the Entity GNN: {no_grad_ultra}")

    # Check if at least one ULTRA parameter has no gradient
    no_grad_ultra_relation = any(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model.relation_model.parameters()
    )
    print(
        f"At least one parameter has no grad in the Relation GNN: {no_grad_ultra_relation}"
    )   