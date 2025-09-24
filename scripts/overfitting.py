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
    with open(f"{root}/processed/swdf_entity_occurrences.json", "r") as f:
        entity_occurrences = json.load(f)
    with open(f"{root}/processed/swdf_relation_occurrences.json", "r") as f:
        relation_occurrences = json.load(f)
    print("Occurrences loaded.")

    print("Loading mappings...")
    with open(f"{root}/processed/swdf_entity_to_id.json", "r") as f:
        entity_mapping = json.load(f)
    with open(f"{root}/processed/swdf_relation_to_id.json", "r") as f:
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

    batch_size = 1
    loader = DataLoader(dataset=queries_list, batch_size=batch_size, shuffle=False)


    # You can replace FICE with your own model class
    model = FICE(data=graph)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nModel created:")
    print(model)

    # Overfitting experiment configuration
    num_training_steps_per_batch = 500  # Number of training steps per batch
    tolerance = 0.01  # Tolerance for considering prediction as "correct"
    
    # Automatically select training mode based on batch size
    # Individual mode: batch_size == 1 (train on single queries)
    # Batch mode: batch_size > 1 (train on entire batches, then evaluate individually)
    training_mode = 'individual' if batch_size == 1 else 'batch'
    
    print(f"\nStarting overfitting experiment...")
    print(f"Training mode: {training_mode} (auto-selected based on batch_size={batch_size})")
    print(f"Training steps per batch: {num_training_steps_per_batch}")
    print(f"Batch size: {batch_size}")
    print(f"Tolerance for correct prediction: ±{tolerance}")
    
    correct_predictions = 0
    total_queries = 0
    
    # Process each batch
    progress_bar = tqdm(total=len(queries_list), desc="Processing queries")
    
    if training_mode == 'individual':
        # Individual mode: Train on each query separately
        batch_loader_progress = tqdm(loader, desc="Processing batches")
        for batch_idx, query_batch in enumerate(batch_loader_progress):
            # Process each query in the batch individually
            query_progress = tqdm(range(query_batch.num_graphs), 
                                desc=f"Queries in Batch {batch_idx + 1}", 
                                leave=False)
            for query_idx in query_progress:
                total_queries += 1
                progress_bar.update(1)
                
                # Reset model parameters for each query
                model = FICE(data=graph)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                
                # Extract single query from batch
                single_query = query_batch[query_idx].unsqueeze(0)
                
                print(f"\n--- Query {total_queries} (Batch {batch_idx + 1}, Item {query_idx + 1}) ---")
                
                graph_size = torch.tensor([graph.num_edges], device=device)
                true_cardinality = single_query.y.to(torch.float32)
                
                print(f"True cardinality: {true_cardinality.item():.6f}")
                
                # Train on this specific query
                model.train()
                training_progress = tqdm(range(num_training_steps_per_batch), 
                                       desc=f"Training Query {total_queries}", 
                                       leave=False)
                for step in training_progress:
                    # Forward pass
                    log_cardinality_pred = model(
                        data=graph,
                        batch=single_query,
                        graph_size=graph_size,
                    )
                    log_cardinality_pred = log_cardinality_pred.to(torch.float32)
                    
                    loss = F.mse_loss(log_cardinality_pred, true_cardinality)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Print intermediate values every 20 iterations
                    if (step + 1) % 20 == 0:
                        current_pred = log_cardinality_pred.item()
                        true_val = true_cardinality.item()
                        difference = abs(current_pred - true_val)
                        print(f"  Step {step + 1}/{num_training_steps_per_batch}: Pred={current_pred:.6f}, True={true_val:.6f}, Diff={difference:.6f}, Loss={loss.item():.6f}")
                
                # Evaluate on the same query
                model.eval()
                with torch.no_grad():
                    final_pred = model(
                        data=graph,
                        batch=single_query,
                        graph_size=graph_size
                    )
                    final_pred = final_pred.to(torch.float32)
                    final_loss = F.mse_loss(final_pred, true_cardinality)
                    
                    prediction_error = abs(final_pred.item() - true_cardinality.item())
                    is_correct = prediction_error <= tolerance
                    
                    if is_correct:
                        correct_predictions += 1
                    
                    print(f"  Final prediction: {final_pred.item():.6f}")
                    print(f"  True cardinality: {true_cardinality.item():.6f}")
                    print(f"  Prediction error: {prediction_error:.6f}")
                    print(f"  Correct (within tolerance): {is_correct}")
                    print(f"  Final loss: {final_loss.item():.6f}")
    
    else:  # batch mode
        # Batch mode: Train on entire batch, then evaluate each query individually
        total_batches = 0
        batch_progress = tqdm(loader, desc="Processing batches")
        for batch_idx, query_batch in enumerate(batch_progress):
            total_batches += 1
            batch_queries = query_batch.num_graphs
            progress_bar.update(batch_queries)
            
            # Reset model parameters for each batch
            model = FICE()
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
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
                # Forward pass on entire batch
                log_cardinality_pred = model(
                    data=graph,
                    batch=query_batch,
                    graph_size=graph_size,
                )
                log_cardinality_pred = log_cardinality_pred.to(torch.float32)
                
                loss = F.mse_loss(log_cardinality_pred, true_cardinalities)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print intermediate values every 20 iterations
                if (step + 1) % 20 == 0:
                    batch_pred = log_cardinality_pred.cpu().numpy()
                    batch_true = true_cardinalities.cpu().numpy()
                    batch_diff = abs(batch_pred - batch_true)
                    print(f"  Step {step + 1}/{num_training_steps_per_batch}: Loss={loss.item():.6f}")
                    print(f"    Predictions: {[f'{val:.6f}' for val in batch_pred]}")
                    print(f"    Differences: {[f'{val:.6f}' for val in batch_diff]}")
            
            # Evaluate each query in the batch individually
            model.eval()
            with torch.no_grad():
                eval_progress = tqdm(range(batch_queries), 
                                   desc=f"Evaluating Batch {batch_idx + 1}", 
                                   leave=False)
                for query_idx in eval_progress:
                    total_queries += 1
                    
                    # Extract single query from batch
                    single_query = query_batch[query_idx].unsqueeze(0)
                    true_cardinality = true_cardinalities[query_idx].unsqueeze(0)
                    
                    # Evaluate single query
                    final_pred = model(
                        data=graph,
                        batch=single_query,
                        graph_size=graph_size
                    )
                    final_pred = final_pred.to(torch.float32)
                    final_loss = F.mse_loss(final_pred, true_cardinality)
                    
                    prediction_error = abs(final_pred.item() - true_cardinality.item())
                    is_correct = prediction_error <= tolerance
                    
                    if is_correct:
                        correct_predictions += 1
                    
                    print(f"\n  Query {query_idx + 1} in batch:")
                    print(f"    Final prediction: {final_pred.item():.6f}")
                    print(f"    True cardinality: {true_cardinality.item():.6f}")
                    print(f"    Prediction error: {prediction_error:.6f}")
                    print(f"    Correct (within tolerance): {is_correct}")
                    print(f"    Final loss: {final_loss.item():.6f}")
    
    progress_bar.close()
    
    # Summary statistics
    accuracy = correct_predictions / total_queries
    print(f"\n=== OVERFITTING EXPERIMENT RESULTS ===")
    print(f"Training mode: {training_mode}")
    print(f"Total queries processed: {total_queries}")
    print(f"Correct predictions (within ±{tolerance}): {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Training steps per batch: {num_training_steps_per_batch}")
    
    if training_mode == 'individual':
        print(f"Each query was trained on individually with a fresh model.")
    else:
        print(f"Queries were trained in batches of {batch_size}, then evaluated individually.")
    
    if accuracy > 0.8:
        print("✅ Model shows good overfitting capability!")
    elif accuracy > 0.5:
        print("⚠️  Model shows moderate overfitting capability.")
    else:
        print("❌ Model struggles to overfit to individual examples.")

    # Check gradients (this part remains the same)
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
