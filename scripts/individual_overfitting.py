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

    # Individual overfitting: Process one query at a time
    batch_size = 1
    loader = DataLoader(dataset=queries_list, batch_size=batch_size, shuffle=False)

    # Individual overfitting experiment configuration
    num_training_steps_per_query = 500  # Number of training steps per individual query
    tolerance = 0.01  # Tolerance for considering prediction as "correct"
    learning_rate = 1e-3  # Learning rate for the optimizer
    
    print(f"\nStarting individual overfitting experiment...")
    print(f"Training mode: Individual (each query trained separately)")
    print(f"Training steps per query: {num_training_steps_per_query}")
    print(f"Learning rate: {learning_rate}")
    print(f"Tolerance for correct prediction: ±{tolerance}")
    
    correct_predictions = 0
    total_queries = 0
    
    # Process each query individually
    progress_bar = tqdm(total=len(queries_list), desc="Processing queries")
    
    batch_loader_progress = tqdm(loader, desc="Processing batches")
    for batch_idx, query_batch in enumerate(batch_loader_progress):
        # Process each query in the batch individually
        total_queries += 1
        progress_bar.update(1)
        
        # Reset model parameters for each query
        model = FICE(data=graph)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        
        
        graph_size = torch.tensor([graph.num_edges], device=device)
        true_cardinality = query_batch.y.to(torch.float32)
        
        print(f"True cardinality: {true_cardinality.item():.6f}")
        
        # Train on this specific query
        model.train()
        training_progress = tqdm(range(num_training_steps_per_query), 
                                desc=f"Training Query {total_queries}", 
                                leave=False)
        for step in training_progress:
            # Forward pass
            log_cardinality_pred = model(
                batch=query_batch,
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
                print(f"  Step {step + 1}/{num_training_steps_per_query}: Pred={current_pred:.6f}, True={true_val:.6f}, Diff={difference:.6f}, Loss={loss.item():.6f}")
        
        # Evaluate on the same query
        model.eval()
        with torch.no_grad():
            final_pred = model(
                batch=query_batch,
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
    
    progress_bar.close()
    
    # Summary statistics
    accuracy = correct_predictions / total_queries
    print(f"\n=== INDIVIDUAL OVERFITTING EXPERIMENT RESULTS ===")
    print(f"Total queries processed: {total_queries}")
    print(f"Correct predictions (within ±{tolerance}): {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Training steps per query: {num_training_steps_per_query}")
    print(f"Each query was trained on individually with a fresh model.")
    
    if accuracy > 0.8:
        print("✅ Model shows good overfitting capability!")
    elif accuracy > 0.5:
        print("⚠️  Model shows moderate overfitting capability.")
    else:
        print("❌ Model struggles to overfit to individual examples.")

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