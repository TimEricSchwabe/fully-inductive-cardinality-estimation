import torch
import json
from pprint import pprint
from easydict import EasyDict
import random

import torch.utils.data

import os
import sys

# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fice.ultra import datasets
from fice.models import FICE

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict
from tqdm import tqdm


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load dataset
    # Using FB15k-237 for demonstration. It will be downloaded if not present.
    print("Loading dataset...")
    # dataset = datasets.FB15k237(root="data/fb15k237")
    # train_data = dataset[0]
    root = "data/swdf/"
    nt_path = "data/swdf/raw/swdf.nt"

    graph = datasets.NTDatasets(root=root, nt_path=nt_path)
    graph = graph[0]
    print("Dataset loaded.")

    print("Loading occurrences...")
    with open(f"{root}processed/swdf_entity_occurrences.json", "r") as f:
        entity_occurrences = json.load(f)
    with open(f"{root}processed/swdf_relation_occurrences.json", "r") as f:
        relation_occurrences = json.load(f)
    print("Occurrences loaded.")

    print("Loading mappings...")
    with open(f"{root}processed/swdf_entity_to_id.json", "r") as f:
        entity_mapping = json.load(f)
    with open(f"{root}processed/swdf_relation_to_id.json", "r") as f:
        relation_mapping = json.load(f)
    print("Mappings loaded.")

    print("Load queries...")
    query_root = "data/swdf_queries"
    query_path = "data/swdf_queries/raw/Joined_Queries.json"
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
    # Train-test split
    
    train_ratio = 0.8 
    train_size = int(len(queries_list) * train_ratio)
    test_size = len(queries_list) - train_size

    # Shuffle queries for random split
    import random
    random.seed(420)  # For reproducibility
    random.shuffle(queries_list)

    train_queries = queries_list[:train_size]
    test_queries = queries_list[train_size:]

    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")
    # TODO: batch_size 1 testen
    batch_size = 16
    loader = DataLoader(dataset=train_queries, batch_size=batch_size, shuffle=False)


    # You can replace FICE with your own model class
    model = FICE()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nModel created:")
    print(model)

    # Training loop
    num_epochs = 3  # You can adjust this
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_queries = 0

        progress_bar = tqdm(total=len(train_queries), desc=f"Training Epoch {epoch+1}")
        for query_batch in loader:
            num_queries += query_batch.num_graphs
            progress_bar.update(query_batch.num_graphs)

            graph_size = torch.tensor(
                [graph.num_edges], device=device
            ) 

            # Getting a cardinality prediction from the combined model
            log_cardinality_pred = model(
                data=graph,
                batch=query_batch,
                graph_size=graph_size,
            )
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            true_cardinality = query_batch.y.to(torch.float32)

            loss = F.mse_loss(log_cardinality_pred, true_cardinality)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / num_queries
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_epoch_loss:.6f}")

    print("Training completed!")

    # Evaluation on test set
    print("\nEvaluating on test set...")
    test_loss = 0.0
    test_predictions = []
    test_true_values = []

    with torch.no_grad():
        loader = DataLoader(dataset=test_queries, batch_size=1, shuffle=False)
        for query in tqdm(loader, desc="Evaluating test queries"):
            graph_size = torch.tensor(
                [graph.num_edges], device=device
            ) 

            log_cardinality_pred = model(
                data=graph,
                batch=query,
                graph_size=graph_size
            )
            true_cardinality = torch.tensor([0.3], device=device)
            
            loss = F.mse_loss(log_cardinality_pred, true_cardinality)
            test_loss += loss.item()
            
            test_predictions.append(log_cardinality_pred.item())
            test_true_values.append(true_cardinality.item())
    
    avg_test_loss = test_loss / len(test_queries)
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Sample predictions: {test_predictions[:5]}")
    print(f"True values: {test_true_values[:5]}")

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
