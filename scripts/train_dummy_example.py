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

from fice.ultra import datasets  # TODO
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

    data = datasets.NTDatasets(root=root, nt_path=nt_path)
    data = data[0]
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
    print(queries_list[0])  # Print the first query for inspection

    # Train-test split
    train_ratio = 0.8
    train_size = int(len(queries_list) * train_ratio)
    test_size = len(queries_list) - train_size

    # Shuffle queries for random split
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(queries_list)

    train_queries = queries_list[:train_size]
    test_queries = queries_list[train_size:]

    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")


    # You can replace FICE with your own model class
    model = FICE(graph=None)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nModel created:")
    print(model)

    # 3. Create a single data point for forward pass
    # We take the first triple from the training set.
    print(data.edge_type)
    print(data.edge_type.shape)
    h_index = data.edge_index[0, 0:1]
    t_index = data.edge_index[1, 0:1]
    r_index = data.edge_type[0:1]
    print(h_index, t_index, r_index)

    # Create a "batch" with a single positive sample.
    # The model expects a shape of (batch_size, num_samples, 3).
    # Here, batch_size=1, num_samples=1.
    batch = torch.stack([h_index, t_index, r_index], dim=1).unsqueeze(1)

    # Training loop
    num_epochs = 3  # You can adjust this
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_queries = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for query in tqdm(train_queries):
            num_queries += 1
            
            graph_size = torch.tensor(
                [1000000], device=device
            )  # example for the full size of the graph

            # Getting a cardinality prediction from the combined model
            log_cardinality_pred = model(
                data=data,
                batch=batch,
                query_features=query.x,
                edge_attr=query.edge_attr,
                query_edge_index=query.edge_index,
                embedding_indices=query.node_embedding_indices,
                edge_embedding_indices=query.edge_embedding_indices,
                graph_size=graph_size
            )
            log_cardinality_pred = log_cardinality_pred.to(torch.float32)
            true_cardinality = query.y.to(torch.float32)

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
        for query in test_queries:
            graph_size = torch.tensor([1000000], device=device)
            
            log_cardinality_pred = model(
                data=data,
                batch=batch,
                query_features=query.x,
                query_edge_index=query.edge_index,
                edge_attr=query.edge_attr,
                embedding_indices=query.node_embedding_indices,
                edge_embedding_indices=query.edge_embedding_indices,
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
