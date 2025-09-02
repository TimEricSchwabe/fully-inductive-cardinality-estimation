import torch
import json
from pprint import pprint
from easydict import EasyDict

import torch.utils.data

import os
import sys

# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fice.ultra import datasets  # TODO
from fice.models import FICE

import torch.nn.functional as F
from typing import Dict


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
    with open("data/swdf/processed/swdf_entity_occurrences.json", "r") as f:
        entity_occurrences = json.load(f)
    with open("data/swdf/processed/swdf_relation_occurrences.json", "r") as f:
        relation_occurrences = json.load(f)
    print("Occurrences loaded.")

    print("Loading mappings...")
    with open("data/swdf/processed/swdf_entity_to_id.json", "r") as f:
        entity_mapping = json.load(f)
    with open("data/swdf/processed/swdf_relation_to_id.json", "r") as f:
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


    # You can replace FICE with your own model class
    model = FICE(graph=None)

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

    #########################################################
    ## Example Query Graph, we assume we have those saved to disk as .pt file
    for query in queries:
        print(query)

        graph_size = torch.tensor(
            [1000000], device=device
        )  # example for the full size of the graph
        #########################################################

        # Getting a cardinality prediction from the combined model
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
        loss.backward()

        print("Final Predicted Cardinality: ", log_cardinality_pred.item())
    # Print the true cardinality used for training

    # Check if we can backprop the cardinality error to the ULTRA node GNN
    has_grad_ultra = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.entity_model.parameters()
    )
    print(
        f"Pred: {log_cardinality_pred.item():.4f} | Loss: {loss.item():.6f} | Grad->ULTRA: {has_grad_ultra}"
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
