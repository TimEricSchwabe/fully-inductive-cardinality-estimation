import torch
from easydict import EasyDict

import torch.utils.data

import os
import sys
# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fice.ultra import datasets #TODO
from fice.models import FICE

import torch.nn.functional as F



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load dataset
    # Using FB15k-237 for demonstration. It will be downloaded if not present.
    print("Loading dataset...")
    dataset = datasets.FB15k237(root="data/fb15k237")
    train_data = dataset[0]
    print("Dataset loaded.")

    
    # You can replace FICE with your own model class
    model = FICE(graph=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nModel created:")
    print(model)

    # 3. Create a single data point for forward pass
    # We take the first triple from the training set.
    h_index = train_data.target_edge_index[0, 0:1]
    t_index = train_data.target_edge_index[1, 0:1]
    r_index = train_data.target_edge_type[0:1]

    # Create a "batch" with a single positive sample.
    # The model expects a shape of (batch_size, num_samples, 3).
    # Here, batch_size=1, num_samples=1.
    batch = torch.stack([h_index, t_index, r_index], dim=1).unsqueeze(1)


    #########################################################
    ## Example Query Graph, we assume we have those saved to disk as .pt file
    query_edge_index = torch.tensor([[0, 0],
                                     [1, 2]], dtype=torch.long).to(device) 

    # Feature matrix that holds the occurrences (precomputed), otherwise zeros                                  
    query_features = torch.zeros(3, 67) # 64 features for ULTRA embeddings + 3 for occurrences
    query_features[0, :3] = torch.tensor([0,0,0])         # variable node has no occurrences
    query_features[1, :3] = torch.tensor([4.0, 5.0, 6.0]) # example occurrences
    query_features[2, :3] = torch.tensor([7.0, 8.0, 9.0])

    # List to denote which index of the query-feature should receive which ULTRA embedding
    # -1 means the node is a variable and thus has no ULTRA embedding, for the 2nd and 3rd node, we denote
    # that they are node 22 and 12 from the Knowledge Graph
    embedding_indices = torch.tensor([-1, 22, 12], dtype=torch.long)

    # Edge features - similar to query_features but for edges
    # We have 2 edges based on query_edge_index, each with 67 dimensions
    edge_attr = torch.zeros(2, 67)  # 64 features for relation embeddings + 3 for occurrences
    edge_attr[0, :3] = torch.tensor([1.0, 2.0, 3.0])  # example occurrences for first edge
    edge_attr[1, :3] = torch.tensor([4.0, 5.0, 6.0])  # example occurrences for second edge

    # List to denote which relation index each edge should receive from relation_model
    # For our 2 edges, we'll use relation indices 5 and 8 as examples
    edge_embedding_indices = torch.tensor([5, 8], dtype=torch.long)

    query_features = query_features.to(device)
    embedding_indices = embedding_indices.to(device)
    edge_attr = edge_attr.to(device)
    edge_embedding_indices = edge_embedding_indices.to(device)

    graph_size = torch.tensor([1000000], device=device) # example for the full size of the graph
    #########################################################


    # Getting a cardinality prediction from the combined model
    log_cardinality_pred = model(train_data, batch, query_features, query_edge_index, embedding_indices, edge_attr, edge_embedding_indices, graph_size)
    true_cardinality = torch.tensor([0.3], device=device)
    loss   = F.mse_loss(log_cardinality_pred, true_cardinality)
    loss.backward()


    print("Final Predicted Cardinality: ", log_cardinality_pred.item())

    # Check if we can backprop the caridnality error to the ULTRA node GNN
    has_grad_ultra = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.entity_model.parameters())
    print(f"Pred: {log_cardinality_pred.item():.4f} | Loss: {loss.item():.6f} | Grad->ULTRA: {has_grad_ultra}")

    # Check if at least one ULTRA parameter has no gradient 
    no_grad_ultra = any(p.grad is None or p.grad.abs().sum() == 0
                        for p in model.entity_model.parameters())
    print(f"At least on parameter has no Grad in the Entity GNN: {no_grad_ultra}")


    # Check if at least one ULTRA parameter has no gradient 
    no_grad_ultra_relation = any(p.grad is None or p.grad.abs().sum() == 0
                        for p in model.relation_model.parameters())
    print(f"At least on parameter has no Grad in the Relation GNN: {no_grad_ultra_relation}")

