import torch
from easydict import EasyDict

import torch.utils.data

import os
import sys
# Ensure the fice package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fice.ultra import datasets #TODO
from fice.models import FICE



if __name__ == "__main__":
    # 1. Load dataset
    # Using FB15k-237 for demonstration. It will be downloaded if not present.
    print("Loading dataset...")
    dataset = datasets.FB15k237(root="data/fb15k237")
    train_data = dataset[0]
    print("Dataset loaded.")

    
    # You can replace FICE with your own model class
    model = FICE(graph=None)
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

    print(f"\nUsing single triple for forward pass: (h, t, r) = ({h_index.item()}, {t_index.item()}, {r_index.item()})")
    print(f"Batch created with shape: {batch.shape}")

    # 4. Forward pass
    print("\nPerforming forward pass...")
    try:
        # Move data to the same device as the model
        device = next(model.parameters()).device
        train_data = train_data.to(device)
        batch = batch.to(device)

        relation_representations, node_representations = model(train_data, batch) # forward pass
        
        print(f"Forward pass successful!")
        print("\n=== Forward Pass Output Statistics ===")
        print(f"Relation representations:")
        print(f"  Shape: {relation_representations.shape} (torch.Size: {relation_representations.shape})")
        print(f"  Dtype: {relation_representations.dtype}")
        print(f"  Sample values: {relation_representations.flatten()[:5].tolist()}")

        print(f"\nNode representations:")
        print(f"  Shape: {node_representations.shape} (torch.Size: {node_representations.shape})")
        print(f"  Dtype: {node_representations.dtype}")
        print(f"  Sample values: {node_representations.flatten()[:5].tolist()}")

        # Print statistics about dataset
        num_relations = train_data.edge_type.unique().numel()
        num_nodes = train_data.num_nodes if hasattr(train_data, 'num_nodes') else train_data.edge_index.max().item() + 1
        num_edges = train_data.edge_index.shape[1]
        vector_dim = rel_model_cfg.input_dim

        print("\n=== Dataset Statistics ===")
        print(f"Unique relations: {num_relations}")
        print(f"Unique nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Chosen vector dimension: {vector_dim}")
        print("=====================================\n")

        print("Explanation of torch.Size dimensions:")
        print("  - For relation representations: (num_relations, vector_dim)")
        print("  - For node representations: (num_nodes, vector_dim)")
        print("  - num_relations: number of unique relation types in the dataset")
        print("  - num_nodes: number of unique entities/nodes in the dataset")
        print("  - vector_dim: dimensionality of the embedding vectors (chosen in config)")
        print("=====================================\n")
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during the forward pass: {e}")
        traceback.print_exc()
        print("\nPlease ensure your model's forward method is compatible with the provided data and batch.")
        print(f"Data type: {type(train_data)}")
        print(f"Batch shape: {batch.shape}, Batch type: {batch.dtype}")