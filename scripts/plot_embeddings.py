import torch
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

sys.path.append('.')
from utils.config import load_config_yaml
from utils.data_helper import make_factor_graph
from utils.models import TrainingContext
from scripts.train import initialize_model_components
from fice.datasets import OrthogonalDataset, TwoClustersDataset

def plot_embeddings_from_checkpoint(checkpoint_path: str, dataset_name: str):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist.")
        return

    # Automatically find matching config.yaml in the parent directory of the checkpoint folder
    run_dir = checkpoint_file.parent.parent
    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        print(f"Error: Config file '{config_file}' not found.")
        return

    print(f"Loading config from: {config_file}")
    config = load_config_yaml(str(config_file))
    device = torch.device('cpu')

    context = TrainingContext(
        config=config, 
        device=device,
        run_dir=Path("dummy"),
        log_dir=Path("dummy"),
        checkpoint_dir=Path("dummy")
    )

    # Initialize model with the correct config parameters (e.g., dimension)
    components = initialize_model_components(context, total_training_steps=1)
    model = components.model

    print(f"Loading model weights from: {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    # Determine dataset
    if dataset_name.lower() == "orthogonal":
        ds = OrthogonalDataset()
    elif dataset_name.lower() in ["two_clusters", "twoclusters"]:
        ds = TwoClustersDataset()
    else:
        print(f"Error: Unknown dataset '{dataset_name}'. Choose 'orthogonal' or 'two_clusters'.")
        return

    name = ds.dataset_name
    print(f"Processing dataset: {name}...")
    
    graph = make_factor_graph(ds).to(device)
    n_ent = graph.original_num_nodes
    n_rel = graph.original_num_relations
    
    with torch.no_grad():
        out_emb = model.embedding_model(graph).squeeze(0)
        rel_embs = out_emb[n_ent:n_ent+n_rel].cpu().numpy()
        
    print(f"Extracted {len(rel_embs)} relation embeddings. Shape: {rel_embs.shape}")

    # Compute Gram Matrix (Cosine Similarity)
    norms = np.linalg.norm(rel_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    norm_embs = rel_embs / norms
    gram_matrix = np.dot(norm_embs, norm_embs.T)

    output_heatmap = run_dir / f"{name}_gram_matrix.png"
    output_pca = run_dir / f"{name}_pca.png"

    # 1. Plot & Save Cosine Similarity Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(gram_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Gram Matrix (Cosine Similarity)\n{name} (Run: {run_dir.name})")
    plt.tight_layout()
    plt.savefig(output_heatmap, dpi=300)
    plt.close()
    print(f"Saved Gram Matrix Heatmap to: {output_heatmap}")

    # 2. Plot & Save PCA Projection (if dimensionality allows)
    dim = rel_embs.shape[1]
    if dim >= 2:
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(rel_embs)
        
        plt.figure(figsize=(6, 5))
        if name == "two_clusters":
            labels = ['Cluster A', 'Cluster A', 'Cluster A', 'Cluster B', 'Cluster B', 'Cluster B']
            colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']
            for i, coord in enumerate(pca_coords):
                plt.scatter(coord[0], coord[1], color=colors[i], label=labels[i] if i in [0, 3] else "", s=100)
                plt.text(coord[0]+0.01, coord[1]+0.01, f"p{i+1}", fontsize=12)
            plt.legend()
        else:
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for i, coord in enumerate(pca_coords):
                plt.scatter(coord[0], coord[1], color=colors[i % len(colors)], s=100)
                plt.text(coord[0]+0.01, coord[1]+0.01, f"p{i+1}", fontsize=12)
                
        plt.title(f"PCA 2D of Relation Embeddings\n{name} (Run: {run_dir.name})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_pca, dpi=300)
        plt.close()
        print(f"Saved 2D PCA Plot to: {output_pca}")
    else:
        print(f"Skipping PCA plot: Embedding dimension is {dim} (requires >= 2).")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 plot_embeddings.py <path_to_model.pt> <dataset_name>")
        sys.exit(1)
        
    plot_embeddings_from_checkpoint(sys.argv[1], sys.argv[2])
