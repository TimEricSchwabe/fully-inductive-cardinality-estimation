import torch
import yaml
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

sys.path.append('.')
from utils.config import load_config_yaml, Config
from utils.data_helper import prepare_data, make_factor_graph, make_bipartit_graph
from fice.models import FICE
from utils.models import TrainingContext
from scripts.train import initialize_model_components
from pathlib import Path

def main():
    config = load_config_yaml("configs/train_custom.yaml")
    device = torch.device('cpu')

    context = TrainingContext(
        config=config, 
        device=device,
        run_dir=Path("dummy"),
        log_dir=Path("dummy"),
        checkpoint_dir=Path("dummy")
    )
    from fice.datasets import OrthogonalDataset, TwoClustersDataset
    datasets = [OrthogonalDataset(), TwoClustersDataset()]

    components = initialize_model_components(context, total_training_steps=1)
    model = components.model

    # Use the original successful run
    latest_run = "run_2026-06-09_20-27-34"
    
    state_dict = torch.load(f"output/{latest_run}/checkpoints/final_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])

    model.eval()

    for ds in datasets:
        name = ds.dataset_name
        print(f"\nProcessing {name}...")
        
        graph = make_factor_graph(ds).to(device)
        n_ent = graph.original_num_nodes
        n_rel = graph.original_num_relations
        
        with torch.no_grad():
            out_emb = model.embedding_model(graph)
            out_emb = out_emb.squeeze(0)
            rel_embs = out_emb[n_ent:n_ent+n_rel].cpu().numpy()
            
        print(f"Extracted {len(rel_embs)} relation embeddings for {name}.")
            
        # Calculate Gram matrix (Cosine Similarity)
        norms = np.linalg.norm(rel_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        norm_embs = rel_embs / norms
        gram_matrix = np.dot(norm_embs, norm_embs.T)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(gram_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title(f"Gram Matrix (Cosine Similarity)\n{name}")
        plt.tight_layout()
        plt.savefig(f"{name}_gram_matrix.png")
        plt.close()
        
        if name == "orthogonal":
            print(f"Orthogonal Gram Matrix:\n{gram_matrix}")
        
        if name == "two_clusters":
            eigenvalues = np.linalg.eigvalsh(gram_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1] # descending
            print(f"Two Clusters Eigenvalues:\n{eigenvalues}")
            
            # PCA 2D
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(rel_embs)
            
            plt.figure(figsize=(6, 5))
            labels = ['Cluster A', 'Cluster A', 'Cluster A', 'Cluster B', 'Cluster B', 'Cluster B']
            colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']
            
            for i, coord in enumerate(pca_coords):
                plt.scatter(coord[0], coord[1], color=colors[i], label=labels[i] if i in [0, 3] else "", s=100)
                plt.text(coord[0]+0.01, coord[1]+0.01, f"p{i+1}", fontsize=12)
                
            plt.title("PCA 2D of Relation Embeddings (Two Clusters)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            plt.grid(True)
            plt.savefig("two_clusters_pca.png")
            plt.close()

if __name__ == "__main__":
    main()
