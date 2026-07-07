import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append('.')
from utils.config import load_config_yaml
from utils.data_helper import make_factor_graph
from utils.models import TrainingContext
from scripts.train import initialize_model_components
from fice.datasets import OrthogonalDataset, TwoClustersDataset

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
    
    datasets = [OrthogonalDataset(), TwoClustersDataset()]
    components = initialize_model_components(context, total_training_steps=1)
    model = components.model

    latest_run = "run_2026-06-09_20-27-34"
    model_path = f"output/{latest_run}/checkpoints/final_model.pt"
    
    print(f"Loading checkpoint from: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    print("\n" + "="*50)
    print("           GRAM MATRIX DIAGNOSTICS")
    print("="*50)

    for ds in datasets:
        name = ds.dataset_name
        graph = make_factor_graph(ds).to(device)
        n_ent = graph.original_num_nodes
        n_rel = graph.original_num_relations
        
        with torch.no_grad():
            out_emb = model.embedding_model(graph)
            out_emb = out_emb.squeeze(0)
            rel_embs = out_emb[n_ent:n_ent+n_rel].cpu().numpy()
            
        # L2 normalize relation embeddings
        norms = np.linalg.norm(rel_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        norm_embs = rel_embs / norms
        
        # Compute Gram Matrix
        gram_matrix = np.dot(norm_embs, norm_embs.T)
        
        # Compute Eigenvalues and Determinant
        eigenvalues = np.linalg.eigvalsh(gram_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1] # descending
        determinant = np.linalg.det(gram_matrix)
        
        print(f"\n[ Dataset: {name.upper()} ]")
        print(f"  Relation Embeddings Count: {len(rel_embs)}")
        print(f"  Embedding Shape:           {rel_embs.shape}")
        print("\n  Gram Matrix (Cosine Similarities):")
        print(np.array2string(gram_matrix, precision=4, suppress_small=True))
        
        print("\n  Gram Spectrum (Eigenvalues in descending order):")
        print(np.array2string(eigenvalues, precision=6, suppress_small=True))
        
        print(f"\n  Gram Determinant (Spanned Volume):")
        print(f"    det(G) = {determinant:.6e}")
        
        # Calculate effective rank (eigenvalues > 1e-5)
        effective_rank = np.sum(eigenvalues > 1e-5)
        print(f"  Effective Subspace Rank:   {effective_rank}D")
        print("-"*50)

if __name__ == "__main__":
    main()
