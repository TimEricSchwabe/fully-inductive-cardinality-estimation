import torch
import numpy as np

# Load the model checkpoint
model_path = "/Users/tomiris/PycharmProjects/fully-inductive-cardinality-estimation/output/run_2026-06-12_16-58-10/checkpoints/best_model.pt"
checkpoint = torch.load(model_path, map_location='cpu')

# Extract relation embeddings
embeddings = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()


# Select the first 3 embeddings for the cycle dataset
my_preds = embeddings[0:3]
gram_matrix = np.dot(my_preds, my_preds.T)

# 1. Gram Spectrum (Eigenvalues)
eigenvalues = np.linalg.eigvals(gram_matrix)
print(f"Gram Spectrum (Eigenvalues): {np.sort(eigenvalues)[::-1]}")

# 2. Gram Determinant (Volume)
gram_det = np.linalg.det(gram_matrix)
print(f"Gram Determinant: {gram_det}")