import torch
import numpy as np

# Load the model checkpoint
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pt)")
args = parser.parse_args()
model_path = args.model_path
checkpoint = torch.load(model_path, map_location='cpu')

# Extract relation embeddings
embeddings = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()

# Select the embeddings for the two_clusters dataset
my_preds = embeddings[0:6]
gram_matrix = np.dot(my_preds, my_preds.T)

# 1. Gram Spectrum (Eigenvalues)
eigenvalues = np.linalg.eigvals(gram_matrix)
print(f"Gram Spectrum (Eigenvalues): {np.sort(eigenvalues)[::-1]}")

# 2. Gram Determinant (Volume)
gram_det = np.linalg.det(gram_matrix)
print(f"Gram Determinant: {gram_det}")
