import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
# Select the 7 predicates for the fano dataset
my_preds = embeddings[0:7]
labels = [f'p_{i}' for i in range(7)]

# 2. Apply PCA (Principal Component Analysis)
pca = PCA(n_components=2)
preds_pca = pca.fit_transform(my_preds)

# 3. Plot the PCA graph
plt.figure(figsize=(6, 6))

# Plot the points
plt.scatter(preds_pca[:, 0], preds_pca[:, 1], color='blue', zorder=5)

# Add labels to the points
for i, txt in enumerate(labels):
    plt.annotate(txt, (preds_pca[i, 0], preds_pca[i, 1]), fontsize=12, xytext=(5, 5), textcoords='offset points')

# Display the explained variance percentage on the axes
plt.title("PCA Projection of Fano Predicates")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(True)
# Force the axes to have equal scaling so the shape isn't distorted
plt.axis('equal')

# Save and show the plot
plt.savefig("fano_pca_projection.png")
# plt.show()
