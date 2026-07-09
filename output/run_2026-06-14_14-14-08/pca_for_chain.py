import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the model checkpoint
model_path = "/Users/tomiris/PycharmProjects/fully-inductive-cardinality-estimation/output/run_2026-06-14_14-14-08/checkpoints/best_model.pt"
checkpoint = torch.load(model_path, map_location='cpu')

embeddings = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()

# Select the 5 predicates for the Path dataset
my_preds = embeddings[0:5]
labels = [f'p_{i+1}' for i in range(5)]

# 2. Apply PCA (Principal Component Analysis)
pca = PCA(n_components=2)
preds_pca = pca.fit_transform(my_preds)

# 3. Plot the PCA graph with a connecting line!
plt.figure(figsize=(7, 5))

# Plot the points
plt.scatter(preds_pca[:, 0], preds_pca[:, 1], color='blue', zorder=5)

# PLOT THE TRAJECTORY (this will visually prove the "linear ordering")
plt.plot(preds_pca[:, 0], preds_pca[:, 1], color='gray', linestyle='--', zorder=4)

# Add labels to the points
for i, txt in enumerate(labels):
    plt.annotate(txt, (preds_pca[i, 0], preds_pca[i, 1]), fontsize=12, xytext=(5, 5), textcoords='offset points')

# Display the explained variance percentage on the axes
plt.title("PCA Projection of Path Predicates (Trajectory)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(True)

# Save and show the plot
plt.savefig("path_pca_projection.png")
plt.show()