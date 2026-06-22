import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Load the model checkpoint
model_path = "/Users/tomiris/PycharmProjects/fully-inductive-cardinality-estimation/output/run_2026-06-12_16-58-10/checkpoints/best_model.pt"
checkpoint = torch.load(model_path, map_location='cpu')

# Extract relation embeddings
embeddings = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()
# Select the 3 predicates for the Cycle dataset
my_preds = embeddings[0:3]
labels = [f'p_{i+1}' for i in range(3)]

# 2. Apply PCA (Principal Component Analysis)
pca = PCA(n_components=2)
preds_pca = pca.fit_transform(my_preds)

# 3. Plot the PCA graph with a closed loop!
plt.figure(figsize=(6, 6))

# Plot the points
plt.scatter(preds_pca[:, 0], preds_pca[:, 1], color='blue', zorder=5)

# PLOT THE CLOSED LOOP TRAJECTORY
# We stack the first point at the end of the array to draw the line back to p_1
loop_pca = np.vstack((preds_pca, preds_pca[0]))
plt.plot(loop_pca[:, 0], loop_pca[:, 1], color='gray', linestyle='--', zorder=4)

# Add labels to the points
for i, txt in enumerate(labels):
    plt.annotate(txt, (preds_pca[i, 0], preds_pca[i, 1]), fontsize=12, xytext=(5, 5), textcoords='offset points')

# Display the explained variance percentage on the axes
plt.title("PCA Projection of Cycle Predicates (Closed Loop)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(True)
# Force the axes to have equal scaling so the shape isn't distorted
plt.axis('equal')

# Save and show the plot
plt.savefig("cycle_pca_projection.png")
plt.show()