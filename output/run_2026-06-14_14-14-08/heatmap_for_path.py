import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the model checkpoint
model_path = "/Users/tomiris/PycharmProjects/fully-inductive-cardinality-estimation/output/run_2026-06-14_14-14-08/checkpoints/best_model.pt"
checkpoint = torch.load(model_path, map_location='cpu')

# Extract relation embeddings
embeddings = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()

# Select the first 5 embeddings (p_1 to p_5)
my_preds = embeddings[0:5]

# 1. Heatmap (Cosine Similarity)
plt.figure(figsize=(8, 6))
sns.heatmap(cosine_similarity(my_preds), annot=True, cmap='coolwarm',
            xticklabels=[f'p_{i+1}' for i in range(5)],
            yticklabels=[f'p_{i+1}' for i in range(5)])
plt.title("Predicate Similarity Heatmap")
plt.savefig("predicate_heatmap_for_path.png")
plt.close()

# 2. 2D Scatter Plot (Geometric analysis)
plt.figure(figsize=(6, 6))
plt.scatter(my_preds[:, 0], my_preds[:, 1], color='blue')
for i, txt in enumerate([f'p_{i+1}' for i in range(5)]):
    plt.annotate(txt, (my_preds[i, 0], my_preds[i, 1]), fontsize=12, xytext=(5, 5), textcoords='offset points')
plt.title("Predicate Geometry in 2D Space")
plt.grid(True)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("predicate_geometry_2d_for_path.png")
plt.show()

print("Plots saved: predicate_heatmap_for_path.png and predicate_geometry_2d_for_path.png")