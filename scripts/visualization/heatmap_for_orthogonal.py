import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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

# Select the embeddings for the orthogonal dataset
my_preds = embeddings[0:5]
labels = [f'p_{i+1}' for i in range(5)]

# 1. Heatmap (Cosine Similarity)
plt.figure(figsize=(6, 5))
sns.heatmap(cosine_similarity(my_preds), annot=True, cmap='coolwarm',
            xticklabels=labels,
            yticklabels=labels)
plt.title("Orthogonal Predicate Similarity Heatmap")
plt.savefig("predicate_heatmap_for_orthogonal.png")
plt.close()

# 2. 2D Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(my_preds[:, 0], my_preds[:, 1], color='blue')
for i, txt in enumerate(labels):
    plt.annotate(txt, (my_preds[i, 0], my_preds[i, 1]), fontsize=12, xytext=(5, 5), textcoords='offset points')
plt.title("Orthogonal Predicate Geometry in 2D Space")
plt.grid(True)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("predicate_geometry_2d_for_orthogonal.png")
# plt.show()

print("Plots saved: predicate_heatmap_for_orthogonal.png and predicate_geometry_2d_for_orthogonal.png")
