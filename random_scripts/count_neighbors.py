import sys
from collections import Counter
import matplotlib.pyplot as plt

file_path = "data/wikidata/raw/wikidata.ttl"

object_counts = Counter()
relation_counts = Counter()

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.rstrip(' .').split()
        
        if len(parts) >= 3:
            object_uri = parts[2]
            object_counts[object_uri] += 1
            relation_uri = parts[1]
            relation_counts[relation_uri] += 1

# print total number of objects
print(f"Total number of objects: {len(object_counts)}")
print(f"Total number of relations: {len(relation_counts)}")

# Print top 100 objects
print(f"\nTop 100 objects by occurrence count:")
print("=" * 80)
for rank, (obj, count) in enumerate(object_counts.most_common(100), 1):
    print(f"{rank:3d}. {obj:<60} {count:>10,}")

# Create histogram
counts = list(object_counts.values())
plt.figure(figsize=(12, 6))
plt.hist(counts, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Occurrences')
plt.ylabel('Number of Objects')
plt.title('Histogram of Object Occurrences in RDF File')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
