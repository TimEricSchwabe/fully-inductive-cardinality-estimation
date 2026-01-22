import json
import re
import requests
import tqdm
import matplotlib.pyplot as plt
import numpy as np 


with open('data/fb237_v4_star_queries/raw/59651327?private_link=780ccfbe4856b223d7b5', 'r') as f:
    data = json.load(f)

cardinalities = []

for query in tqdm.tqdm(data):
    cardinalities.append(query['y'])


# Calculate statistics
cardinalities_array = np.array(cardinalities)
mean_cardinality = np.mean(cardinalities_array)
median_cardinality = np.median(cardinalities_array)
percentiles = np.percentile(cardinalities_array, [25, 50, 75, 90, 95, 99])

print(f"\nCardinality Statistics:")
print(f"Mean: {mean_cardinality:.2f}")
print(f"Median: {median_cardinality:.2f}")
print(f"Min: {np.min(cardinalities_array)}")
print(f"Max: {np.max(cardinalities_array)}")
print(f"25th percentile: {percentiles[0]:.2f}")
print(f"75th percentile: {percentiles[2]:.2f}")
print(f"90th percentile: {percentiles[3]:.2f}")
print(f"95th percentile: {percentiles[4]:.2f}")
print(f"99th percentile: {percentiles[5]:.2f}")
print(f"Total queries: {len(cardinalities)}")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Log scale histogram (shows full distribution)
ax1 = axes[0, 0]
ax1.hist(cardinalities, bins=np.logspace(np.log10(max(1, min(cardinalities))), 
                                          np.log10(max(cardinalities)), 50), 
         edgecolor='black', alpha=0.7)
ax1.set_xscale('log')
ax1.set_xlabel('Cardinality (log scale)')
ax1.set_ylabel('Frequency')
ax1.set_title('Full Distribution (Log Scale X-axis)')
ax1.grid(True, alpha=0.3)
ax1.axvline(median_cardinality, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cardinality:.2f}')
ax1.legend()

# Plot 2: Up to 95th percentile (removes extreme outliers)
ax2 = axes[0, 1]
p95 = percentiles[4]
filtered_cards = cardinalities_array[cardinalities_array <= p95]
ax2.hist(filtered_cards, bins=50, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Cardinality')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Distribution up to 95th Percentile (≤{p95:.0f})')
ax2.grid(True, alpha=0.3)
ax2.axvline(mean_cardinality if mean_cardinality <= p95 else median_cardinality, 
            color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cardinality:.2f}')
ax2.axvline(median_cardinality, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cardinality:.2f}')
ax2.legend()

# Plot 3: Log-log scale (good for power-law distributions)
ax3 = axes[1, 0]
counts, bins, _ = ax3.hist(cardinalities, bins=np.logspace(np.log10(max(1, min(cardinalities))), 
                                                             np.log10(max(cardinalities)), 50), 
                            edgecolor='black', alpha=0.7)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Cardinality (log scale)')
ax3.set_ylabel('Frequency (log scale)')
ax3.set_title('Full Distribution (Log-Log Scale)')
ax3.grid(True, alpha=0.3)

# Plot 4: Up to 99th percentile
ax4 = axes[1, 1]
p99 = percentiles[5]
filtered_cards_99 = cardinalities_array[cardinalities_array <= p99]
ax4.hist(filtered_cards_99, bins=50, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Cardinality')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Distribution up to 99th Percentile (≤{p99:.0f})')
ax4.grid(True, alpha=0.3)
ax4.axvline(median_cardinality, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cardinality:.2f}')
ax4.legend()

plt.tight_layout()
plt.show()