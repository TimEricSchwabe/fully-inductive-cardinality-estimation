#!/bin/bash
set -e

# Run Two Clusters dataset for transductive training

DIMS=(128 16 8 4)

for d in "${DIMS[@]}"; do
    echo "========================================"
    echo "Training two_clusters with dim $d"
    echo "========================================"
    uv run --extra pyg-extras python3 scripts/train.py --config configs/train_custom.yaml \
        --override "data=[{'dataset': 'two_clusters', 'queries': 'two_clusters_queries', 'batch_size': 64, 'num_queries': 20000, 'type': 'train-val'}]" \
        --override "embedding_dim=$d" \
        --override "min_cardinality=1" \
        --override "wandb_run_name=transductive_two_clusters_d${d}"
done
