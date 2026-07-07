#!/bin/bash
set -e

# Run Orthogonal dataset for transductive training

DIMS=(128 16 8 4)

for d in "${DIMS[@]}"; do
    echo "========================================"
    echo "Training orthogonal with dim $d"
    echo "========================================"
    uv run --extra pyg-extras python3 scripts/train.py --config configs/train_custom.yaml \
        --override "data=[{'dataset': 'orthogonal', 'queries': 'orthogonal_queries', 'batch_size': 64, 'num_queries': 20000, 'type': 'train-val'}]" \
        --override "embedding_dim=$d" \
        --override "min_cardinality=1" \
        --override "wandb_run_name=transductive_orthogonal_d${d}"
done
