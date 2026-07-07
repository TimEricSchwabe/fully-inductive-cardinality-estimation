#!/bin/bash
set -euo pipefail

# Usage: ./cluster_install.sh /path/to/requirements.txt [cu126|cpu]
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_requirements.txt> [cuda_tag]"
    echo "Example: $0 /path/to/requirements.txt cu126"
    exit 1
fi

REQUIREMENTS_PATH="$1"
CUDA_TAG="${2:-cu126}"   # default to cu126; pass "cpu" for CPU-only wheels

if [ ! -f "$REQUIREMENTS_PATH" ]; then
    echo "Error: Requirements file '$REQUIREMENTS_PATH' not found."
    exit 1
fi

echo "Installing PyTorch and related packages..."
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

echo "Installing PyTorch Geometric binary deps (pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv)..."
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

echo "Installing torch_geometric..."
pip3 install torch_geometric

echo "Installing rest of requirements from: $REQUIREMENTS_PATH"
pip3 install -r "$REQUIREMENTS_PATH"

echo "All done. Installed packages pinned for torch 2.8.0 (CUDA tag: $CUDA_TAG)."