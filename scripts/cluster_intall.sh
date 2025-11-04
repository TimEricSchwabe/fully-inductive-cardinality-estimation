#!/bin/bash

# Check if requirements.txt path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_requirements.txt>"
    echo "Example: $0 /path/to/requirements.txt"
    exit 1
fi

REQUIREMENTS_PATH="$1"

# Check if the requirements file exists
if [ ! -f "$REQUIREMENTS_PATH" ]; then
    echo "Error: Requirements file '$REQUIREMENTS_PATH' not found."
    exit 1
fi

echo "Installing PyTorch and related packages..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "Installing PyTorch Geometric and related packages..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
pip install torch_geometric

echo "Installing requirements from: $REQUIREMENTS_PATH"
pip install -r "$REQUIREMENTS_PATH"

echo "Installation completed successfully!"