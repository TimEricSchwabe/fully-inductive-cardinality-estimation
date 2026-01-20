#!/bin/bash
# Build G-Care Apptainer container
# This script should be run on the cluster

set -e

# Variables
CONTAINER_DIR="/storage/home/lukasketzer"
CONTAINER_NAME="gcare.sif"
DEF_FILE="gcare.def"

echo "================================"
echo "Building G-Care Apptainer Container"
echo "================================"
echo ""

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file $DEF_FILE not found"
    exit 1
fi

# Build the container
echo "Building container image..."
echo "Output: $CONTAINER_DIR/$CONTAINER_NAME"
echo ""

apptainer build "$CONTAINER_DIR/$CONTAINER_NAME" "$DEF_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✓ Container built successfully!"
    echo "================================"
    echo "Location: $CONTAINER_DIR/$CONTAINER_NAME"
    echo ""
    echo "Test the container with:"
    echo "  apptainer exec $CONTAINER_DIR/$CONTAINER_NAME python3 --version"
else
    echo ""
    echo "ERROR: Container build failed"
    exit 1
fi
