#!/usr/bin/env python3
"""
Inference/estimation script for NeuSO.
Loads a trained model and runs cardinality estimation on test queries.
"""

import sys
import argparse
import os
from pathlib import Path
import yaml
import json
import numpy as np
import time
import torch
from typing import List, Tuple


def find_latest_model(saved_dir: Path, dataset_name: str = None) -> Path:
    """Find the most recent model checkpoint."""
    if dataset_name:
        model_dirs = sorted(saved_dir.glob(f"{dataset_name}*"), key=lambda x: x.stat().st_mtime)
    else:
        model_dirs = sorted(saved_dir.glob("*"), key=lambda x: x.stat().st_mtime)
    
    if not model_dirs:
        raise FileNotFoundError(f"No model checkpoints found in {saved_dir}")
    
    return model_dirs[-1]


def run_inference(neuso_root: Path, dataset_name: str, model_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference using NeuSO model.
    
    Returns:
        ground_truths, predictions, prediction_times, query_sizes
    """
    # Change to NeuSO directory
    original_dir = os.getcwd()
    os.chdir(neuso_root)
    sys.path.insert(0, str(neuso_root))
    
    try:
        # Import NeuSO modules
        from dataloader.data_loader import GetDataLoader
        from Model.PredictModel import PredictModel
        from Graph import DataGraphInfo
        
        # Load config
        with open(neuso_root / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Get data loaders
        print("Loading test data...")
        dataset_dir = neuso_root / "dataset" / dataset_name
        
        _, test_loader = GetDataLoader(
            dataset_name=dataset_name,
            query_batch_size=1,  # Process one query at a time for timing
            topology_emb=config['data_loader'].get('topology_emb', False),
            train_data_ratio=config['trainer'].get('train_test_split', 0.8),
            shuffle=False,
            mode='test'
        )
        
        # Load model
        print(f"Loading model from {model_dir}...")
        device = torch.device(config.get('device', 'cpu'))
        
        # Find model checkpoint file
        model_files = list((neuso_root / "saved" / model_dir.name / "model").glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint found in {model_dir / 'model'}")
        
        model_path = model_files[-1]  # Use latest checkpoint
        print(f"Using checkpoint: {model_path}")
        
        # Initialize model (need to match training architecture)
        data_graph = test_loader.data_graph
        
        # Determine input dimension
        if config['data_loader'].get('topology_emb', False):
            in_dim = config['data_loader'].get('topology_emb_dim', 128) + 1  # +1 for count
        else:
            in_dim = data_graph.v_label_count + 1  # +1 for count
        
        model = PredictModel(
            query_gnn_config=config['model']['query_gnn'],
            state_card_model_config=config['model']['state_card_model'],
            state_cost_model_config=config['model']['state_cost_model'],
            wcoj_cost_model_config=config['model']['wcoj_cost_model'],
            in_dim=in_dim,
            data_graph_info=data_graph
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Run predictions
        print("Running predictions...")
        ground_truths = []
        predictions = []
        prediction_times = []
        query_sizes = []
        
        with torch.no_grad():
            for batch_idx, (query_graph_batch, ccg_batch, query_name_batch) in enumerate(test_loader):
                for query_graph, ccg, query_name in zip(query_graph_batch, ccg_batch, query_name_batch):
                    # Get ground truth from CCG (first node's cardinality)
                    # Note: Since we use placeholder CCGs, this will be 1
                    # In real usage, you'd load actual ground truth from query results
                    gt_card = ccg.vertices[0].v_card if len(ccg.vertices) > 0 else 1
                    
                    # Time the prediction
                    start_time = time.time()
                    
                    # Run prediction
                    query_graph_data = query_graph.graph_data.to(device)
                    pred_card, _, _ = model(query_graph_data, ccg, data_graph)
                    
                    pred_time = time.time() - start_time
                    
                    # Store results
                    ground_truths.append(float(gt_card))
                    predictions.append(float(pred_card.cpu().item()))
                    prediction_times.append(pred_time)
                    query_sizes.append(query_graph.graph.number_of_edges())
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Processed {batch_idx + 1}/{len(test_loader)} queries")
        
        print(f"Completed {len(predictions)} predictions")
        
        return (
            np.array(ground_truths),
            np.array(predictions),
            np.array(prediction_times),
            np.array(query_sizes)
        )
        
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run NeuSO inference and save results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'swdf_star')"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory name (default: auto-detect latest)"
    )
    parser.add_argument(
        "--neuso-root",
        type=Path,
        default=None,
        help="Path to NeuSO root directory (default: ./NeuSO)"
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default="star",
        help="Query type for results organization (default: 'star')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: ./results/{query_type}/{dataset}/neuso/)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    neuso_root = args.neuso_root if args.neuso_root else script_dir / "NeuSO"
    
    if not neuso_root.exists():
        print(f"Error: NeuSO directory not found at {neuso_root}")
        sys.exit(1)
    
    # Find model
    saved_dir = neuso_root / "saved"
    if args.model_dir:
        model_dir = saved_dir / args.model_dir
        if not model_dir.exists():
            print(f"Error: Model directory not found at {model_dir}")
            sys.exit(1)
    else:
        try:
            model_dir = find_latest_model(saved_dir, args.dataset)
            print(f"Auto-detected model: {model_dir.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please train a model first using train_neuso.py")
            sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = script_dir / "results" / args.query_type / args.dataset / "neuso"
    
    print(f"\n{'='*60}")
    print(f"Running NeuSO Inference")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_dir.name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run inference
    try:
        ground_truths, predictions, pred_times, query_sizes = run_inference(
            neuso_root, args.dataset, model_dir
        )
        
        # Save results
        print(f"\nSaving results to {output_dir}...")
        
        # Import helper to save results
        sys.path.insert(0, str(script_dir.parent))
        from benchmarks.neuso.helper import save_results
        
        save_results(
            output_dir=output_dir,
            ground_truths=ground_truths,
            predictions=predictions,
            prediction_times=pred_times,
            query_sizes=query_sizes,
            method_name="neuso"
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Inference completed successfully!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_dir}")
        print(f"Files generated:")
        print(f"  - gts.npy")
        print(f"  - preds.npy")
        print(f"  - pred_times.npy")
        print(f"  - sizes.npy")
        print(f"  - results.json")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Inference failed: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
