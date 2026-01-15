#!/usr/bin/env python3
"""
Training script for NeuSO model.
Executes training using the existing NeuSO/train.py with the configured parameters.
"""

import sys
import argparse
import os
from pathlib import Path
import yaml
from datetime import datetime
from NeuSO.parser_config import ConfigParser
import NeuSO.train as train
        

def main():
    parser = argparse.ArgumentParser(
        description="Train NeuSO model for cardinality estimation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'swdf_star')"
    )
    parser.add_argument(
        "--neuso-root",
        type=Path,
        default=None,
        help="Path to NeuSO root directory (default: ./NeuSO)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test (default: train)"
    )
    
    args = parser.parse_args()
    
    # Determine NeuSO root
    script_dir = Path(__file__).parent
    neuso_root = args.neuso_root if args.neuso_root else script_dir / "NeuSO"
    
    if not neuso_root.exists():
        print(f"Error: NeuSO directory not found at {neuso_root}")
        sys.exit(1)
    
    # Check if dataset exists
    dataset_dir = neuso_root / "dataset" / args.dataset
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        print(f"Run prepare_data.py first to prepare the dataset.")
        sys.exit(1)
    
    # Update config.yaml
    config_path = neuso_root / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update dataset and mode
    config['dataset'] = args.dataset
    config['mode'] = args.mode
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"Training NeuSO Model")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"NeuSO Root: {neuso_root}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
    # Display training configuration
    print("Training Configuration:")
    print(f"  Epochs: {config.get('trainer', {}).get('epochs', 'N/A')}")
    print(f"  Learning Rate: {config.get('optimizer', {}).get('lr', 'N/A')}")
    print(f"  Batch Size: {config.get('data_loader', {}).get('query_batch_size', 'N/A')}")
    print(f"  Train/Test Split: {config.get('trainer', {}).get('train_test_split', 'N/A')}")
    print(f"  Topology Embeddings: {config.get('data_loader', {}).get('topology_emb', 'N/A')}")
    print()
    
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Change to NeuSO directory (required for imports and relative paths)
    original_dir = os.getcwd()
    os.chdir(neuso_root)
    
    # Add NeuSO to Python path
    sys.path.insert(0, str(neuso_root))
    
    try:
        # Import NeuSO's train module and parser
       
        # Use NeuSO's ConfigParser to load configuration
        parser = ConfigParser('config.yaml')
        
        # Run training using NeuSO's main function
        train.main(parser.config, parser.exp_name, checkpoint_dir=parser.model_save_dir)
        
        print(f"\n{'='*60}")
        print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Find the saved model
        saved_dir = neuso_root / "saved"
        if saved_dir.exists():
            model_dirs = sorted(saved_dir.glob(f"{args.dataset}*"), key=lambda x: x.stat().st_mtime)
            if model_dirs:
                latest_model = model_dirs[-1]
                print(f"\nModel saved to: {latest_model}")
                print(f"\nNext step:")
                print(f"  Run inference: python run_estimation.py --dataset {args.dataset} --model-dir {latest_model.name}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"Training interrupted by user")
        print(f"{'='*60}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Error during training: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
