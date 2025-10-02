from pathlib import Path
import logging
import random
import torch
import numpy as np
from datetime import datetime

import argparse
import json

def parse_arguments():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train FICE model for cardinality estimation"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration JSON file containing all training parameters",
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        type=float,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--train_ratio",
        "--ratio",
        "-r",
        type=float,
        help="Ratio of training data to use.",
    )
    
    # Data and paths
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to the data root directory.",
    )
    parser.add_argument(
        "--queries_root",
        type=str,
        help="Path to the queries root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results and checkpoints.",
    )
    
    # Debug and evaluation
    parser.add_argument(
        "--DEBUG_num_queries",
        type=int,
        help="Number of queries to use for debugging. -1 means all queries.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        help="Save model checkpoint every N epochs.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        help="Evaluate model every N epochs.",
    )
    
    # System
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (auto, cpu, cuda, mps).",
    )
    
    # Weights & Biases flag
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    return parser.parse_args()

def merge_config_with_args(config, args):
    """Merge command-line arguments with config, giving priority to command-line args."""
    # Create a copy of the config to avoid modifying the original
    merged_config = config.copy()
    
    # Get all argument names and their values
    args_dict = vars(args)
    
    # Override config values with non-None command-line arguments
    overrides = []
    additions = []
    
    for arg_name, arg_value in args_dict.items():
        # TODO: Change if more complex types are added, not just config
        if arg_name != 'config' and arg_value is not None:
            if arg_name in merged_config:
                overrides.append(f"  {arg_name}: {merged_config[arg_name]} -> {arg_value}")
            else:
                additions.append(f"  {arg_name}: {arg_value}")
            merged_config[arg_name] = arg_value
   
    return merged_config
# Utility functions
def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('fice_training')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"training_{timestamp}.log")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)