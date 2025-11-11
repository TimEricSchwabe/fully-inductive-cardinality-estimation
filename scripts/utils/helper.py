from pathlib import Path
import logging
import random
import torch
import numpy as np
from datetime import datetime
import wandb
import os
import argparse
import json

logger = logging.getLogger(__name__)


# Weights & Biases configuration
WANDB_CONFIG = {
    'project': 'fice',
    'entity': "fice",  # Set to your wandb username/team or leave None for default
    'run_name': None,  # Will be auto-generated if None
    'tags': [],
    'notes': '',
    'mode': 'online',  # 'online', 'offline', or 'disabled'
}


def initialize_wandb(config, run_dir):
    """Initialize Weights & Biases logging if enabled."""
    # Check if wandb is enabled: command-line flag takes precedence over WANDB_CONFIG

    # Prepare training config for wandb
    training_config = config.copy()

    os.environ["WANDB_DIR"] = str(run_dir)
    logger.info(f"Weights & Biases dir: {run_dir}")
    # Initialize wandb using WANDB_CONFIG defaults
    run = wandb.init(   
        project=WANDB_CONFIG['project'],
        entity=WANDB_CONFIG['entity'],
        name=WANDB_CONFIG['run_name'],
        config=training_config,
        tags=WANDB_CONFIG['tags'],
        notes=WANDB_CONFIG['notes'],
        mode=WANDB_CONFIG['mode'],
        dir=str(run_dir)
    )
    
    logger.info(f"Weights & Biases initialized: {run.url}")



def parse_arguments():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train FICE model for cardinality estimation"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        help="Path to configuration JSON file containing all training parameters",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint to resume training from.",
    )
    # Weights & Biases flag
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    args = parser.parse_args()
    # Enforce --config is required unless --checkpoint is specified
    #if not args.checkpoint and not args.config:
    #    parser.error("--config is mandatory unless --checkpoint is specified")
    return args

# Utility functions
def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.propagate = True

    # IMPORTANT: avoid duplicate handlers if called again
    if root.hasHandlers():
        root.handlers.clear()

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
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def process_seed(seed_value):
    """
    Process seed value from config or command line arguments.
    
    Args:
        seed_value: Can be int, str, or None
        
    Returns:
        int: The processed seed value
        
    Logic:
        - If seed_value is "random" or "Random": generate a random seed
        - If seed_value is a string representing an integer: convert to int
        - If seed_value is None or "None": generate a random seed
        - If seed_value is already an int: use as is
    """
    if seed_value is None or seed_value == "None":
        # Generate random seed
        processed_seed = random.randint(0, 2**32 - 1)
        logger.info(f"No seed specified, using random seed: {processed_seed}")
        return processed_seed
    
    if isinstance(seed_value, str):
        if seed_value.lower() == "random":
            # Generate random seed
            processed_seed = random.randint(0, 2**32 - 1)
            logger.info(f"Random seed requested, using: {processed_seed}")
            return processed_seed
        else:
            try:
                # Try to convert string to integer
                processed_seed = int(seed_value)
                logger.info(f"Using seed from string: {processed_seed}")
                return processed_seed
            except ValueError:
                logger.warning(f"Invalid seed value '{seed_value}', using random seed instead")
                processed_seed = random.randint(0, 2**32 - 1)
                logger.info(f"Using random seed: {processed_seed}")
                return processed_seed
    
    if isinstance(seed_value, int):
        logger.info(f"Using specified seed: {seed_value}")
        return seed_value
    
    # Fallback to random seed
    processed_seed = random.randint(0, 2**32 - 1)
    logger.warning(f"Unexpected seed type '{type(seed_value)}', using random seed: {processed_seed}")
    return processed_seed


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
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