from pathlib import Path
import logging
import random
from typing import Union, Optional
import torch
import numpy as np
from datetime import datetime
import wandb
import os
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hashlib
from git import Repo, InvalidGitRepositoryError


logger = logging.getLogger(__name__)


def get_git_info():
    """Get current git branch, commit hash, and dirty status.
    
    Returns:
        tuple: (branch_name, commit_hash, is_dirty)
               Falls back to ("nogit", "unknown", False) if not a git repo
    """
    try:
        repo = Repo(search_parent_directories=True)
        branch = repo.active_branch.name
        commit = repo.head.commit.hexsha[:7]  # Short hash (7 chars)
        is_dirty = repo.is_dirty()
        return branch, commit, is_dirty
    except (InvalidGitRepositoryError, Exception) as e:
        logger.warning(f"Could not get git info: {e}")
        return "nogit", "unknown", False


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

    # Get git information
    branch, commit, is_dirty = get_git_info()
    git_suffix = f"{branch}_{commit}"
    if is_dirty:
        git_suffix += "-dirty"

    # Prepare training config for wandb
    training_config = config.model_dump()
    # Add git metadata to config
    training_config['git_branch'] = branch
    training_config['git_commit'] = commit
    training_config['git_dirty'] = is_dirty


    os.environ["WANDB_DIR"] = str(run_dir)
    logger.info(f"Weights & Biases dir: {run_dir}")
    # Initialize wandb using WANDB_CONFIG defaults
    
    run_name = None
    if config.wandb_run_name is not None:
        run_name = config.wandb_run_name.strip().replace(" ", "_")
        config_hash = hashlib.sha1(str(config.model_dump_json()).encode()).hexdigest()[:6]
        run_name = f"{run_name}_{config_hash}_{git_suffix}"
    else:
        run_name = WANDB_CONFIG['run_name']


    run = wandb.init(   
        project=WANDB_CONFIG['project'],
        entity=WANDB_CONFIG['entity'],
        name=run_name,
        config=training_config,
        tags=WANDB_CONFIG['tags'],
        notes=WANDB_CONFIG['notes'],
        mode=WANDB_CONFIG['mode'],
        dir=str(run_dir)
    )
    
    logger.info(f"Weights & Biases initialized: {run.url}")
    logger.info(f"Git info - Branch: {branch}, Commit: {commit}, Dirty: {is_dirty}")



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

### Helper Functions ###
def analyze_vector_similarity(embeddings, name="Embeddings", sample_size=2000, output_dir: Union[Path, str]=None):
    """
    Analyzes pairwise cosine similarity of a set of vectors.
    
    Args:
        embeddings: Tensor of shape [N, D]
        name: String identifier for plots/logs
        sample_size: Max number of vectors to use (to avoid OOM on large N*N matrices)
        output_dir: Directory to save plots (optional)
    """

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if embeddings.size(0) < 2:
        print(f"[{name}] Not enough vectors to compute similarity.")
        return

    # Detach and move to computation device (can stay on GPU for speed)
    vecs = embeddings.detach()
    
    # Optional: Downsample if too many vectors to avoid creating a massive NxN matrix
    if vecs.size(0) > sample_size:
        perm = torch.randperm(vecs.size(0))
        vecs = vecs[perm[:sample_size]]
    
    # 1. Normalize vectors to unit length (L2 norm)
    # cosine_similarity(a, b) = dot(a, b) / (|a| * |b|)
    # If we normalize first, it becomes just dot(a, b)
    vecs_norm = torch.nn.functional.normalize(vecs, p=2, dim=1)
    
    # 2. Compute Pairwise Similarity Matrix (N x N)
    similarity_matrix = torch.mm(vecs_norm, vecs_norm.t())
    
    # 3. Extract Upper Triangle (off-diagonal)
    # We only care about unique pairs (i, j) where i < j
    triu_indices = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), offset=1)
    similarities = similarity_matrix[triu_indices[0], triu_indices[1]]
    
    # --- Metrics ---
    avg_sim = similarities.mean().item()
    std_sim = similarities.std().item()
    min_sim = similarities.min().item()
    max_sim = similarities.max().item()
    
    print(f"--- {name} Analysis ---")
    print(f"Avg Cosine Similarity: {avg_sim:.4f} ± {std_sim:.4f}")
    print(f"Range: [{min_sim:.4f}, {max_sim:.4f}]")
    
    # --- Plotting ---
    # Move to CPU for plotting
    sim_values = similarities.cpu().numpy()
    
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(sim_values, bins=50, range=(-1, 1), alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(avg_sim, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_sim:.2f}')
        plt.title(f'Pairwise Cosine Similarity Distribution: {name}')
        plt.xlabel('Cosine Similarity (-1 to 1)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure instead of showing to avoid blocking execution
        if output_dir:
            output_path = output_dir / f'similarity_hist_{name}.png'
        else:
            output_path = f'similarity_hist_{name}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved histogram to {output_path}")
    except Exception as e:
        print(f"Could not plot histogram: {e}")


    vecs_np = vecs.cpu().numpy()
    
    # Project to 2D
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs_np)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], alpha=0.6, s=10, color='teal')
    plt.title(f'PCA Projection (2D): {name}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        output_path = output_dir / f'pca_projection_{name}.png'
    else:
        output_path = f'pca_projection_{name}.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PCA plot to {output_path}")

    # t-SNE Projection
    try:
        # Use a smaller perplexity if n_samples is small, typically between 5 and 50
        n_samples = vecs_np.shape[0]
        perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        vecs_tsne = tsne.fit_transform(vecs_np)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(vecs_tsne[:, 0], vecs_tsne[:, 1], alpha=0.6, s=10, color='purple')
        plt.title(f't-SNE Projection (2D): {name}')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            output_path = output_dir / f'tsne_projection_{name}.png'
        else:
            output_path = f'tsne_projection_{name}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Saved t-SNE plot to {output_path}")
    except Exception as e:
        print(f"Could not compute/plot t-SNE: {e}")