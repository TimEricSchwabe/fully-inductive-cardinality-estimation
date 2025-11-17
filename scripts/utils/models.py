import torch
from pathlib import Path
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import List
from .config import Config
from utils.neighborhood_sampler import MultigraphNeighborSampler

@dataclass
class TrainingContext:
    """Container for training configuration and environment."""
    config: Config
    device: torch.device
    run_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    use_wandb: bool = False
    current_epoch: int = 1  # Start from 0, will be incremented to 1 before first epoch


@dataclass
class ModelComponents:
    """Container for model and optimizer."""
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer


@dataclass
class DataComponents:
    """Container for data and data loaders."""
    data: Data
    train_loader: DataLoader
    test_loader: DataLoader
    neighbor_sampler: MultigraphNeighborSampler = None  # Type depends on the sampler implementation

@dataclass
class TrainingState:
    """Container for training state information."""
    best_test_loss: float
    train_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame

    training_context: TrainingContext 
    model_components: ModelComponents 
    data_components: DataComponents


