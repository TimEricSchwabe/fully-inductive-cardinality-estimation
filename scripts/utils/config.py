
import yaml
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class DataConfig(BaseSettings):
    dataset: str 
    queries: str
    batch_size: int = 32  # Batch size for training
    num_queries: int = -1  # Number of queries for debugging

class Config(BaseSettings):
    data: List[DataConfig] # Name of the dataset to use
    epochs: int = 10  # Number of training epochs
    learning_rate: float = 1e-4  # Learning rate for optimizer
    train_ratio: float = 0.8  # Ratio of data used for training
    output_dir: str = "output/"  # Directory for output files
    save_every: int = 5  # Save checkpoint every N epochs
    eval_every: int = 1  # Evaluate model every N epochs
    seed: int = 42  # Random seed for reproducibility
    device: str = "auto"  # Device to use ("cpu", "cuda", or "auto")
    weight_decay: float | None = None  # Optional weight decay for optimizer
    EntityGNNNeighborhood: List[int] = [100, 100]  # Neighborhood sizes for entity GNN
    EntityGNNMessageFunc: str = "gine"  # Message function for entity GNN
    RelationGNNMessageFunc: str = "GAT"  # Message function for relation GNN
    Conditioning: str = "all"  # Conditioning strategy for GNN
    min_cardinality: int | None = None  # Minimum cardinality filter
    model: str = "FICE"  # Model type to use
    layer_norm: bool = True  # Whether to use layer normalization in GNNs

    class Config:
        extra = "allow"  # Allow extra fields if present


def load_config_yaml(config_path: str) -> Config:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)