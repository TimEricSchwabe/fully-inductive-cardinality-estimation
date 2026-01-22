
import yaml
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List, Union, Literal


class DataConfig(BaseSettings):
    dataset: str 
    queries: str
    batch_size: int = 1  # Batch size for training
    num_queries: int = -1  # Number of queries for debugging
    type: Literal['train', 'validation', 'test', 'train-val'] = "train-val"  # Type of data split (train/test/val/train-val)

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
    EntityGNNMessageFunc: Literal['rotate', 'distmult', 'transe', 'gine', 'GAT'] = "gine"  # Message function for entity GNN
    RelationGNNMessageFunc: Literal['rotate', 'distmult', 'transe', 'GAT', 'gine'] = "GAT"  # Message function for relation GNN
    Conditioning: Literal['seed only', 'all', 'false'] = "all"  # Conditioning strategy for GNN
    QueryGNN: Literal['QueryGNN', 'QueryCardinalityGNN', 'TripleModel', 'SimpleSumGNN', 'QueryTransformer'] = 'TripleModel' 
    min_cardinality: int | None = None  # Minimum cardinality filter
    model: Literal['FICE', 'FiceBinary'] = "FICE"  # Model type to use
    layer_norm: bool = True  # Whether to use layer normalization in GNNs
    embedding_dim: int = 128  # Dimension of entity and relation embeddings
    n_embedding_layers: int = 2  # Number of layers in embedding MLP
    wandb: bool = False  # Whether to use Weights & Biases for logging
    
    entity_graph: Literal['bipartite', 'KG'] = "KG"
    relation_graph: Literal['bipartite', 'RelationGraph'] = "bipartite"


    class Config:
        extra = "allow"  # Allow extra fields if present


def load_config_yaml(config_path: str) -> Config:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
