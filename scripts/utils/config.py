
import yaml
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List, Optional, Literal


class DataConfig(BaseSettings):
    dataset: str 
    queries: str
    batch_size: int = 1  # Batch size for training / test / validation
    num_queries: int = -1  # Number of queries for debugging, -1 means all
    type: Literal['train', 'validation', 'test', 'train-val'] = "train-val"  # Type of data split (train/test/val/train-val)

# # TODO: Besser für mehr Übersicht, aber erst einbauen wenn es fertig ist
class EntityGNNConfig(BaseSettings):
    message_func: Literal['gine', 'GAT'] = "gine"
    model: Literal['EntityGNN'] = 'EntityGNN'
    skip_connection: bool = True
    layer_norm: bool = False
    dropout: float = 0.0

class RelationGNNConfig(BaseSettings):
    message_func: Literal['GAT', 'gine'] = "gine"
    model: Literal['RelationGNN', 'BipartiteGraphGNN'] = 'RelationGNN'
    skip_connection: bool = True
    layer_norm: bool = False
    dropout: float = 0.0

class QueryGNNConfig(BaseSettings):
    # message_func: Literal['Triple', 'gine'] = "gine"
    model: Literal['QueryGNN', 'QueryCardinalityGNN','TripleModel', 'SimpleSumGNN', 'QueryTransformer'] = 'TripleModel'
    skip_connection: bool = False
    layer_norm: bool = True
    dropout: float = 0.0

class SchedulerConfig(BaseSettings):
    enabled: bool = True
    min_lr: float = 1e-6


class Config(BaseSettings):
    data: List[DataConfig] # Name of the dataset to use
    epochs: int = 10  # Number of training epochs
    learning_rate: float = 1e-4  # Learning rate for optimizer
    train_ratio: float = 0.8  # Ratio of data used for training
    output_dir: str = "output/"  # Directory for output files
    save_every: int = 5  # Save checkpoint every N epochs
    save_metrics_every: int = 1  # Save metrics CSV files every N epochs
    eval_every: int = 1  # Evaluate model every N epochs
    seed: int = 42  # Random seed for reproducibility
    device: str = "auto"  # Device to use ("cpu", "cuda", or "auto")
    weight_decay: Optional[float] = None  # Optional weight decay for optimizer
    n_embedding_layers: int = 2  # Number of layers in embedding MLP
    GNNNeighborhood: List[int] = [100, 100]  # Neighborhood sizes for entity GNN
    Conditioning: Literal['seed only', 'all', 'false', 'random'] = "false"  # Conditioning strategy for GNN
    min_cardinality: int | None = None  # Minimum cardinality filter
    model: Literal['FICE', 'FiceBinary', 'FiceIndependent'] = "FICE"  # Model type to use
    embedding_dim: int = 128  # Dimension of entity and relation embeddings
    wandb: bool = False  # Whether to use Weights & Biases for logging
    wandb_run_name: Optional[str] = None  # Name of the Weights & Biases run
    laplacian_k: int = 0  # Number of Laplacian eigenvectors for Positional Encoding
    gradient_clipping: float = 0.5  # Maximum norm for gradient clipping

    query_gnn: QueryGNNConfig = QueryGNNConfig()
    entity_gnn: EntityGNNConfig = EntityGNNConfig()
    relation_gnn: RelationGNNConfig = RelationGNNConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    

    # class Config:
    #     extra = "allow"  # Allow extra fields if present


def load_config_yaml(config_path: str) -> Config:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
