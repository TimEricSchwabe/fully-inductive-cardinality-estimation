
import yaml
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pathlib import Path
from typing import List, Optional, Literal


class DataConfig(BaseSettings):
    dataset: str 
    queries: str
    batch_size: int = 1  # Batch size for training / test / validation
    num_queries: int = -1  # Number of queries for debugging, -1 means all
    type: Literal['train', 'validation', 'test', 'train-val'] = "train-val"  # Type of data split (train/test/val/train-val)

class EntityGNNConfig(BaseSettings):
    message_func: Literal['gine', 'GAT', 'deepset', 'gps'] = "gine"
    source: Literal['KG', 'bipartite', 'factor'] = 'KG'
    skip_connection: bool = True
    layer_norm: bool = True
    dropout: float = 0.0

class RelationGNNConfig(BaseSettings):
    message_func: Literal['GAT', 'gine', 'deepset', 'gps'] = "gine"
    source: Literal['bipartite', 'factor'] = 'bipartite'
    skip_connection: bool = True
    layer_norm: bool = True
    dropout: float = 0.0
    relation_attention: bool = False
    relational: bool = False
    role_aware_degrees: bool = True
    # GPS options (only used when message_func == 'gps')
    heads: int = 4  # Attention heads per GPS layer
    attn_type: Literal['multihead', 'performer'] = 'multihead'  # Global attention type
    local_mp: bool = True  # If False, GPS layers use attention only (no local GINE conv)

class QueryGNNConfig(BaseSettings):
    message_func: Literal['gine', 'deepset', 'triple', 'edge_triple'] = "gine"
    model: Literal['QueryGNN', 'QueryCardinalityGNN','TripleModel', 'BoxTripleModel', 'ProductTripleModel', 'SimpleSumGNN', 'QueryTransformer'] = 'TripleModel'
    skip_connection: bool = False
    layer_norm: bool = True
    dropout: float = 0.0
    num_layers: int = 2
    readout: Literal['sum', 'attention'] = 'sum'
    message_activation: Literal['relu', 'silu', 'linear'] = 'relu'
    # Virtual node for TripleModel: a per-graph global node updated between
    # message-passing layers. Gives long-range information flow beyond the
    # num_layers-hop receptive field. False reproduces the legacy model.
    virtual_node: bool = False
    # QueryTransformer options (used only when model == 'QueryTransformer').
    # Defaults match the previously hardcoded prototype (1 GPSConv layer,
    # 4 heads, no Laplacian PE), so old configs build the identical module.
    transformer_layers: int = 1
    transformer_heads: int = 4
    transformer_laplacian_pe_k: int = 0

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
    # Neighborhood sampling mode:
    # 'joint'    — one shared neighborhood sampled around the union of all query seeds in the batch (legacy behavior)
    # 'per_node' — one precalculated, independent neighborhood per seed node, identical at train and test
    neighborhood_mode: Literal['joint', 'per_node'] = 'joint'
    # Monte Carlo neighborhood sampling (per_node mode only):
    # a pool of mc_pool_size neighborhoods is precalculated per node; each forward
    # pass (train AND inference) averages the embeddings of mc_samples of them.
    mc_pool_size: int = 1  # Precalculated neighborhoods per node
    mc_samples: int = 1  # Neighborhoods averaged per forward pass (drawn randomly from pool in training, fixed subset at eval)
    # Neighbor selection for per_node subgraphs (all use GNNNeighborhood caps):
    # 'uniform'    — uniform random neighbors per hop
    # 'ppr'        — top-cap neighbors per hop by personalized PageRank mass (deterministic, mc_pool_size must be 1)
    # 'ppr_sample' — neighbors sampled without replacement ~ PPR mass (stochastic; composes with MC pooling)
    neighbor_selection: Literal['uniform', 'ppr', 'ppr_sample'] = 'uniform'
    ppr_alpha: float = 0.15  # PPR teleport probability
    # Append true (uncapped) structural features to per_node subgraph nodes,
    # computed once from the FULL factor/bipartite graph: log1p degree per
    # role (3, all nodes) + log1p per-relation max fan-out bounds (2, only on
    # relation nodes; GBJO pred_stats: max triples sharing one subject/object)
    degree_features: bool = False
    min_cardinality: int | None = None  # Minimum cardinality filter
    model: Literal['FICE', 'FiceBinary', 'FiceIndependent'] = "FICE"  # Model type to use
    embedding_dim: int = 128  # Dimension of entity and relation embeddings
    wandb: bool = False  # Whether to use Weights & Biases for logging
    wandb_run_name: Optional[str] = None  # Name of the Weights & Biases run
    wandb_run_id: Optional[str] = None  # W&B run ID (auto-generated if None)
    random_walk_pe: int = 0  # Random walk PE walk length (0 = disabled)
    gradient_clipping: float = 0.5  # Maximum norm for gradient clipping
    prediction_mode: Literal['log-cardinality', 'log-selectivity'] = 'log-cardinality'
    embedding_space: Literal['euclidean', 'box', 'gaussian', 'hyperbolic', 'hgcn', 'product'] = 'euclidean'
    count_features: Literal['raw', 'normalized', 'none'] = 'raw'  # How to use occurrence counts in query GNN
    disable_embedding_gnn: bool = False  # If True, skip entity/relation GNN; query GNN sees only count features

    # Ranking loss over sub-BGPs of the same parent (requires a ranking query set)
    ranking_loss: bool = False  # Enable pairwise ranking loss across sub-BGPs per group
    ranking_loss_weight: float = 1.0  # Weight of the ranking term relative to the log-cardinality loss
    ranking_margin: float = 0.5  # Margin (in log-cardinality units), used only when ranking_loss_type == "hinge"
    ranking_group_batch: int = 4  # Parent groups per training step when ranking is enabled
    ranking_loss_type: Literal['hinge', 'ranknet'] = 'hinge'  # Per-pair loss form
    ranking_size_weight_exponent: float = 0.0  # Per-pair weight (size_i*size_j)^(exp/2); 0 disables
    ranking_top_weight_strength: float = 0.0  # Per-pair weight 1/(1 + s*min(rank_i,rank_j)); 0 disables

    # Pointwise regression term. 'smooth_l1' is the legacy default.
    # 'pinball' trains the pinball_tau quantile of log-cardinality: tau > 0.5
    # penalizes underestimation more than overestimation (asymmetry matched to
    # join planning, where underestimates produce the catastrophic plans).
    regression_loss_type: Literal['smooth_l1', 'pinball'] = 'smooth_l1'
    pinball_tau: float = 0.9  # Target quantile when regression_loss_type == 'pinball'

    query_gnn: QueryGNNConfig = QueryGNNConfig()
    entity_gnn: EntityGNNConfig = EntityGNNConfig()
    relation_gnn: RelationGNNConfig = RelationGNNConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    @model_validator(mode='after')
    def validate_regression_loss(self):
        if self.regression_loss_type == 'pinball' and not (0.0 < self.pinball_tau < 1.0):
            raise ValueError(f"pinball_tau must be in (0, 1), got {self.pinball_tau}")
        return self

    @model_validator(mode='after')
    def validate_gnn_sources(self):
        if self.entity_gnn.source != 'KG' and self.entity_gnn.source != self.relation_gnn.source:
            raise ValueError(f"entity source '{self.entity_gnn.source}' must match relation source '{self.relation_gnn.source}'")
        return self

    @model_validator(mode='after')
    def validate_neighborhood_mode(self):
        if self.neighborhood_mode == 'per_node':
            if self.entity_gnn.source == 'KG':
                raise ValueError("neighborhood_mode='per_node' requires entity_gnn.source 'bipartite' or 'factor'")
            if self.relation_gnn.relation_attention:
                raise ValueError("neighborhood_mode='per_node' is incompatible with relation_gnn.relation_attention")
            if self.embedding_space not in ('euclidean', 'box', 'gaussian'):
                raise ValueError("neighborhood_mode='per_node' only supports euclidean/box/gaussian embedding spaces")
        if self.mc_samples > self.mc_pool_size:
            raise ValueError(f"mc_samples ({self.mc_samples}) must be <= mc_pool_size ({self.mc_pool_size})")
        if self.mc_pool_size > 1 and self.neighborhood_mode != 'per_node':
            raise ValueError("mc_pool_size > 1 requires neighborhood_mode='per_node'")
        if self.neighbor_selection != 'uniform' and self.neighborhood_mode != 'per_node':
            raise ValueError(f"neighbor_selection='{self.neighbor_selection}' requires neighborhood_mode='per_node'")
        if self.neighbor_selection == 'ppr' and self.mc_pool_size != 1:
            raise ValueError("neighbor_selection='ppr' is deterministic; mc_pool_size must be 1")
        if self.degree_features and self.neighborhood_mode != 'per_node':
            raise ValueError("degree_features=True requires neighborhood_mode='per_node'")
        return self

    # class Config:
    #     extra = "allow"  # Allow extra fields if present


def load_config_yaml(config_path: str) -> Config:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
