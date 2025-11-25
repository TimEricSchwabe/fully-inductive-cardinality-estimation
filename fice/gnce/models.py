import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, RGCNConv, RGATConv, HEATConv, GATConv, GINEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch import Tensor
from torch_geometric.nn.inits import reset
import torch.nn as nn

class QueryGNN(torch.nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.0, train_eps: bool =False):
        super().__init__()

        self.dropout = dropout


        def gine_mlp():
            # The MLP used inside each GINEConv (applied after aggregation)
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.layers = nn.ModuleList([
            GINEConv(gine_mlp(), edge_dim=hidden_dim, train_eps=train_eps),
            GINEConv(gine_mlp(), edge_dim=hidden_dim, train_eps=train_eps),
            GINEConv(gine_mlp(), edge_dim=hidden_dim, train_eps=train_eps),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ])

        # 2-layer MLP head to a single scalar (linear activation)
        # Input dimension is hidden_dim + 1 to account for graph_size concatenation
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch, graph_size):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        for layer, bn in zip(self.layers, self.bns):
            if edge_attr is not None:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global add pooling over nodes -> graph embedding
        # TODO: Funktioniert, aber kein ahnung ob es das macht, was es machen soll
        g = global_add_pool(x, batch.batch)


        # TODO: Adding the graph size |E| to the final MLP
        if graph_size.dim() == 1 and graph_size.size(0) == 1:
            graph_size_expanded = graph_size.expand(g.size(0))
        else:
            graph_size_expanded = graph_size
        
        # Concatenate along feature dimension
        g_with_size = torch.cat([g, graph_size_expanded.unsqueeze(-1)], dim=-1)


        out = self.head(g_with_size)     # [batch_size, 1]
        return torch.abs(out.squeeze(-1))           # [batch_size]




class TripleConv(MessagePassing):
    r"""
    Message passing function for directional message passing
    based on the GINE Conv operator
    Equation:
        .. math::
             x_i^{(k)} = h_\theta^{(k)}  \biggl( x_i^{(k-1)} \ +& \sum_{j \in \mathcal{N}^+(i)}
            \mathrm{ReLU}(x_i^{(k-1)}||e^{j,i}||x_j^{(k-1)}) \ +\\
            & \sum_{j \in \mathcal{N}^-(i)}
            \mathrm{ReLU}(x_j^{(k-1)}||e^{i,j}||x_i^{(k-1)}) \biggr)


    The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps

        # Whether to switch the input to the MLP based on the directionality
        self.DIRECTIONAL = False

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            # Use the provided edge_dim as the edge feature dimensionality.
            # Do not override with a hard-coded value here — the dataset
            # determines the true feature size (e.g. feat_dim + 3).
            self.lin = Linear(3 * edge_dim, edge_dim)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:


        # Switch i and j based on direction of triple:
        reverse = edge_attr[:, -1] == -1

        if self.DIRECTIONAL:
            x_i[reverse], x_j[reverse] = x_j[reverse], x_i[reverse]

        #x_i[edge_attr[:, -1] == -1] = x_j[edge_attr[:, -1] == -1]
        # edge_attr = edge_attr[:, :-1]


        return self.lin(torch.cat((x_i, edge_attr, x_j), 1)).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class TripleModel(torch.nn.Module):
    """
    GNN model to predict cardinality of a query,
    given the query graph and embeddings of nodes
    and edges.

    Args:


    """
    def __init__(self, feat_dim: int = 101, layer_norm=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.layer_norm = layer_norm


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim, self.feat_dim),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feat_dim, self.feat_dim),
            # torch.nn.Dropout(p=0.2)
        )

        # second MLP maps feat_dim -> 200 (via 100)
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim, 100),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            # torch.nn.Dropout(p=0.2)
        )

        self.ln1 = nn.LayerNorm(self.feat_dim)
        self.ln2 = nn.LayerNorm(200)

        # Use feat_dim as edge_dim so TripleConv knows the true edge_attr size
        #self.conv1 = TripleConv(nn=self.mlp, edge_dim=self.feat_dim)
        #self.conv2 = TripleConv(nn=self.mlp2, edge_dim=self.feat_dim)
        self.conv1 = GINEConv(nn=self.mlp, edge_dim=self.feat_dim)
        self.conv2 = GINEConv(nn=self.mlp2, edge_dim=self.feat_dim)
        self.lin = Linear(200, 50)
        self.lin2 = Linear(50, 1)

        # self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, batch, graph_size):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr



        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        if self.layer_norm:
            x = self.ln1(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        if self.layer_norm:
            x = self.ln2(x)
        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.abs(x.squeeze(-1))



class NeuralSelectivity(nn.Module):
    """
    Implements S_theta: Maps aggregated embeddings to a scalar log-selectivity.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, 1), 
        )

    def forward(self, x):
        return self.net(x)



class CardinalityConv(MessagePassing):
    """
    Implements the Generalized Neural Cardinality Operator (Psi).
    Handles specific slicing for the first layer where inputs have 3 count dimensions.

    The model expects log values for all counts ! And log(N) for all variable counts !
    """
    def __init__(self, emb_dim, is_first_layer=False):
        super().__init__(aggr='mean', flow='target_to_source') 
        self.emb_dim = emb_dim
        self.is_first_layer = is_first_layer

        # S_theta for Triple Pattern Join (Message)
        self.sel_triple = NeuralSelectivity(emb_dim) 
        
        self.emb_update_triple = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.SiLU(), 
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            # torch.nn.Dropout(p=0.2)
        )
        #self.emb_update_triple = nn.Linear(emb_dim, emb_dim)

        # S_theta for Node Join (Update)
        self.sel_node = NeuralSelectivity(emb_dim)
        self.emb_update_node = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.SiLU(), 
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
        #self.emb_update_node = nn.Linear(emb_dim, emb_dim)

    def forward(self, x_curr, edge_index, edge_attr, log_N):
        self.log_N = log_N
        # Propagate generates the messages (m_j->i) and sums them.
        aggr_out = self.propagate(edge_index, x=x_curr, edge_attr=edge_attr)
        

        # Split Aggregated Message in cardinality estimate and semantic embeddings
        c_agg = aggr_out[:, 0:1]
        e_agg = aggr_out[:, 1:]
        
        # 1. Independence Assumption (Sum of logs)
        c_base = c_agg
        
        # 2. Selectivity Correction
        e_combined = e_agg
        correction = self.sel_node(e_combined)
        
        c_new = c_base - correction
        
        # 3. Update Embedding
        e_new = self.emb_update_node(e_combined)
        
        return torch.cat([c_new, e_new], dim=1)

    def message(self, x_i, x_j, edge_attr):
        # --- Case Distinction for Message (Source Node j) ---
        
        # Handle x_j (Source)
        if self.is_first_layer:
            # Layer 1: x_j has shape [N, 3 + D]
            c_j = x_j[:, 0:1] #x_j is the sender, i.e. we pick the subject counts
            c_i = x_i[:, 2:3] #x_i is the receiver, i.e. we pick the object counts
            e_j = x_j[:, 3:]
            e_i = x_i[:, 3:]
        else:
            # Layer >1: x_j has shape [N, 1 + D]
            c_j = x_j[:, 0:1]
            c_i = x_i[:, 0:1]
            e_j = x_j[:, 1:]
            e_i = x_i[:, 1:]
            
        # Handle edge_attr (Predicate)
        # We assume edge_attr ALWAYS has shape [E, 3 + D] (raw stats as they are not transformed by message-passing)
        # We always pick Index 1 (Count as Predicate)
        c_edge = edge_attr[:, 1:2]
        e_edge = edge_attr[:, 3:]
        
        # L2 Normalize the embedding inputs
        e_i = F.normalize(e_i, p=2, dim=1)
        e_j = F.normalize(e_j, p=2, dim=1)
        e_edge = F.normalize(e_edge, p=2, dim=1)

        # 1. Independence Assumption (Triple Pattern)
        c_base = c_i + c_j + c_edge - 2*self.log_N
        
        # Calculating Embedding of triple pattern
        # Revert to sum (remove /3) to preserve gradient magnitude
        e_combined = e_i + e_j + e_edge

        # 2. Selectivity Correction
        tp_sel = self.sel_triple(e_combined)
        
        # Complete Cardinality Estimate
        c_msg = c_base - tp_sel
        
        # 3. Update Embedding for message
        e_msg = self.emb_update_triple(e_combined)
        
        return torch.cat([c_msg, e_msg], dim=1)


class QueryCardinalityGNNOLD(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # feat_dim is the total width of input vector (3 counts + embedding)
        self.emb_dim = feat_dim - 3 
        
        # Layer 1: Handles 3-count input
        self.layer1 = CardinalityConv(self.emb_dim, is_first_layer=True)
        
        # Layer 2: Handles 1-count refined input
        self.layer2 = CardinalityConv(self.emb_dim, is_first_layer=False)
        
        # Global Aggregation
        self.sel_global = NeuralSelectivity(self.emb_dim)


        self.emb_update_global = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
        #self.emb_update_global = nn.Linear(self.emb_dim, self.emb_dim)
        
    def forward(self, batch, N):
        x = batch.x # [Nodes, 3 + D]
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr # [Edges, 3 + D]
        log_N = torch.log(torch.tensor(N))
        
        
        # Layer 1
        # Output x becomes [Nodes, 1 + D]
        x = self.layer1(x, edge_index, edge_attr, log_N)
        
        # Layer 2 (we dont do this for stars as we only have one hop there )
        #x = self.layer2(x, edge_index, edge_attr, log_N)
        
        # --- Step 3: Global Aggregation ---
        #global_sum = global_add_pool(x, batch.batch)


        # Pick the first node of each graph in the batch (which corresponds to the -1 node)
        # batch.ptr[:-1] gives the start index of every graph in the batch
        global_sum = x[batch.ptr[:-1]]
        
        c_global = global_sum[:, 0:1]
        e_global = global_sum[:, 1:]
        
        #correction = self.sel_global(e_global)
        #final_est = c_global - correction
        final_est = c_global
        
        return torch.abs(final_est.squeeze(-1))


class IndependentQueryGNN(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # feat_dim is the total width of input vector (3 counts + embedding)
        self.emb_dim = feat_dim - 3 
        
        # Layer 1: Handles 3-count input
        self.layer1 = IndependentNeuralJoin(self.emb_dim, is_first_layer=True)
        
        
    def forward(self, batch, N):
        x = batch.x # [Nodes, 3 + D]
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr # [Edges, 3 + D]
        log_N = torch.log(torch.tensor(N))
        
        
        # Layer 1
        # Output x becomes [Nodes, 1 + D]
        x = self.layer1(x, edge_index, edge_attr, log_N)
        
        
        # --- Step 3: Global Aggregation ---
        global_sum = x[batch.ptr[:-1]]
        
        c_global = global_sum[:, 0:1]
        e_global = global_sum[:, 1:]
        
        final_est = c_global
        
        return torch.abs(final_est.squeeze(-1))


class IndependentNeuralJoin(MessagePassing):
    """
    Implements the Generalized Neural Cardinality Operator (Psi).
    Handles specific slicing for the first layer where inputs have 3 count dimensions.

    The model expects log values for all counts ! And log(N) for all variable counts !
    """
    def __init__(self, emb_dim, is_first_layer=False):
        super().__init__(aggr='add', flow='target_to_source') 
        self.emb_dim = emb_dim
        self.is_first_layer = is_first_layer



    def forward(self, x_curr, edge_index, edge_attr, log_N):
        self.log_N = log_N
        # Propagate generates the messages (m_j->i) and sums them.
        aggr_out = self.propagate(edge_index, x=x_curr, edge_attr=edge_attr)
        

        # Split Aggregated Message in cardinality estimate and semantic embeddings
        c_agg = aggr_out[:, 0:1]
        e_agg = aggr_out[:, 1:]
        
        # 1. Independence Assumption (Sum of logs)
        c_base = c_agg
        
        # 2. Selectivity Correction
        e_combined = e_agg
        
        c_new = c_base
        
        # 3. Update Embedding
        e_new = e_combined
        
        return torch.cat([c_new, e_new], dim=1)

    def message(self, x_i, x_j, edge_attr):
        # --- Case Distinction for Message (Source Node j) ---
        
        # Handle x_j (Source)
        if self.is_first_layer:
            # Layer 1: x_j has shape [N, 3 + D]
            c_j = x_j[:, 0:1] #x_j is the sender, i.e. we pick the subject counts
            c_i = x_i[:, 2:3] #x_i is the receiver, i.e. we pick the object counts
            e_j = x_j[:, 3:]
            e_i = x_i[:, 3:]
        else:
            # Layer >1: x_j has shape [N, 1 + D]
            c_j = x_j[:, 0:1]
            c_i = x_i[:, 0:1]
            e_j = x_j[:, 1:]
            e_i = x_i[:, 1:]
            
        # Handle edge_attr (Predicate)
        # We assume edge_attr ALWAYS has shape [E, 3 + D] (raw stats as they are not transformed by message-passing)
        # We always pick Index 1 (Count as Predicate)
        c_edge = edge_attr[:, 1:2]
        e_edge = edge_attr[:, 3:]
        
        # L2 Normalize the embedding inputs
        e_i = F.normalize(e_i, p=2, dim=1)
        e_j = F.normalize(e_j, p=2, dim=1)
        e_edge = F.normalize(e_edge, p=2, dim=1)

        # 1. Independence Assumption (Triple Pattern)
        c_base = c_i + c_j + c_edge - 2*self.log_N
        
        # Calculating Embedding of triple pattern
        e_combined = e_i + e_j + e_edge

        
        # Complete Cardinality Estimate
        c_msg = c_base
        
        # 3. Update Embedding for message
        e_msg = e_combined
        
        return torch.cat([c_msg, e_msg], dim=1)


class BaseCardinalityEstimator(MessagePassing):
    """
    Deterministic layer that calculates the independent log-cardinality 
    for every triple pattern (edge) and aggregates them to the nodes.
    
    Formula for triple t = (s, p, o):
    log(|t|) = log(|s|) + log(|p|) + log(|o|) - 2 * log(N)
    """
    def __init__(self):
        super().__init__(aggr='add', flow='target_to_source')
        
    def forward(self, x, edge_index, edge_attr, log_N):
        self.log_N = log_N
        # Propagate calculates log|t| for each edge and sums them at the target node
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_j is Source (Subject), x_i is Target (Object) based on flow='target_to_source'
        # Assuming data loading convention:
        # x index 0: subject count (for bound atoms or variables)
        # x index 2: object count
        # edge_attr index 1: predicate count
        
        # Note: Inputs are assumed to be already in log-space
        c_s = x_j[:, 0:1]
        c_o = x_i[:, 2:3]
        c_p = edge_attr[:, 1:2]
        
        # Independent Joint Cardinality
        # log(|t|) = log(|s|) + log(|p|) + log(|o|) - 2*log(N)
        log_card_t = c_s + c_p + c_o - 2 * self.log_N
        
        return log_card_t



class QueryCardinalityGNNv2(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = feat_dim - 3
        
        # 1. Deterministic Base Estimator
        self.base_estimator = BaseCardinalityEstimator()
        
        # 2. GNN for Semantic Embeddings (Learns Selectivity/Correlations)
        # Input dimension is emb_dim. 
        # We can use GINEConv or a custom TripleConv that acts on embeddings.
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.emb_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=self.emb_dim))
            
        # 3. Node Correction Head
        # Predicts the local selectivity/join reduction term for each node
        self.node_correction_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch, N):
        x = batch.x # [Nodes, 3 + D]
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr # [Edges, 3 + D]
        log_N = torch.log(torch.tensor(N, device=x.device))

        # --- Part 1: Base Cardinality (Sum of Independent Triples) ---
        # Calculate log|t| for every edge and sum up to nodes
        node_base_estimates = self.base_estimator(x, edge_index, edge_attr, log_N)
        
        # Sum over all nodes to get the global sum of edge cardinalities
        # (Since edges are directed in message passing, each edge is counted once at its target)
        # We select the first node of each graph to represent the global sum if we aggregated there,
        # but here we have distributed sums, so we use global_add_pool.
        global_base_est = global_add_pool(node_base_estimates, batch.batch)


        # --- Part 2: Learn Selectivity Correction (GNN on Embeddings) ---
        # Extract embeddings
        h = x[:, 3:]
        edge_emb = edge_attr[:, 3:]
        
        # Run Message Passing
        for conv in self.convs:
            h = conv(h, edge_index, edge_emb)
            h = F.silu(h)
            
        # Predict Correction per Node
        # This captures correlations + structural join reductions
        node_corrections = self.node_correction_mlp(h)
        
        # Sum corrections over the graph
        global_correction = global_add_pool(node_corrections, batch.batch)
        
        # --- Part 3: Combine ---
        # log|Q| = Sum(log|t|) - Correction
        final_est = global_base_est - global_correction
        
        return torch.abs(final_est.squeeze(-1))

class SelectivityConvv3(MessagePassing):
    """
    Specialized layer for selectivity learning:
    - message: computes per-edge log-selectivity ψ(tp) >= 0 from (e_s, e_p, e_o)
               and an embedding message for target.
    - aggregate: sums ψ(tp) at the target node (matches BaseCardinalityEstimator counting).
    - forward: returns updated node embeddings and per-node summed corrections.
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 64, nonneg: bool = True):
        super().__init__(aggr='add', flow='target_to_source')
        self.emb_dim = emb_dim
        self.nonneg = nonneg

        # Role-aware projections
        self.proj_s = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.proj_p = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.proj_o = nn.Linear(emb_dim, hidden_dim, bias=False)

        # Triple composer
        self.triple_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Heads: selectivity scalar and embedding message
        self.psi_head = nn.Linear(hidden_dim, 1)
        self.msg_head = nn.Linear(hidden_dim, emb_dim)

        # Node update from aggregated message (residual)
        self.node_update = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # e_s (source), e_p (predicate), e_o (target)
        e_s = F.normalize(x_j, p=2, dim=1)
        e_o = F.normalize(x_i, p=2, dim=1)
        e_p = F.normalize(edge_attr, p=2, dim=1)

        # Role-aware triple encoding
        h_s = self.proj_s(e_s)
        h_p = self.proj_p(e_p)
        h_o = self.proj_o(e_o)

        h_t = self.triple_mlp(torch.cat([h_s, h_p, h_o], dim=1))

        psi = self.psi_head(h_t)
        if self.nonneg:
            psi = F.softplus(psi)  # enforce ψ(tp) >= 0

        delta = self.msg_head(h_t)  # embedding message for target node
        return torch.cat([psi, delta], dim=1)  # [E, 1 + emb_dim]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_emb: torch.Tensor):
        """
        x: [N, emb_dim]           node embeddings (only semantics)
        edge_emb: [E, emb_dim]    edge embeddings (only semantics)
        returns:
          x_new: [N, emb_dim]
          node_corr: [N, 1]       summed ψ(tp) for incoming edges
        """
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_emb)  # [N, 1 + emb_dim]
        node_corr = aggr[:, :1]
        msg = aggr[:, 1:]

        x_new = x + self.node_update(torch.cat([x, msg], dim=1))


        return x_new, node_corr


class QueryCardinalityGNNv3(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = feat_dim - 3

        self.base_estimator = BaseCardinalityEstimator()
        self.sel_convs = nn.ModuleList(
            [SelectivityConvv3(self.emb_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, batch, N):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        log_N = torch.log(torch.tensor(N, device=x.device))

        # Track 1: independent base estimate
        node_base = self.base_estimator(x, edge_index, edge_attr, log_N)
        global_base = global_add_pool(node_base, batch.batch)

        # Track 2: selectivity as per-edge ψ(tp), counted once at targets
        h = x[:, 3:]            # node semantics only
        e = edge_attr[:, 3:]    # edge semantics only

        total_node_corr = None
        for conv in self.sel_convs:
            h, node_corr = conv(h, edge_index, e)
            total_node_corr = node_corr if total_node_corr is None else total_node_corr + node_corr

        global_corr = global_add_pool(total_node_corr, batch.batch)

        final = global_base - global_corr
        return torch.abs(final.squeeze(-1))


class CardinalityConv(MessagePassing):
    """
    Phase 1:
    1. Calculates independent log-cardinality for every triple pattern.
    2. Predicts and subtracts a Triple Correction (ψ_edge) from it.
    3. Aggregates (sums) these corrected triple estimates at the node.
    4. Predicts and subtracts a Join Correction (ψ_node) at the node
    IMPORTANT: This conv should be only appplied ONCE to the initial query graph.
    """
    def __init__(self, emb_dim, hidden_dim=128):
        super().__init__(aggr='mean', flow='target_to_source')
        self.emb_dim = emb_dim
        
        # Triple Pattern Correction
        # Input: [emb_s, emb_p, emb_o]
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1), # Outputs ψ_edge
            # nn.Softplus() # Optional: enforce positive correction
        )
        
        # Node Join Correction (Node Level)
        # Input: [aggregated_embedding]
        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1), # Outputs ψ_node
             # nn.Softplus()
        )
        
        # Embedding Update (to pass to Phase 2)
        self.emb_update = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        
    def forward(self, x, edge_index, edge_attr, log_N):
        self.log_N = log_N
        
        # Propagate: Calculates corrected edge messages and sums them
        # Returns: [log_card_sum, aggregated_embedding]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        c_sum = out[:, 0:1]
        e_agg = out[:, 1:]
        
        # --- Node Level Correction (Join) ---
        # The sum of edge logs implies a Cartesian product of all incoming edges.
        # We need to subtract a correction because they join on this node.
        join_correction = self.node_mlp(e_agg)
        
        c_final = c_sum - join_correction
        
        # Update embedding for next phase
        e_final = self.emb_update(e_agg)
        
        return c_final, e_final

    def message(self, x_i, x_j, edge_attr):
        # x_j: Source (Subject), x_i: Target (Object)
        
        # 1. Counts (Log Space)
        c_s = x_j[:, 0:1]
        c_o = x_i[:, 2:3]
        c_p = edge_attr[:, 1:2]
        
        # 2. Embeddings
        e_s = x_j[:, 3:]
        e_o = x_i[:, 3:]
        e_p = edge_attr[:, 3:]
        
        # Normalize embeddings (so model learns to model similarity via orientation and not magnitude)
        e_s = F.normalize(e_s, p=2, dim=1)
        e_o = F.normalize(e_o, p=2, dim=1)
        e_p = F.normalize(e_p, p=2, dim=1)
        
        # --- Independent Cardinality ---
        # log(|t|) = log(|s|) + log(|p|) + log(|o|) - 2*log(N)
        log_card_ind = c_s + c_p + c_o - 2 * self.log_N
        
        # --- Learned Triple Correction ---
        # ψ_edge(e_s, e_p, e_o)
        triple_emb = torch.cat([e_s, e_p, e_o], dim=1)
        edge_correction = self.edge_mlp(triple_emb)
        
        # Corrected Edge Estimate
        log_card_corrected = log_card_ind - edge_correction
        
        # Pass the combined embedding to the node for aggregation
        # (Simple sum aggregation of embeddings)
        msg_emb = e_s + e_p + e_o 
        
        return torch.cat([log_card_corrected, msg_emb], dim=1)


class SelectivityConv(MessagePassing):
    """
    Phase 2: Refinement
    Operates on the graph structure to refine estimates.
    Does NOT use raw edge counts, only embeddings.
    """
    def __init__(self, emb_dim, hidden_dim=128):
        super().__init__(aggr='add', flow='target_to_source')
        self.emb_dim = emb_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs scalar correction update
        )
        
        self.emb_update = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x_emb, edge_index):
        # x_emb: Node embeddings from previous step
        return self.propagate(edge_index, x=x_emb)

    def message(self, x_i, x_j):
        # Only uses node embeddings (source and target)
        # No edge features used here as per requirement
        cat_emb = torch.cat([x_i, x_j], dim=1)
        
        correction = self.mlp(cat_emb)
        new_emb = self.emb_update(cat_emb)
        
        return torch.cat([correction, new_emb], dim=1)
        
    def update(self, aggr_out, x):
        # aggr_out has [sum_correction, sum_new_emb]
        delta_c = aggr_out[:, 0:1]
        delta_e = aggr_out[:, 1:]
        
        # Update embedding residual
        x_new = x + delta_e
        
        # Return both correction delta and updated embedding
        return delta_c, x_new


class QueryCardinalityGNN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=0):
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = feat_dim - 3
        
        # Phase 1: Base + Initial Correction
        self.phase1 = CardinalityConv(self.emb_dim, hidden_dim)
        
        # Phase 2: Refinement (No edge features)
        self.phase2_layers = nn.ModuleList([
            SelectivityConv(self.emb_dim, hidden_dim) 
            for _ in range(num_layers)
        ])

    def forward(self, batch, N):
        x = batch.x # [Nodes, 3 + D]
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr # [Edges, 3 + D]
        log_N = torch.log(torch.tensor(N, device=x.device))

        # --- Phase 1 ---
        # c_curr: [Nodes, 1] (Log cardinality estimate per node; accounting for tp cardinalitiies and first node joins)
        # h_curr: [Nodes, D] (Updated semantic embedding)
        c_curr, h_curr = self.phase1(x, edge_index, edge_attr, log_N)
        
        # --- Phase 2 ---
        # Propagate first node joins to perform all subsequent joins
        # Here we only calculate selectivities to avoid overcounting issues
        for conv in self.phase2_layers:
            delta_c, h_curr = conv(h_curr, edge_index)
            # Subtract corrections per node and per local neighborhood aggregation
            c_curr = c_curr - delta_c
            
        # --- Global Aggregation ---
        # Sum the final log-cardinalities from all nodes
        # They now already represent selectivity corrected cardinalities
        
        global_est = global_add_pool(c_curr, batch.batch)
        
        return torch.abs(global_est.squeeze(-1))


