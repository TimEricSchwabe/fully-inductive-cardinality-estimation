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
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), 
            nn.ReLU() 
        )

    def forward(self, x):
        return self.net(x)



class NeuralJoin(MessagePassing):
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
        # Basically Message Passing MLP of the embeddings
        self.emb_update_triple = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            # torch.nn.Dropout(p=0.2)
        )
        #self.emb_update_triple = nn.Linear(emb_dim, emb_dim)

        # S_theta for Node Join (Update)
        self.sel_node = NeuralSelectivity(emb_dim)
        self.emb_update_node = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
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
        e_new = F.relu(self.emb_update_node(e_combined))
        
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
        e_combined = (e_i + e_j + e_edge) / 3 # mean now

        # 2. Selectivity Correction
        tp_sel = self.sel_triple(e_combined)
        
        # Complete Cardinality Estimate
        c_msg = c_base - tp_sel
        
        # 3. Update Embedding for message
        e_msg = F.normalize(self.emb_update_triple(e_combined), p=2, dim=1)
        
        return torch.cat([c_msg, e_msg], dim=1)


class QueryCardinalityGNN(torch.nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # feat_dim is the total width of input vector (3 counts + embedding)
        self.emb_dim = feat_dim - 3 
        
        # Layer 1: Handles 3-count input
        self.layer1 = NeuralJoin(self.emb_dim, is_first_layer=True)
        
        # Layer 2: Handles 1-count refined input
        self.layer2 = NeuralJoin(self.emb_dim, is_first_layer=False)
        
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