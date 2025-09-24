import torch
from torch import nn

from fice.ultra import tasks, layers
from fice.ultra.base_nbfnet import BaseNBFNet
from easydict import EasyDict


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, data, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.data = data
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )


    def bellmanford(self, batch):
        query_relations = batch.edge_type
        batch_size = batch.batch_size
        query = torch.ones(batch_size, self.dims[0], device=query_relations.device, dtype=torch.float)
        
        #TODO: wenn ich eine query einlese, dann gebe ich ja jedem edge_type eine lokale id.
        # wenn ich dann beim batching die queries zuammenfasse, dann haben dadurchv verschiedene
        # edges und edge types die selbe lokale id
        # Beispiel: 
        # Query 1:
        #   Paris - haupstadt von - ?
        # Also lokales mapping: 2 - 1 - 3
        # 1: haupstadt von
        # 2: Paris
        # 3: ?
        #
        ## Query 2:
        #   Berlin - liegt in  - Deutschland
        # Also lokales mapping: 2 - 1 - 3
        # 1: liegt in
        # 2: Berlin
        # 3: Deutschland
        # 
        #  Als batch: 
        # 2 - 1 - 3 (Query 1)
        # 4 - 1 - 5 (Query 2)
        #  
        #  Beide eges haben die id 1 obwohl es zwei verschiedene relationen sind
        #  
        # # TODO: Lösungsidee: mit __inc__() arbeiten
        # https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html
        # Kann aber auch sein, dass das ganze kein problem ist 
        #  




        # Labeling Trick
        # initialize initial nodes (relations of interest in the batch) with all ones
        boundary = torch.zeros(batch_size, self.data.num_nodes, self.dims[0], device=query_relations.device)
        
        # Create a mask to filter out invalid edge embedding indices (-1)
        valid_mask = batch.edge_embedding_indices != -1

        # Get the graph index for each edge's source node
        edge2graph = batch.batch[batch.edge_index[0]]

        # Filter the graph indices and edge embedding indices using the mask
        graph_indices = edge2graph[valid_mask]
        node_indices = batch.edge_embedding_indices[valid_mask]

        # Use advanced indexing to set the boundary values to 1.0 in a single operation
        boundary[graph_indices, node_indices] = 1.0

        assert boundary.shape[0] == batch.batch_size, "Output shape does not match batch size"
        assert boundary.shape[1] == self.data.num_nodes, "Output shape does not match number of relations"
        assert boundary.shape[2] == self.dims[0]
        for i in range(batch_size):
            data_list = batch.to_data_list()[i].edge_embedding_indices.unique()
            mask = data_list != -1
            edge_idx = data_list[mask]
            assert (int(boundary[i].sum() / self.dims[0]) == edge_idx.numel()), f"Boundary condition for graph {i} is all zeros!"



        size = (self.data.num_nodes, self.data.num_nodes)
        edge_weight = torch.ones(self.data.num_edges, device=query_relations.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, self.data.edge_index, self.data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, self.data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        assert output.shape[0] == batch.batch_size, "Output shape does not match batch size"
        assert output.shape[1] == self.data.num_nodes, "Output shape does not match number of relations"
        assert output.shape[2] == 64

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, batch):
        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(batch)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        return output
    



class EntityNBFNet(BaseNBFNet):

    def __init__(self, data, input_dim, hidden_dims, num_relation=1, **kwargs):
    
        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.data = data
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

    def bellmanford(self, batch, relation_representations):
        batch_size = batch.batch_size
        device = batch.edge_index.device

        # initialize queries (relation types of the given triples)
        # TODO: hier war mal r_index anstell von ones, rechercheiren 
        # https://github.com/DeepGraphLearning/ULTRA/blob/main/ultra/models.py
        query = self.query[torch.arange(batch_size, device=device), torch.ones(batch_size, device=device, dtype=torch.long)]

        boundary = torch.zeros(batch_size, self.data.num_nodes, self.dims[0], device=query.device)

        # Create a mask to filter out invalid edge embedding indices (-1)
        valid_mask = batch.node_embedding_indices != -1

        # Get the graph index for each edge's source node
        node2graph = batch.batch

        # Filter the graph indices and edge embedding indices using the mask
        graph_indices = node2graph[valid_mask]
        node_indices = batch.node_embedding_indices[valid_mask]

        # Use advanced indexing to set the boundary values to 1.0 in a single operation
        boundary[graph_indices, node_indices] = 1.0

        assert boundary.shape[0] == batch.batch_size, "Output shape does not match batch size"
        assert boundary.shape[1] == self.data.num_nodes, "Output shape does not match number of relations"
        assert boundary.shape[2] == self.dims[0]
        for i in range(batch_size):
            data_list = batch.to_data_list()[i].node_embedding_indices.unique()
            mask = data_list != -1
            edge_idx = data_list[mask]
            assert (int(boundary[i].sum() / self.dims[0]) == edge_idx.numel()), f"Boundary condition for graph {i} is all zeros!"

        size = (self.data.num_nodes, self.data.num_nodes)
        edge_weight = torch.ones(self.data.num_edges, device=device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            # if separate_grad:
            #     edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, self.data.edge_index, self.data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, batch ,relation_representations):

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # message passing and updated node representations
        output = self.bellmanford(batch, relation_representations)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"] 
        return feature

