import torch
from typing import Dict, List, Union
from torch_geometric.data import Data
# Try to import the low-level NeighborSampler (newer API). If unavailable, we'll fallback.
try:
    from torch_geometric.sampler import NeighborSampler as PYGNeighborSampler  # type: ignore
    from torch_geometric.sampler import NodeSamplerInput  # type: ignore
    HAS_PYG_SAMPLER = True
except Exception:
    HAS_PYG_SAMPLER = False


class MultigraphNeighborSampler():
    def __init__(self, num_neighbors: List[int] = []):
        self.cache: Dict[str, Union[PYGNeighborSampler, NodeSamplerInput]] = {}
        self.num_neighbors = num_neighbors
            
    def _create_reusable_sampler(self, data_graph, num_neighbors):
        """Create a reusable sampler if the low-level sampler API is available.

        Returns sampler or None.
        """
        if not HAS_PYG_SAMPLER:
            return None
        # Newer API takes a Data object / graph store
        sampler = PYGNeighborSampler(
            data_graph,
            num_neighbors=num_neighbors,
            replace=False,
        )
        return sampler


    def _sample_with_sampler(self, sampler, data_graph, seeds: torch.Tensor):
        # 1) Sample neighborhood
        # Wrap seeds in NodeSamplerInput
        inputs = NodeSamplerInput(input_id=None, node=seeds)
        # get output with fields
        # node: sampled node ids (global ids)
        # row: row indices of the sampled edges (local indices)
        # col: column indices of the sampled edges (local indices)
        # edge: edge indices of the sampled edges (original edge ids from full graph)
        out = sampler.sample_from_nodes(inputs)

        # 2) Get sampled global node IDs
        n_id = getattr(out, 'node', None)
        if n_id is None:
            n_id = getattr(out, 'n_id', None)
        if n_id is None:
            raise RuntimeError("Sampler output missing node ids (node / n_id)")

        # 3) Local edge_index from sampler (row/col are local indices)
        edge_index_local = torch.stack([out.row, out.col], dim=0)

        # 4) Build subgraph Data directly
        sub_data = Data()
        sub_data.edge_index = edge_index_local
        sub_data.num_nodes = n_id.numel()
        
        # TODO: carry other node features in the subgraph 
        

        # carry edge types via original edge IDs
        # we put this as optional but should always be there
        if hasattr(out, 'edge'):
            if hasattr(data_graph, 'edge_type'):
                sub_data.edge_type = data_graph.edge_type[out.edge]
            
            if hasattr(data_graph, 'edge_attr') and data_graph.edge_attr is not None:
                sub_data.edge_attr = data_graph.edge_attr[out.edge]

        # 5) Global->local mapping (global ID -> local [0..N-1])
        # i.e. which id do all the global ids from the full graph have (if they dont have one, they map to -1)
        global_to_local = torch.full((data_graph.num_nodes,), -1, dtype=torch.long, device=n_id.device)
        global_to_local[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        return sub_data, global_to_local.detach().cpu()
    

    def sample(self, full_graph, seeds: torch.Tensor):
        """Sample neighborhood for given seeds using cached sampler."""


        sampler = self.cache.get(full_graph.dataset_name, None)

        if sampler is None:
            sampler = self._create_reusable_sampler(full_graph, num_neighbors=self.num_neighbors)
            self.cache[full_graph.dataset_name] = sampler

        return self._sample_with_sampler(sampler, full_graph, seeds)

