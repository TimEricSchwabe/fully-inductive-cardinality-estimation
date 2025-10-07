import os
import pathlib
import torch
import json
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from typing import Optional, Dict, List, Tuple

from fice.ultra.tasks import build_relation_graph
from tqdm import tqdm



# TODO: Make the nt_path file optinal
class NTDatasets(InMemoryDataset):
    def __init__(
        self,
        root: str,
        nt_path: str,
        dataset_name: Optional[str] = None,
        transform=None,
        pre_transform=None,
    ):

        self.root = pathlib.Path(root)
        self.nt_path = pathlib.Path(nt_path)
        self.dataset_name = (
            dataset_name if dataset_name is not None else self.nt_path.stem
        )

        # forward transform / pre_transform and use string root as expected by InMemoryDataset
        super().__init__(str(self.root), transform, pre_transform)
        # load processed data (if present)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}.pt"]

    @property
    def raw_file_names(self):
        return [self.nt_path.name]

    # @property
    # def raw_dir(self):
    #     return super().raw_dir

    # @property
    # def processed_dir(self):
    #     return super().processed_dir

    def process(self):
        # ensure processed folder exists
        os.makedirs(self.processed_dir, exist_ok=True)

        # Create id mapping and count occurrences
        entity_to_id: Dict[str, int] = {}
        relation_to_id: Dict[str, int] = {}
        occurrences: Dict[str, List[int]] = {}
        mapped_triplets: List[Tuple[int, int, int]] = []

        # Stream the NT file line-by-line and parse robustly
        with open(self.nt_path, "r", encoding="utf-8") as file:
            for line in file:
                # split into three last whitespace-separated tokens (s, p, o.)
                parts = line.strip().split(" ")[:3]
                if len(parts) != 3:
                    # skip malformed/empty lines
                    continue
                s, p, o = tuple(parts)

                # count occurrences: [head_count, relation_count, tail_count]
                occurrences.setdefault(s, [0, 0, 0])[0] += 1
                occurrences.setdefault(p, [0, 0, 0])[1] += 1
                occurrences.setdefault(o, [0, 0, 0])[2] += 1

                if s not in entity_to_id:
                    entity_to_id[s] = len(entity_to_id)
                if o not in entity_to_id:
                    entity_to_id[o] = len(entity_to_id)
                if p not in relation_to_id:
                    relation_to_id[p] = len(relation_to_id)

                mapped_triplets.append(
                    (entity_to_id[s], relation_to_id[p], entity_to_id[o])
                )

        num_nodes = len(entity_to_id)
        num_relations = len(relation_to_id)

        # write mappings in a readable/stable way
        with open(
            os.path.join(self.processed_dir, f"{self.dataset_name}_entity_to_id.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(entity_to_id, f, indent=2, sort_keys=True)
        with open(
            os.path.join(
                self.processed_dir, f"{self.dataset_name}_relation_to_id.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(relation_to_id, f, indent=2, sort_keys=True)

        # split occurrences by kind (keep original string keys for readability)
        occurrences_relations: Dict[str, List[int]] = {}
        occurrences_entity: Dict[str, List[int]] = {}
        for key, counts in occurrences.items():
            if key in entity_to_id:
                occurrences_entity[key] = counts
            elif key in relation_to_id:
                occurrences_relations[key] = counts
            else:
                # should not happen, but be robust
                continue

        with open(
            os.path.join(
                self.processed_dir, f"{self.dataset_name}_entity_occurrences.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(occurrences_entity, f, indent=2, sort_keys=True)

        with open(
            os.path.join(
                self.processed_dir, f"{self.dataset_name}_relation_occurrences.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(occurrences_relations, f, indent=2, sort_keys=True)

        # build tensors with explicit dtypes
        if len(mapped_triplets) == 0:
            raise ValueError(f"No valid triplets parsed from {self.nt_path}")

        triplets = torch.tensor(mapped_triplets, dtype=torch.long)
        edge_index = triplets[:, [0, 2]].t().contiguous()
        edge_type = triplets[:, 1].contiguous().long()

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            num_relations=num_relations,
        )

        # Apply pre_transform if provided, otherwise ensure relation graph is built
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        else:
            data = build_relation_graph(data)

        assert (
            getattr(data, "relation_graph", None) is not None
        ), "Relation graph not built"

        # Save in the same location that __init__ expects
        torch.save(self.collate([data]), self.processed_paths[0])
        return None


class NTQueries(InMemoryDataset):
    def __init__(
        self,
        root: str,
        queries_path: str,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        device: str = "cpu",
        dataset_name: Optional[str] = None,
        transform=None,
        pre_transform=None,
        dim: int = 128,
    ):
        self.dim = dim
        if entity_occurrences is None:
            entity_occurrences = {}
        if relation_occurrences is None:
            relation_occurrences = {}
        if global_entity_mapping is None:
            global_entity_mapping = {}
        if global_relation_mapping is None:
            global_relation_mapping = {}

        self.entity_occurrences = entity_occurrences
        self.relation_occurrences = relation_occurrences
        self.global_entity_mapping = global_entity_mapping
        self.global_relation_mapping = global_relation_mapping
        self.device = device

        self.root = pathlib.Path(root)
        self.queries_path = pathlib.Path(queries_path)
        self.dataset_name = (
            dataset_name if dataset_name is not None else self.queries_path.stem
        )

        # forward transform / pre_transform and use string root as expected by InMemoryDataset
        super().__init__(str(self.root), transform, pre_transform)
        # load processed data (if present)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}.pt"]

    @property
    def raw_file_names(self):
        return [self.queries_path.name]

    def process_query_single(
        self,
        query: Dict,
    ):
        feat_dim = self.dim
        # Create local mapping
        local_entity_mapping = {}
        local_relation_mapping = {}
        query_edge_index = []
        query_edge_type = []

        for triple in query["triples"]:

            h, r, t = triple[0], triple[1], triple[2]
            if h not in local_entity_mapping:
                local_entity_mapping[h] = len(local_entity_mapping)
            if t not in local_entity_mapping:
                local_entity_mapping[t] = len(local_entity_mapping)
            if r not in local_relation_mapping:
                local_relation_mapping[r] = len(local_relation_mapping)

            query_edge_index.append([local_entity_mapping[h], local_entity_mapping[t]])
            query_edge_type.append(local_relation_mapping[r])

        num_nodes = len(local_entity_mapping)
        num_edges = len(query["triples"])

        query_edge_index = (
            torch.tensor(query_edge_index, dtype=torch.long).t().to(self.device)
        )
        assert query_edge_index.dim() == 2 and query_edge_index.size(0) == 2 and query_edge_index.size(1) == num_edges

        query_edge_type = torch.tensor(query_edge_type, dtype=torch.long).to(self.device)

        assert query_edge_type.dim() == 1 and query_edge_type.size(0) == num_edges

        # Process Nodes
        query_features = torch.zeros(
            num_nodes, feat_dim + 3, dtype=torch.float, device=self.device
        )
        embedding_indices = torch.full(
            (num_nodes,), -1, dtype=torch.long
        )
        for entity, idx in local_entity_mapping.items():
            if entity in self.global_entity_mapping:
                embedding_indices[idx] = self.global_entity_mapping[entity]

            if entity in self.entity_occurrences:
                occ = self.entity_occurrences[entity]
                query_features[idx, 0] = occ[0]
                query_features[idx, 1] = occ[1]
                query_features[idx, 2] = occ[2]



        # Process Edges
        # HACK: Works for now but maybe find more beatuful solution
        query_edge_type_string = [t[1] for t in query.get('triples')]

        query_edge_attr = torch.zeros(
            num_edges, feat_dim + 3, dtype=torch.float, device=self.device
        )
        edge_embedding_indices = torch.full(
            (num_edges,), -1, dtype=torch.long
        )
        for idx, edge in enumerate(query_edge_type_string):
            if edge in self.global_relation_mapping:
                edge_embedding_indices[idx] = self.global_relation_mapping[edge]
        
            if edge in self.relation_occurrences:
                occ = self.relation_occurrences[edge]
                query_edge_attr[idx, 0] = occ[0]
                query_edge_attr[idx, 1] = occ[1]
                query_edge_attr[idx, 2] = occ[2]

        assert query_edge_index.size(1) == query_edge_type.size(0)
        assert query_edge_index.size(1) == query_edge_attr.size(0)
        assert query_edge_attr.size(0) == query_edge_type.size(0)
        assert edge_embedding_indices.size(0) == query_edge_attr.size(0)

        return Data(
            x=query_features,
            edge_attr=query_edge_attr,

            edge_index=query_edge_index,
            edge_type=query_edge_type,
            y=query["y"],

            node_embedding_indices=embedding_indices,
            edge_embedding_indices=edge_embedding_indices
        )

    def process(self):

        with open(self.queries_path, "r") as file:
            data = json.load(file)
        queries = [
            self.process_query_single(
                q
            )
            for q in tqdm(data, desc="Processing queries")
        ]

        torch.save(self.collate(queries), self.processed_paths[0])
