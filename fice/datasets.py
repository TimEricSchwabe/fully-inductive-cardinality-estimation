import pathlib
from pathlib import Path
import torch
import json
from torch_geometric.data import Data, InMemoryDataset
from typing import Optional, Dict, List, Tuple, Union, Literal
import requests
from tqdm import tqdm as _tqdm

from fice.tasks import build_relation_graph
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data"

def download(url: str, dest: Union[Path, str], progress: bool = False):
    resp = requests.get(url, stream=True, timeout=30) # headers={"User-Agent": "Mozilla/5.0"}
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0))
    chunk_size = 1024 * 16

    if progress:
        with open(dest, 'wb') as f, _tqdm(
            total=total, unit='B', unit_scale=True, desc=f"Downloading {dest.name}"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    else:
        with open(dest, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)

class NTDatasets(InMemoryDataset):
    def __init__(
        self,
        url: str,
        root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        transform=None,
        pre_transform=None,
    ):

        self.url = url
        raw_file_path = pathlib.Path(self.url).name

        if raw_file_path is None or raw_file_path == "":
            raw_file_path = "raw_dataset.nt"

        self.dataset_name = (
            dataset_name if dataset_name is not None else Path(self.url).stem
        )

        if root is None:
            # use resolved/normalized dataset name attribute
            self.root = DEFAULT_DATA_ROOT / self.dataset_name
        else:
            self.root = pathlib.Path(root)

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

        # prepare raw file path from url
        raw_dir = pathlib.Path(self.root) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        self.raw_file_path = raw_dir / raw_file_path

        # forward transform / pre_transform and use string root as expected by InMemoryDataset
        super().__init__(str(self.root), transform, pre_transform)

        # load processed data (keep on CPU - will be moved to device during batching)
        data, slices = torch.load(self.processed_paths[0], weights_only=False)
        # Force data to stay on CPU regardless of device parameter
        self.data, self.slices = data, slices

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}.pt"]

    @property
    def raw_file_names(self):
        return [self.raw_file_path.name]

    def download(self):
        # Stream download with requests and show tqdm progress bar
        try:
            download(self.url, self.raw_file_path, progress=True)
        except Exception as download_exc:
            raise RuntimeError(f"Failed to download dataset from {self.url}: {download_exc}") from download_exc


    def process(self):
        # ensure processed folder exists
        pathlib.Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        # Create id mapping and count occurrences
        entity_to_id: Dict[str, int] = {}
        relation_to_id: Dict[str, int] = {}
        occurrences_relations: Dict[str, List[int]] = {}
        occurrences_entity: Dict[str, List[int]] = {}

        mapped_triplets: List[Tuple[int, int, int]] = []

        # Stream the NT file line-by-line and parse robustly
        with open(self.raw_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Processing triples"):
                # split into three last whitespace-separated tokens (s, p, o.)
                parts = line.strip().split(" ")[:3]
                if len(parts) != 3:
                    raise ValueError(f"Malformed line: {line}")
                    # skip malformed/empty lines
                    continue
                s, p, o = tuple(parts)

                # count occurrences: [head_count, relation_count, tail_count]
                occurrences_entity.setdefault(s, [0, 0, 0])[0] += 1
                occurrences_relations.setdefault(p, [0, 0, 0])[1] += 1
                occurrences_entity.setdefault(o, [0, 0, 0])[2] += 1

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
        entity_to_id_path = pathlib.Path(self.processed_dir) / "entity_to_id.json"
        relation_to_id_path = pathlib.Path(self.processed_dir) / "relation_to_id.json"

        with entity_to_id_path.open("w", encoding="utf-8") as f:
            json.dump(entity_to_id, f, indent=2, sort_keys=True)
        with relation_to_id_path.open("w", encoding="utf-8") as f:
            json.dump(relation_to_id, f, indent=2, sort_keys=True)

        entity_occurrences_path = pathlib.Path(self.processed_dir) / "entity_occurrences.json"
        relation_occurrences_path = pathlib.Path(self.processed_dir) / "relation_occurrences.json"

        with entity_occurrences_path.open("w", encoding="utf-8") as f:
            json.dump(occurrences_entity, f, indent=2, sort_keys=True)

        with relation_occurrences_path.open("w", encoding="utf-8") as f:
            json.dump(occurrences_relations, f, indent=2, sort_keys=True)

        # build tensors with explicit dtypes
        if len(mapped_triplets) == 0:
            raise ValueError(f"No valid triplets parsed from {self.nt_path}")

        triplets = torch.tensor(mapped_triplets, dtype=torch.long)
        edge_index = triplets[:, [0, 2]].t().contiguous()
        edge_type = triplets[:, 1].contiguous().long()

        graph = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            num_relations=num_relations,
        )

        # Apply pre_transform if provided, otherwise ensure relation graph is built
        if self.pre_transform is not None:
            rel_graph = self.pre_transform(graph)
        else:
            rel_graph = build_relation_graph(graph)

        graph.relation_graph = rel_graph

        assert (
            getattr(graph, "relation_graph", None) is not None
        ), "Relation graph not built"

        # Save in the same location that __init__ expects
        graph.dataset_name = self.dataset_name
        torch.save(self.collate([graph]), self.processed_paths[0])

class LUBM(NTDatasets):       
    def __init__(
        self,**kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59410577",
            dataset_name="lubm",
            **kwargs
        )


class SWDF(NTDatasets):       
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59320634",
            dataset_name="swdf",
            **kwargs
        )


class fb237_v4(NTDatasets):
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59621618?private_link=a931a2f5f4c8c39cfb22",
            dataset_name="fb237_v4",
            **kwargs
        )

class fb237_v4_ind(NTDatasets):
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59621825?private_link=364494eb331e3973ecb3",
            dataset_name="fb237_v4_ind",
            **kwargs
        )

class WN18RR_v4(NTDatasets):
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59622641?private_link=e97f83a035f7e24bb37c",
            dataset_name="wn18rr_v4",
            **kwargs
        )

class WN18RR_v4_ind(NTDatasets):
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59622656?private_link=49ad78ce6e253240223f",
            dataset_name="wn18rr_v4_ind",
            **kwargs
        )

class DBpedia100k(NTDatasets):
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59622674?private_link=f9b1fb2716ce2814515e",
            dataset_name="dbpedia100k",
            **kwargs
        )

class Wikidata(NTDatasets):
    def __init__(
        self, **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59320361",
            dataset_name="wikidata",
            **kwargs
        )
class NTQueries(InMemoryDataset):
    def __init__(
        self,
        url: str,
        root: Optional[str] = None,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        dataset_name: Optional[str] = None,
        transform=None,
        pre_transform=None,
        dim: int = 128,
        type: Literal["star", "path"] = "star",
    ):
        self.url = url
        self.dim = dim
        self.query_type = type
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
        # determine dataset root and raw file locations similar to NTDatasets
        self.dataset_name = (
            dataset_name if dataset_name is not None else pathlib.Path(self.url).stem
        )

        if root is None:
            self.root = DEFAULT_DATA_ROOT / self.dataset_name
        else:
            self.root = pathlib.Path(root)

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

        raw_dir = pathlib.Path(self.root) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Keep the original queries source (can be a URL or a local path)
        raw_file_name = pathlib.Path(self.url).name
        if raw_file_name is None or raw_file_name == "":
            raw_file_name = "raw_queries.json"

        # The dataset expects the raw file to live in the dataset's raw/ folder
        self.raw_file_path = raw_dir / raw_file_name
        # For backward compatibility keep attribute used elsewhere
        self.dataset_name = (
            dataset_name if dataset_name is not None else self.raw_file_path.stem
        )

        # forward transform / pre_transform and use string root as expected by InMemoryDataset
        super().__init__(str(self.root), transform, pre_transform)
        # load processed data (keep on CPU - will be moved to device during batching)
        data, slices = torch.load(self.processed_paths[0], weights_only=False)
        # Force data to stay on CPU regardless of device parameter
        self.data, self.slices = data, slices

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}.pt"]

    @property
    def raw_file_names(self):
        return [self.raw_file_path.name]

    def download(self):
        try:
            download(self.url, self.raw_file_path, progress=True)
        except Exception as download_exc:
            raise RuntimeError(f"Failed to download dataset from {self.url}: {download_exc}") from download_exc


    def process_query_single(
        self,
        query: Dict,
        query_set_name: str = "unknown",
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
            torch.tensor(query_edge_index, dtype=torch.long).t()
        )
        assert query_edge_index.dim() == 2 and query_edge_index.size(0) == 2 and query_edge_index.size(1) == num_edges

        query_edge_type = torch.tensor(query_edge_type, dtype=torch.long)

        assert query_edge_type.dim() == 1 and query_edge_type.size(0) == num_edges

        # Process Nodes
        query_features = torch.zeros(
            num_nodes, feat_dim + 3, dtype=torch.float
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
        query_edge_type_string = [t[1] for t in query.get('triples')]

        query_edge_attr = torch.zeros(
            num_edges, feat_dim + 3, dtype=torch.float
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
            edge_embedding_indices=edge_embedding_indices,
            type=self.query_type,
            query_set_name=query_set_name,
        )

    def process(self):
        # ensure processed folder exists
        pathlib.Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        # Read the queries JSON from the dataset raw/ folder
        with open(self.raw_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        queries = [
            self.process_query_single(
                q,
                query_set_name=self.dataset_name

            )
            for q in tqdm(data, desc="Processing queries")
        ]

        torch.save(self.collate(queries), self.processed_paths[0])

class LUBMQueries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59410580",
            dataset_name="lubm_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )

class SWDF_Star_Queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59320652",
            dataset_name="swdf_star_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )

class WIKIDATA_Star_Queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59320583",
            dataset_name="wikidata_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )


class fb237_v4_star_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59651327?private_link=780ccfbe4856b223d7b5",
            dataset_name="fb237_v4_star_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )

class fb237_v4_ind_star_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59731859?private_link=953281688dbdb230c09d",
            dataset_name="fb237_v4_ind_star_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )
    
class wn18rr_v4_star_queries(NTQueries):

    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59731877?private_link=2b8a1c9a43910e6a2189",
            dataset_name="wn18rr_v4_star_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )

class wn18rr_v4_ind_star_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59731889?private_link=ec764d8f68a39b467a60",
            dataset_name="wn18rr_v4_ind_star_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )


class dbpedia100k_star_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59687708?private_link=1e23a3388a4b6b6161d1",
            dataset_name="dbpedia100k_star_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            **kwargs
        )

class lubm_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/60037037?private_link=8e4493e070acfd065b77",
            dataset_name="lubm_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )


class swdf_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/60037112?private_link=81b137c60854d11ee270",
            dataset_name="swdf_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )

class wikidata_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/60037220?private_link=435c4febc01f57ca3b6d",
            dataset_name="wikidata_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )




class wn18rr_v4_ind_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59898623?private_link=87c951797ea81f7c43f0",
            dataset_name="wn18rr_v4_ind_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )

class wn18rr_v4_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59898641?private_link=6a9649f116fe9d8803db",
            dataset_name="wn18rr_v4_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )

class dbpedia100k_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59898644?private_link=d6411dc19831edc2f780",
            dataset_name="dbpedia100k_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )

class fb237_v4_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59900678?private_link=69c356b2701ace402cf9",
            dataset_name="fb237_v4_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )

class fb237_v4_ind_path_queries(NTQueries):
    def __init__(
        self,
        entity_occurrences: Dict = None,
        relation_occurrences: Dict = None,
        global_entity_mapping: Dict = None,
        global_relation_mapping: Dict = None,
        **kwargs
    ):
        super().__init__(
            url="https://figshare.com/ndownloader/files/59900708?private_link=103743702bc11e9b4672",
            dataset_name="fb237_v4_ind_path_queries",
            entity_occurrences=entity_occurrences,
            relation_occurrences=relation_occurrences,
            global_entity_mapping=global_entity_mapping,
            global_relation_mapping=global_relation_mapping,
            type="path",
            **kwargs
        )

if __name__ == "__main__":
    dataset = LUBM()
    proc = pathlib.Path(dataset.processed_dir)
    queries = LUBMQueries(
        entity_occurrences=json.load((proc / "entity_occurrences.json").open("r", encoding="utf-8")),
        relation_occurrences=json.load((proc / "relation_occurrences.json").open("r", encoding="utf-8")),
        global_entity_mapping=json.load((proc / "entity_to_id.json").open("r", encoding="utf-8")),
        global_relation_mapping=json.load((proc / "relation_to_id.json").open("r", encoding="utf-8")),
    )
