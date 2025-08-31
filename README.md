# 📊 Fully Inductive Cardinality Estimator

## 📂 Datasets

### `NTDatasets()`
`NTDtasets()` is a Dataset-class that inherits from `torch_geometric.InMemoryDataset` and reads in and processses a `.nt` datasets. It returns a `torch_geometric.Data` obejct with the following structure. Additionally in maps the RDF URIs to unique indices and stores them inside of `.json` file.
```
# G = (V, E, R), V = Nodes, E = Edges, R = Relations  
Data(
  edge_index=[2, |E|],
  edge_type=[|E|],
  num_nodes=|V|,
  num_relations=|R|,
  occurrences_relations=[|R|, 3],
  occurrences_nodes=[|V|, 3],
  relation_graph=Data(
    edge_index=[2, n],
    edge_type=[n],
    num_nodes=|V|,
    num_relations=4
  )
)
```
**Top-level `Data(...)` (entity graph)**

- **edge_index = [2, |E|]** — A 2×E tensor of directed edges between nodes. Row 0 has source (head) node IDs; row 1 has target (tail) node IDs.
- **edge_type = [|E|]** — Length-E tensor giving the relation ID (integer in `[0, num_relations-1]`) for each edge in `edge_index` (same column order).
- **num_nodes = |V|** — Number of entity nodes.
- **num_relations = |R|** — Number of distinct relation types used in `edge_type`.
- **occurrences_relations = [|R|, 3]** — For each relation ID, counts of how often it appears across three buckets. In most pipelines these 3 columns are **[train, validation, test]** triple counts for that relation.
- **occurrences_nodes = [|V|, 3]** — For each node ID, counts of how often the node appears in triples across the same three buckets (typically **[train, validation, test]**). These counts usually aggregate both head and tail appearances.

**`relation_graph = Data(...)` (meta-graph over relations)**

- **num_nodes = |R|** — One node per relation from the main graph.
- **edge_index = [2, n]** — Directed edges between relation-nodes.
- **edge_type = [n]** — Length-n tensor giving the **meta-relation** type between two relations for each edge.
- **num_relations = 4** — h2h, t2t, h2t, t2h


### How to Use
1. **Download your dataset:** Download your `.nt` file from a source like the one provided here: [https://nx36303.your-storageshare.de/s/wMbJJ2JLnkXcSE6](https://nx36303.your-storageshare.de/s/wMbJJ2JLnkXcSE6)

2. **Create the folder structure:** Organize your dataset with the following hierarchy:
```
your-dataset/
└── <dataset-name>/
    └── raw/
        └── <dataset>.nt
```

3. **Process the dataset:** The `NTDatasets()` class automatically processes the .nt file, counting occurrences and creating a mapping in JSON format. This processed data will be saved in a new folder named `processed/`, which will be created in the same directory as `raw/`.
```python
    root = "your-dataset/<dataset-name>/"
    nt_path =  "your-dataset/<dataset-name>/raw/<dataset>.nt"

    dataset = datasets.NTDatasets(root=root, nt_path=nt_path)
    dataset = dataset[0]
```



### ✅ Example

#### Folder Structure

```
data/
└── swdf/
    └── raw/
        └── swdf.nt
```


#### Code
```python
    root = "data/swdf/"
    nt_path =  "data/swdf/raw/swdf.nt"

    dataset = datasets.NTDatasets(root=root, nt_path=nt_path)
    dataset = dataset[0]
```
