
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import collections
import numpy as np

def transform_ttl_to_neuso(
    graph_path: Path,
    rdf_type_uri: str,
    dataset_name: str,
    out_dir: Path
) -> Tuple[Dict, List, Dict, Dict]:
    """
    Transform a TTL file to NeuSO format. 
    
    Args:
        graph_path: Path to the input TTL file
        rdf_type_uri: URI that corresponds to rdf:type for this dataset
        dataset_name: Name of the dataset
        out_dir: Output directory path where NeuSO files will be written
    
    Returns:
        Tuple of (vertex_dict, edge_list, id_to_id_mapping, id_to_id_mapping_predicate)
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    # Prepare output directories
    data_graph_dir = out_dir / "data_graph"
    data_graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Sets of entities and predicates
    vertices: Set[str] = set()
    predicates: Set[str] = set()

    print(f"Reading graph from {graph_path}...")
    with open(graph_path, "r") as ttl_file:
        graph_data = ttl_file.readlines()

    # Pass 1: Extract vertices and predicates
    print("Pass 1: Extracting vertices and predicates...")
    for line in tqdm(graph_data):
        atoms = line.strip().split(" ")
        # Basic TTL parsing: sub pred obj .
        if len(atoms) >= 3:
            vertices.add(atoms[0])
            vertices.add(atoms[2])
            predicates.add(atoms[1])
    
    # Create ID mappings
    id_to_id_mapping: Dict[str, int] = {}
    id_to_id_mapping_predicate: Dict[str, int] = {}
    
    # Vertex mapping
    print("Creating vertex mappings...")
    vertex_dict: Dict[str, List[int]] = {} # uri -> [id, label]
    vid = 0
    for vertex in tqdm(vertices):
        # Default label is 0
        vertex_dict[vertex] = [vid, 0]
        id_to_id_mapping[vertex] = vid
        vid += 1
    
    # Predicate mapping
    print("Creating predicate mappings...")
    pid = 0
    for p in predicates:
        id_to_id_mapping_predicate[p] = pid
        pid += 1
    
    # Vertex Label Mapping (URI -> int ID)
    # We need to map RDF types to integer labels.
    # NeuSO expects integer labels.
    # We will treat rdf:type objects as labels.
    label_to_id: Dict[str, int] = {}
    next_label_id = 1 # 0 is reserved for default/unknown
    
    # Pass 2: Process types (labels)
    print("Pass 2: Processing vertex types...")
    for line in tqdm(graph_data):
        atoms = line.strip().split(" ")
        if len(atoms) >= 3:
            s, p, o = atoms[0], atoms[1], atoms[2]
            if p == rdf_type_uri:
                if o not in label_to_id:
                    label_to_id[o] = next_label_id
                    next_label_id += 1
                
                # Assign label to vertex. 
                # NeuSO supports one label per vertex in its simple format.
                # We overwrite if multiple exists (or could pick first).
                # Here we overwrite, so the last one wins.
                if s in vertex_dict:
                    vertex_dict[s][1] = label_to_id[o]

    # Pass 3: Create Edge List and Collect Stats
    print("Pass 3: Creating edge list and collecting stats...")
    edge_list: List[List[int]] = []
    
    # Stats for filter generation
    v_label_counts = collections.defaultdict(int)
    e_label_counts = collections.defaultdict(int)
    
    # Count vertex labels
    for v_data in vertex_dict.values():
        v_label_counts[v_data[1]] += 1
        
    for line in tqdm(graph_data):
        atoms = line.strip().split(" ")
        if len(atoms) >= 3:
            s, p, o = atoms[0], atoms[1], atoms[2]
            
            # Skip type definitions in edge list if they are treated as labels
            # But NeuSO graph file contains structural edges. 
            # Usually RDF type triples are NOT structural edges in subgraph matching if handled as labels.
            if p == rdf_type_uri:
                continue
                
            if s in vertex_dict and o in vertex_dict:
                s_id = vertex_dict[s][0]
                o_id = vertex_dict[o][0]
                p_id = id_to_id_mapping_predicate[p]
                
                edge_list.append([s_id, o_id, p_id])
                e_label_counts[p_id] += 1

    # Write Data Graph File
    # Format:
    # d node_num edge_num v_label_num e_label_num
    # v v_id v_label
    # ...
    # e src_id dst_id e_label
    # ...
    
    output_file = data_graph_dir / f"{dataset_name}.graph"
    print(f"Writing NeuSO graph to {output_file}...")
    
    node_num = len(vertex_dict)
    edge_num = len(edge_list)
    v_label_num = next_label_id
    e_label_num = len(id_to_id_mapping_predicate)
    
    with open(output_file, "w") as f:
        f.write(f"d {node_num} {edge_num} {v_label_num} {e_label_num}\n")
        
        # Sort by ID for deterministic output
        sorted_vertices = sorted(vertex_dict.items(), key=lambda item: item[1][0])
        for uri, data in sorted_vertices:
            f.write(f"v {data[0]} {data[1]}\n")
            
        for e in edge_list:
            f.write(f"e {e[0]} {e[1]} {e[2]}\n")

    # Save Mappings and Stats
    print("Saving mappings and stats...")
    
    with open(out_dir / "id_to_id_mapping.json", "w") as f:
        json.dump(id_to_id_mapping, f)
        
    with open(out_dir / "id_to_id_mapping_predicate.json", "w") as f:
        json.dump(id_to_id_mapping_predicate, f)
        
    with open(out_dir / "label_to_id_mapping.json", "w") as f:
        json.dump(label_to_id, f)
        
    stats = {
        "v_label_counts": v_label_counts,
        "e_label_counts": e_label_counts
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    # Generate random label embeddings (placeholder for future proper embeddings)
    print("Generating random label embeddings...")
    embedding_dim = 128  # From NeuSO config: topology_emb_dim
    num_labels = v_label_num
    
    # Generate random embeddings: one vector per vertex label
    # Shape: [num_labels, embedding_dim]
    np.random.seed(42)  # For reproducibility
    label_embeddings = np.random.randn(num_labels, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    norms = np.linalg.norm(label_embeddings, axis=1, keepdims=True)
    label_embeddings = label_embeddings / (norms + 1e-10)
    
    # Save embeddings
    embeddings_file = data_graph_dir / f"{dataset_name}.emb.npy"
    np.save(embeddings_file, label_embeddings)
    print(f"Saved label embeddings to {embeddings_file}")
    print(f"Embedding shape: {label_embeddings.shape}")

    print("Transformation complete!")
    return vertex_dict, edge_list, id_to_id_mapping, id_to_id_mapping_predicate
