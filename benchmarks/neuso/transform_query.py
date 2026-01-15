import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import collections


def load_mappings_and_stats(mappings_dir: Path) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Load entity, predicate, and label mappings along with statistics.
    
    Args:
        mappings_dir: Directory containing mapping JSON files
        
    Returns:
        Tuple of (entity_mapping, predicate_mapping, label_mapping, 
                  v_label_counts, e_label_counts)
    """
    with open(mappings_dir / "id_to_id_mapping.json", "r") as f:
        entity_mapping = json.load(f)
        
    with open(mappings_dir / "id_to_id_mapping_predicate.json", "r") as f:
        predicate_mapping = json.load(f)
        
    with open(mappings_dir / "label_to_id_mapping.json", "r") as f:
        label_mapping = json.load(f)
        
    with open(mappings_dir / "stats.json", "r") as f:
        stats = json.load(f)
        v_label_counts = stats["v_label_counts"]
        e_label_counts = stats["e_label_counts"]
    
    return entity_mapping, predicate_mapping, label_mapping, v_label_counts, e_label_counts


def parse_query_to_graph(
    triples: List[List[str]],
    predicate_mapping: Dict,
    label_mapping: Dict,
    rdf_type_uri: str = None
) -> Tuple[Dict[int, int], List[Tuple[int, int, int]], Dict[str, int]]:
    """
    Parse query triples into a graph structure with vertices and edges.
    
    Args:
        triples: List of triples from query (each triple is [s, p, o, ...])
        predicate_mapping: Mapping of predicates to IDs
        label_mapping: Mapping of labels/types to IDs
        rdf_type_uri: URI for rdf:type predicate (optional)
        
    Returns:
        Tuple of (query_vertices, query_edges, var_to_id)
        - query_vertices: Dict mapping vertex ID to label ID
        - query_edges: List of (src_id, dst_id, predicate_id) tuples
        - var_to_id: Mapping of variable names to vertex IDs
    """
    var_to_id = {}
    next_var_id = 0
    query_vertices = {}  # id -> label
    query_edges = []     # (src, dst, label)
    
    def get_var_id(v: str) -> int:
        """Get or create a vertex ID for a variable."""
        nonlocal next_var_id
        if v not in var_to_id:
            var_to_id[v] = next_var_id
            next_var_id += 1
        return var_to_id[v]
    
    # Process each triple
    for t in triples:
        s, p, o = t[0], t[1], t[2]
        
        # Check if this is a type definition triple
        is_type_triple = False
        if rdf_type_uri and p == rdf_type_uri:
            is_type_triple = True
        elif o in label_mapping and not o.startswith("?"):
            # Object is a known label (type) and not a variable
            is_type_triple = True
        
        if is_type_triple:
            # This is a type definition: assign label to subject
            s_id = get_var_id(s)
            if o in label_mapping:
                query_vertices[s_id] = label_mapping[o]
            else:
                query_vertices[s_id] = 0  # Default label
                
        elif p in predicate_mapping:
            # Regular edge: create structural edge
            s_id = get_var_id(s)
            o_id = get_var_id(o)
            p_id = predicate_mapping[p]
            query_edges.append((s_id, o_id, p_id))
    
    # Ensure all vertices have a label (default 0)
    for vid in range(next_var_id):
        if vid not in query_vertices:
            query_vertices[vid] = 0
    
    return query_vertices, query_edges, var_to_id


def write_query_graph_file(
    output_path: Path,
    query_vertices: Dict[int, int],
    query_edges: List[Tuple[int, int, int]]
):
    """
    Write query graph to NeuSO .graph format.
    
    Format:
        q node_num edge_num
        v v_id v_label
        ...
        e src_id dst_id e_label
        ...
    
    Args:
        output_path: Path to output .graph file
        query_vertices: Dict mapping vertex ID to label ID
        query_edges: List of (src_id, dst_id, predicate_id) tuples
    """
    node_num = len(query_vertices)
    edge_num = len(query_edges)
    
    with open(output_path, "w") as f:
        f.write(f"q {node_num} {edge_num}\n")
        for v in range(node_num):
            f.write(f"v {v} {query_vertices[v]}\n")
        for s, d, l in query_edges:
            f.write(f"e {s} {d} {l}\n")


def write_filter_file(
    output_path: Path,
    query_vertices: Dict[int, int],
    query_edges: List[Tuple[int, int, int]],
    v_label_counts: Dict,
    e_label_counts: Dict
):
    """
    Write filter file with cardinality information.
    
    Format:
        f node_num edge_num
        v v_id C(v_id)
        ...
        e src_id dst_id C(src_id, dst_id)
        ...
    
    Args:
        output_path: Path to output .filter file
        query_vertices: Dict mapping vertex ID to label ID
        query_edges: List of (src_id, dst_id, predicate_id) tuples
        v_label_counts: Dict with counts for each vertex label
        e_label_counts: Dict with counts for each edge label
    """
    node_num = len(query_vertices)
    edge_num = len(query_edges)
    
    with open(output_path, "w") as f:
        f.write(f"f {node_num} {edge_num}\n")
        for v in range(node_num):
            label = query_vertices[v]
            count = v_label_counts.get(str(label), 0)
            f.write(f"v {v} {count}\n")
        for s, d, l in query_edges:
            count = e_label_counts.get(str(l), 0)
            f.write(f"e {s} {d} {count}\n")


def write_ccg_file(
    output_path: Path,
    num_nodes: int
):
    """
    Write placeholder CCG (Cardinality-Cost Graph) file.
    
    Note: This generates a minimal valid CCG with dummy values.
    Real CCG files would contain ground truth cardinalities and costs.
    
    Format:
        t node_num edge_num
        v v_id v_card v_min_cost v_v_num v_1 v_2 ...
        ...
    
    Args:
        output_path: Path to output .ccg file
        num_nodes: Number of nodes in the query graph
    """
    with open(output_path, "w") as f:
        # Dummy CCG: 1 node representing the full query, 0 edges
        # v_id=0, card=1, cost=1, subquery includes all nodes
        f.write(f"t 1 0\n")
        nodes_str = " ".join(map(str, range(num_nodes)))
        f.write(f"v 0 1 1 {num_nodes} {nodes_str}\n")


def transform_queries_to_neuso(
    query_path: Path,
    mappings_dir: Path,
    dataset_name: str,
    out_dir: Path,
    rdf_type_uri: str = None
):
    """
    Transform queries to NeuSO format (.graph, .filter, .ccg).
    
    This function orchestrates the transformation of query JSON files into
    NeuSO's required formats. It generates three types of files per query:
    - .graph: Query structure (vertices and edges)
    - .filter: Cardinality filters from data graph statistics
    - .ccg: Placeholder Cardinality-Cost Graph files
    
    Args:
        query_path: Path to the query file (JSON format)
        mappings_dir: Directory containing id and label mappings
        dataset_name: Name of the dataset
        out_dir: Output directory
        rdf_type_uri: URI for rdf:type predicate (optional, for type detection)
    """
    # Load mappings and statistics
    print("Loading mappings...")
    entity_mapping, predicate_mapping, label_mapping, v_label_counts, e_label_counts = \
        load_mappings_and_stats(mappings_dir)

    # Load queries
    print(f"Loading queries from {query_path}...")
    with open(query_path, "r") as f:
        queries = json.load(f)
        
    # Prepare output directories
    query_graph_dir = out_dir / "query_graph"
    filter_dir = out_dir / "filter"
    ccg_dir = out_dir / "ccg"
    
    query_graph_dir.mkdir(parents=True, exist_ok=True)
    filter_dir.mkdir(parents=True, exist_ok=True)
    ccg_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each query
    print(f"Processing {len(queries)} queries...")
    for idx, query_obj in enumerate(queries):
        triples = query_obj["triples"]
        query_name = f"query_{idx}"
        
        # Parse query into graph structure
        query_vertices, query_edges, var_to_id = parse_query_to_graph(
            triples, predicate_mapping, label_mapping, rdf_type_uri
        )
        
        # Write query graph file
        write_query_graph_file(
            query_graph_dir / f"{query_name}.graph",
            query_vertices,
            query_edges
        )
        
        # Write filter file
        write_filter_file(
            filter_dir / f"{query_name}.filter",
            query_vertices,
            query_edges,
            v_label_counts,
            e_label_counts
        )
        
        # Write CCG file
        write_ccg_file(
            ccg_dir / f"{query_name}.ccg",
            len(query_vertices)
        )

    print("Query transformation complete.")
