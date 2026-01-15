"""
Transform RDF/TTL files to G-Care format.

This module provides functionality to convert TTL files with URIs containing IDs
to the G-Care graph format (.txt file) used by the G-Care benchmark.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm


def transform_ttl_to_gcare(
    graph_path: Path,
    rdf_type_uri: str,
    dataset_name: str,
    out_path: Path
) -> Tuple[Dict, List, Dict, Dict]:
    """
    Transform a TTL file to G-Care format.
    
    Args:
        graph_path: Path to the input TTL file
        rdf_type_uri: URI that corresponds to rdf:type for this dataset
        dataset_name: Name of the dataset (used for output file naming)
        out_path: Output directory path where G-Care files will be written
    
    Returns:
        Tuple of (vertex_dict, edge_list, id_to_id_mapping, id_to_id_mapping_predicate)
    
    Raises:
        FileNotFoundError: If graph_path does not exist
        AssertionError: If vertex dictionary validation fails
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    # Ensure output directory exists
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Sets of entities and predicates
    vertices: Set[str] = set()
    predicates: Set[str] = set()

    with open(graph_path, "r") as ttl_file:
        graph_data = ttl_file.readlines()

    # Pass 1: Add all entities and predicates from the .ttl file
    print("Pass 1: Extracting vertices and predicates...")
    for line in tqdm(graph_data):
        atoms = line.split(" ")[:-1]
        if atoms:
            vertices.add(atoms[0])
            vertices.add(atoms[2])
            predicates.add(atoms[1])
    
    # Create ID mappings
    id_to_id_mapping: Dict[str, int] = {}
    id_to_id_mapping_predicate: Dict[str, int] = {}
    
    # Creating Vertex Dict and save entity mappings
    print("Creating vertex mappings...")
    vertex_dict: Dict[str, List[int]] = {}
    vid = 0
    for vertex in tqdm(vertices):
        # dvid = vertex.split("/")[-1].replace(">", "")
        #vertex_dict[vertex] = [dvid]
        vertex_dict[vertex] = [vid]
        id_to_id_mapping[vertex] = vid
        vid += 1
    
    # Saving Mappings for Predicates
    print("Creating predicate mappings...")
    pid = 0
    for p in predicates:
        id_to_id_mapping_predicate[p] = pid
        pid += 1
    
    # Pass 2: Add Vertex types (rdf:type)
    print("Pass 2: Adding vertex types...")
    for line in tqdm(graph_data):
        atoms = line.split(" ")[:-1]
        # If triple has predicate rdf:type
        if atoms and atoms[1] == rdf_type_uri:
            type_id = id_to_id_mapping[atoms[2]]
            vertex_dict[atoms[0]].append(type_id)
    
    # Add default type (0) to vertices without explicit types
    print("Adding default types to vertices without explicit types...")
    for v in vertex_dict.values():
        if len(v) == 1:
            v.append(0)
    
    # Validate: all entries should have length 2 or more
    all_entries_with_length_2_or_more = all([len(v) >= 2 for v in vertex_dict.values()])
    assert all_entries_with_length_2_or_more, "Some vertices don't have type information"
    print(f"Total vertices: {len(vertex_dict)}")
    
    # Pass 3: Creating Edge List
    print("Pass 3: Creating edge list...")
    edge_list: List[List[int]] = []
    for line in tqdm(graph_data):
        tp = line.split(" ")[:-1]
        if tp:
            s = tp[0]
            p = tp[1]
            o = tp[2]
            edge_list.append([
                vertex_dict[s][0],
                vertex_dict[o][0],
                id_to_id_mapping_predicate[p]
            ])
    
    print(f"Total edges: {len(edge_list)}")
    
    # Writing the Data File
    print("Writing G-Care format file...")
    output_file = out_path / f"{dataset_name}.txt"
    with open(output_file, "w") as f:
        f.write("t # 1\n")
        for v in tqdm(vertex_dict, desc="Writing vertices"):
            line = "v " + " ".join(map(str, vertex_dict[v]))
            f.write(line + "\n")
        for e in tqdm(edge_list, desc="Writing edges"):
            f.write(f"e {e[0]} {e[1]} {e[2]}\n")
    
    print(f"G-Care graph written to: {output_file}")
    
    # Save ID to ID mappings for later query transformation
    print("Saving ID mappings...")
    mapping_file = out_path / "id_to_id_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(id_to_id_mapping, f)
    print(f"Entity mapping written to: {mapping_file}")
    
    predicate_mapping_file = out_path / "id_to_id_mapping_predicate.json"
    with open(predicate_mapping_file, "w") as f:
        json.dump(id_to_id_mapping_predicate, f)
    print(f"Predicate mapping written to: {predicate_mapping_file}")
    
    print("Transformation complete!")
    
    return vertex_dict, edge_list, id_to_id_mapping, id_to_id_mapping_predicate
