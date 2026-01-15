import sys
import shutil
from pathlib import Path
import yaml
import json

# Add current directory to sys.path to import local modules
sys.path.append(str(Path(__file__).parent))

from rdf_types import rdf_type_uri_swdf
from transform_graph import transform_ttl_to_neuso
from transform_query import transform_queries_to_neuso

def setup_neuso_benchmark():
    # Paths
    base_dir = Path(__file__).resolve().parent
    neuso_root = base_dir / "NeuSO"
    dataset_name = "swdf_star"
    
    # Input paths (hardcoded based on user context)
    # Assuming relative to workspace root which is ../../ from benchmarks/neuso
    workspace_root = base_dir.parent.parent
    graph_path = workspace_root / "data/swdf/raw/59320634"
    query_dir = workspace_root / "data/swdf_star_queries/raw/59320652"
    
    # Output paths
    neuso_dataset_dir = neuso_root / "dataset" / dataset_name
    neuso_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up NeuSO for dataset: {dataset_name}")
    print(f"Graph Path: {graph_path}")
    print(f"Query Dir: {query_dir}")
    
    # 1. Transform Data Graph
    print("\n--- Transforming Data Graph ---")
    vertex_dict, edge_list, id_map, predicate_map = transform_ttl_to_neuso(
        graph_path=graph_path,
        rdf_type_uri=rdf_type_uri_swdf,
        dataset_name=dataset_name,
        out_dir=neuso_dataset_dir
    )
    
    # Get number of labels for config
    # We can infer it from the max label ID in vertex_dict or from stats
    # transform_graph creates label IDs starting from 1. 0 is default.
    # So max label + 1 is the number of labels (or size of one-hot vector).
    max_label = 0
    for v in vertex_dict.values():
        if v[1] > max_label:
            max_label = v[1]
    
    num_v_labels = max_label + 1 
    print(f"Number of vertex labels: {num_v_labels}")

    # 2. Transform Queries
    print("\n--- Transforming Queries ---")
    # Query input is a directory of JSONs or a single JSON?
    # gcare uses a single JSON file.
    # But the path provided is a directory "59320652".
    # I'll check if there's a JSON file inside.
    query_files = list(query_dir.glob("*.json"))
    if not query_files:
        # Maybe it's not a directory but a file without json extension?
        # Or maybe I should look for the file gcare used: "swdf_star.json"
        # gcare copied it to "queryset/swdf_star/swdf_star.json"
        # I'll look for json files in the directory.
        if query_dir.is_file():
             query_files = [query_dir]
        else:
             print(f"Warning: No query JSON found in {query_dir}. Using gcare queryset if available.")
             gcare_query_path = workspace_root / "benchmarks/gcare/gcare/data/queryset/swdf_star/swdf_star.json"
             if gcare_query_path.exists():
                 query_files = [gcare_query_path]
             else:
                 raise FileNotFoundError(f"Could not find query file.")

    query_file = query_files[0]
    print(f"Using query file: {query_file}")
    
    transform_queries_to_neuso(
        query_path=query_file,
        mappings_dir=neuso_dataset_dir,
        dataset_name=dataset_name,
        out_dir=neuso_dataset_dir
    )
    
    # 3. Update Config
    print("\n--- Updating NeuSO Config ---")
    config_path = neuso_root / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["dataset"] = dataset_name
    # Update dimensions if possible directly here, or rely on parser_config patch
    # The config file has 'in_dim' under model.query_gnn
    # But parser_config.py overwrites it. 
    # I should modify parser_config.py to read from config if set, or I should patch parser_config.py
    
    # Let's write a patched parser_config.py or rely on a new mechanism.
    # I will patch parser_config.py in a separate step.
    # Here I just save the config.
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # Save a metadata file for parser_config to read input dimension
    meta_info = {
        "v_label_num": num_v_labels
    }
    with open(neuso_dataset_dir / "meta.json", "w") as f:
        json.dump(meta_info, f)
        
    print("Setup complete.")

if __name__ == "__main__":
    setup_neuso_benchmark()
