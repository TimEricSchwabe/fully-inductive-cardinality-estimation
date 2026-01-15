#!/usr/bin/env python3
"""
Data preparation script for NeuSO benchmark.
Transforms TTL graph and query JSON files into NeuSO format.
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from benchmarks.neuso.rdf_types import *
from benchmarks.neuso.transform_graph import transform_ttl_to_neuso
from benchmarks.neuso.transform_query import transform_queries_to_neuso


# Mapping of dataset names to their rdf_type URIs
RDF_TYPE_MAPPING = {
    'yago': rdf_type_uri_yago,
    'lubm': rdf_type_uri_lubm,
    'wn18rr': rdf_type_uri_wn18rr,
    'wn18rr_ind': rdf_type_uri_wn18rr,
    'dbpedia100k': rdf_type_uri_dbpedia100k,
    'wikidata': rdf_type_uri_wikidata,
    'swdf': rdf_type_uri_swdf,
    'swdf_star': rdf_type_uri_swdf,
}


def get_rdf_type_uri(dataset_name: str, custom_uri: str = None) -> str:
    """Get the rdf:type URI for a dataset."""
    if custom_uri:
        return custom_uri
    
    # Try to match dataset name
    for key, uri in RDF_TYPE_MAPPING.items():
        if key in dataset_name.lower():
            return uri
    
    # Default fallback
    print(f"Warning: Could not auto-detect rdf:type URI for '{dataset_name}'")
    print("Using SWDF default. Use --rdf-type-uri to specify custom URI.")
    return rdf_type_uri_swdf


def update_neuso_config(dataset_name: str, neuso_root: Path):
    """Update NeuSO config.yaml with dataset name."""
    config_path = neuso_root / "config.yaml"
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update dataset name
    config['dataset'] = dataset_name
    if torch.cuda.is_available():
        config['device']  = 'cuda'
    elif torch.backends.mps.is_available():
        config['device']  = 'mps'
    else:
        config['device'] = 'cpu'

    # Ensure topology_emb is set (we generate random embeddings)
    if 'data_loader' not in config:
        config['data_loader'] = {}
    
    # Can use either topology_emb mode
    # False = one-hot encoding (simpler)
    # True = use the random embeddings we generated
    config['data_loader']['topology_emb'] = True
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Updated config.yaml: dataset='{dataset_name}', topology_emb=True")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for NeuSO benchmark"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'swdf_star')"
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        required=True,
        help="Path to the TTL graph file"
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        required=True,
        help="Path to the query JSON file"
    )
    parser.add_argument(
        "--rdf-type-uri",
        type=str,
        default=None,
        help="Custom rdf:type URI (auto-detected if not provided)"
    )
    parser.add_argument(
        "--neuso-root",
        type=Path,
        default=None,
        help="Path to NeuSO root directory (default: ./NeuSO)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    neuso_root = args.neuso_root if args.neuso_root else script_dir / "NeuSO"
    
    if not neuso_root.exists():
        print(f"Error: NeuSO directory not found at {neuso_root}")
        sys.exit(1)
    
    # Get rdf:type URI
    rdf_type_uri = get_rdf_type_uri(args.dataset_name, args.rdf_type_uri)
    print(f"Using rdf:type URI: {rdf_type_uri}")
    
    # Prepare output directory
    dataset_dir = neuso_root / "dataset" / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Preparing NeuSO dataset: {args.dataset_name}")
    print(f"{'='*60}")
    print(f"Graph: {args.graph_path}")
    print(f"Queries: {args.query_path}")
    print(f"Output: {dataset_dir}")
    print(f"{'='*60}\n")
    
    # Step 1: Transform graph
    print("\n--- Step 1: Transforming Data Graph ---")
    try:
        vertex_dict, edge_list, id_map, predicate_map = transform_ttl_to_neuso(
            graph_path=args.graph_path,
            rdf_type_uri=rdf_type_uri,
            dataset_name=args.dataset_name,
            out_dir=dataset_dir
        )
        print(f"✓ Graph transformation complete")
        print(f"  - Vertices: {len(vertex_dict)}")
        print(f"  - Edges: {len(edge_list)}")
        print(f"  - Predicates: {len(predicate_map)}")
    except Exception as e:
        print(f"✗ Graph transformation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Transform queries
    print("\n--- Step 2: Transforming Queries ---")
    try:
        transform_queries_to_neuso(
            query_path=args.query_path,
            mappings_dir=dataset_dir,
            dataset_name=args.dataset_name,
            out_dir=dataset_dir,
            rdf_type_uri=rdf_type_uri
        )
        print(f"✓ Query transformation complete")
    except Exception as e:
        print(f"✗ Query transformation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Update config
    print("\n--- Step 3: Updating NeuSO Config ---")
    try:
        update_neuso_config(args.dataset_name, neuso_root)
        print(f"✓ Config update complete")
    except Exception as e:
        print(f"Warning: Config update failed: {e}")
    
    # Step 4: Verify files
    print("\n--- Step 4: Verifying Generated Files ---")
    required_files = [
        dataset_dir / "data_graph" / f"{args.dataset_name}.graph",
        dataset_dir / "data_graph" / f"{args.dataset_name}.emb.npy",
        dataset_dir / "id_to_id_mapping.json",
        dataset_dir / "id_to_id_mapping_predicate.json",
        dataset_dir / "label_to_id_mapping.json",
        dataset_dir / "stats.json",
    ]
    
    all_good = True
    for file_path in required_files:
        if file_path.exists():
            print(f"  ✓ {file_path.name}")
        else:
            print(f"  ✗ {file_path.name} - MISSING")
            all_good = False
    
    # Check query files
    query_dir = dataset_dir / "query_graph"
    if query_dir.exists():
        query_files = list(query_dir.glob("*.graph"))
        print(f"  ✓ {len(query_files)} query files generated")
    else:
        print(f"  ✗ query_graph directory - MISSING")
        all_good = False
    
    if all_good:
        print(f"\n{'='*60}")
        print(f"✓ Data preparation completed successfully!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"  1. Train the model: python train_neuso.py --dataset {args.dataset_name}")
        print(f"  2. Run inference: python run_estimation.py --dataset {args.dataset_name}")
    else:
        print(f"\n{'='*60}")
        print(f"✗ Data preparation completed with errors")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
