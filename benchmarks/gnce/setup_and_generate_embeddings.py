import yaml
import json
import shutil
from pathlib import Path
from collections import defaultdict
from generate_embeddings import generate_embeddings_for_dataset


def main():
    config_path = Path(__file__).parent / "experiments_config.yaml"
    base_output_dir = Path(__file__).parent / "data"
    
    # Load config
    with open(config_path, 'r') as f:
        experiments = yaml.safe_load(f)['experiments']
    
    # Group by dataset
    datasets = defaultdict(lambda: {'query_types': {}})
    for exp in experiments:
        dataset_name = exp['dataset_name']
        datasets[dataset_name]['graph_path'] = exp['graph_path'].split('?')[0]
        datasets[dataset_name]['query_types'][exp['query_type']] = exp['query_path'].split('?')[0]
    
    print(f"Found {len(datasets)} datasets")
    
    # Setup data structure and generate embeddings
    for dataset_name, info in datasets.items():
        print(f"\n=== {dataset_name} ===")
        
        dataset_dir = base_output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        statistics_dir = dataset_dir / "statistics"
        statistics_dir.mkdir(exist_ok=True)
        
        # Copy graph file
        graph_src = Path(info['graph_path'])
        graph_dst = dataset_dir / "graph.nt"
        if graph_src.exists():
            shutil.copy2(graph_src, graph_dst)
            print(f"Copied graph: {graph_src.name}")
        
        # Copy query files
        query_files = []
        for query_type, query_src_path in info['query_types'].items():
            query_type_dir = dataset_dir / query_type
            query_type_dir.mkdir(exist_ok=True)
            query_src = Path(query_src_path)
            query_dst = query_type_dir / "Joined_Queries.json"
            if query_src.exists():
                shutil.copy2(query_src, query_dst)
                query_files.append(str(query_dst))
                print(f"Copied {query_type} queries")
        
        # Generate embeddings
        if graph_dst.exists() and query_files:
            existing = list(statistics_dir.glob("*.json"))
            if existing:
                print(f"Skipping embeddings - {len(existing)} already exist")
            else:
                print("Generating embeddings...")
                try:
                    generate_embeddings_for_dataset(
                        graph_file=str(graph_dst),
                        query_files=query_files,
                        output_statistics_dir=str(statistics_dir)
                    )
                    print("Done")
                except Exception as e:
                    print(f"Error: {e}")
    
    print(f"\nOutput: {base_output_dir}")


if __name__ == "__main__":
    main()
