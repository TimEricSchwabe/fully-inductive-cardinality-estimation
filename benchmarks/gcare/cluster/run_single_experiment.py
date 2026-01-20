#!/usr/bin/env python3
"""
G-Care Single Experiment Runner

Runs a single G-Care cardinality estimation experiment (for SLURM array jobs).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['experiments']


def get_experiment_by_name(experiments, exp_name):
    """Find experiment by name."""
    for exp in experiments:
        if exp['name'] == exp_name:
            return exp
    return None


def preprocess_experiment(exp, workspace_root):
    """Transform graph and copy queries for a single experiment."""
    print("=" * 80)
    print(f"PREPROCESSING: {exp['name']}")
    print("=" * 80)
    
    gcare_root = workspace_root / "benchmarks/gcare"
    exp_name = exp['name']
    
    # Transform graph
    output_dir = gcare_root / "gcare/data/dataset" / exp_name
    graph_path = Path(exp['graph_path'])
    
    # Skip if already transformed
    if output_dir.exists() and list(output_dir.glob("*.graph")):
        print(f"Graph already transformed, skipping")
    else:
        # Check input exists
        if not graph_path.exists():
            print(f"ERROR: Graph file not found: {graph_path}")
            sys.exit(1)
        
        print(f"Transforming graph...")
        
        try:
            # Import here to use the module in the container
            sys.path.insert(0, str(gcare_root))
            from transform_graph import transform_ttl_to_gcare
            
            transform_ttl_to_gcare(
                graph_path=graph_path,
                rdf_type_uri=exp['rdf_type_uri'],
                dataset_name=exp_name,
                out_path=output_dir
            )
            print(f"Graph transformation done")
            
        except Exception as e:
            print(f"ERROR: Graph transformation failed: {e}")
            sys.exit(1)
    
    # Copy query file
    query_dir = gcare_root / "gcare/data/queryset" / exp_name
    query_dir.mkdir(parents=True, exist_ok=True)
    
    query_path = Path(exp['query_path'])
    
    # Check if query file exists
    if not query_path.exists():
        print(f"ERROR: Query file not found: {query_path}")
        sys.exit(1)
    
    # Copy query JSON file with standardized name
    dest_file = query_dir / f"{exp_name}.json"
    if not dest_file.exists():
        print(f"Copying query file...")
        shutil.copy(query_path, dest_file)
    else:
        print(f"Query file already exists, skipping")
    
    print("Preprocessing completed\n")


def run_experiment(exp, results_dir, workspace_root):
    """Run the cardinality estimation for a single experiment."""
    print("=" * 80)
    print(f"RUNNING EXPERIMENT: {exp['name']}")
    print("=" * 80)
    
    gcare_root = workspace_root / "benchmarks/gcare"
    exp_name = exp['name']
    
    # Create results directory
    exp_results_dir = results_dir / exp_name
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to gcare directory
    import os
    os.chdir(gcare_root)
    
    # Set environment variables
    os.environ['DATASET_NAME'] = exp_name
    os.environ['DATASET_TYPE'] = exp['query_type']
    os.environ['NUM_QUERIES'] = str(exp.get('num_queries', 100))
    os.environ['RDF_TYPE_URI'] = exp['rdf_type_uri']
    # Don't set RESULTS_ROOT - let scripts use default /usr/src/app/results in container
    
    # Build statistics
    print("\n--- Building Statistics ---")
    try:
        subprocess.run(
            [sys.executable, "build_statistics.py"],
            check=True,
            cwd=gcare_root
        )
        print("Statistics built successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Building statistics failed with code {e.returncode}")
        sys.exit(1)
    
    # Run estimation
    print("\n--- Running Estimation ---")
    try:
        subprocess.run(
            [sys.executable, "run_estimation.py"],
            check=True,
            cwd=gcare_root
        )
        print("Estimation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Estimation failed with code {e.returncode}")
        sys.exit(1)
    
    # Move results
    print("\n--- Collecting Results ---")
    # Results are written to /usr/src/app/results/{dataset_type} inside container
    # or {gcare_root}/gcare/data/result when running locally
    
    # Check both possible locations
    results_source_container = Path("/usr/src/app/results")
    results_source_local = gcare_root / "gcare/data/result"
    
    results_source = None
    if results_source_container.exists() and any(results_source_container.iterdir()):
        results_source = results_source_container
    elif results_source_local.exists() and any(results_source_local.iterdir()):
        results_source = results_source_local
    
    if results_source:
        for item in results_source.iterdir():
            dest = exp_results_dir / item.name
            if item.is_file():
                shutil.copy(item, dest)
            else:
                shutil.copytree(item, dest, dirs_exist_ok=True)
        print(f"Results saved to: {exp_results_dir}")
    else:
        print("WARNING: No results directory found")
    
    print("\n" + "=" * 80)
    print(f"✓ Experiment {exp_name} completed successfully!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run a single G-Care experiment")
    parser.add_argument("--experiment", required=True,
                       help="Name of experiment to run (e.g., wikidata_path)")
    parser.add_argument("--config", type=Path, default=Path("experiments_config.yaml"),
                       help="Experiment configuration YAML file")
    parser.add_argument("--results-dir", type=Path,
                       help="Results output directory (default: ./results/experiment_name)")
    parser.add_argument("--workspace-root", type=Path,
                       default=Path(__file__).resolve().parent.parent.parent,
                       help="Workspace root directory")
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load experiments
    experiments = load_config(config_path)
    
    # Find the requested experiment
    exp = get_experiment_by_name(experiments, args.experiment)
    if not exp:
        print(f"ERROR: Experiment '{args.experiment}' not found in config")
        print(f"Available experiments: {[e['name'] for e in experiments]}")
        sys.exit(1)
    
    # Setup results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = Path(__file__).parent / "results" / f"{args.experiment}_{timestamp}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {args.experiment}")
    print(f"Results directory: {results_dir}")
    print("")
    
    # Run the experiment
    preprocess_experiment(exp, args.workspace_root)
    run_experiment(exp, results_dir, args.workspace_root)


if __name__ == "__main__":
    main()
