#!/usr/bin/env python3
"""
G-Care Benchmark Experiment Runner

Runs G-Care cardinality estimation experiments in parallel using Docker containers.
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['experiments']


def clear_gcare_directories(workspace_root):
    """Clear queryset and dataset directories before starting."""
    print("Clearing G-Care data directories...")
    gcare_root = workspace_root / "benchmarks/gcare/gcare"
    
    queryset_dir = gcare_root / "data/queryset"
    dataset_dir = gcare_root / "data/dataset"
    
    # Remove and recreate directories
    for dir_path in [queryset_dir, dataset_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Directories cleared\n")


def preprocess_graphs(experiments, workspace_root):
    """Transform graphs to G-Care format. Exit on any failure."""
    print("=" * 80)
    print("PHASE 1: Preprocessing Graphs and Queries")
    print("=" * 80)
    
    gcare_root = workspace_root / "benchmarks/gcare"
    
    # Process each experiment (transform graph + copy queries)
    for exp in experiments:
        exp_name = exp['name']
        
        # Transform graph
        output_dir = gcare_root / "gcare/data/dataset" / exp_name
        graph_path = Path(exp['graph_path'])
        
        # Skip if already transformed
        if output_dir.exists() and list(output_dir.glob("*.graph")):
            print(f"[{exp_name}] Graph already transformed, skipping")
        else:
            # Check input exists
            if not graph_path.exists():
                print(f"ERROR: Graph file not found: {graph_path}")
                sys.exit(1)
            
            print(f"[{exp_name}] Transforming graph...")
            
            try:
                from transform_graph import transform_ttl_to_gcare
                
                transform_ttl_to_gcare(
                    graph_path=graph_path,
                    rdf_type_uri=exp['rdf_type_uri'],
                    dataset_name=exp_name,
                    out_path=output_dir
                )
                print(f"[{exp_name}] Graph transformation done")
                
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
            print(f"[{exp_name}] Copying query file...")
            shutil.copy(query_path, dest_file)
        else:
            print(f"[{exp_name}] Query file already exists, skipping")
    
    print("All preprocessing completed successfully\n")


def run_experiments(experiments, results_dir, workspace_root, max_parallel=5):
    """Run experiments in parallel using Docker containers."""
    print("=" * 80)
    print("PHASE 2: Running Experiments")
    print("=" * 80)
    
    gcare_root = workspace_root / "benchmarks/gcare"
    gcare_data = gcare_root / "gcare"
    
    # Prepare all experiments
    running = []
    pending = list(experiments)
    completed = 0
    failed = []
    
    start_time = time.time()
    
    while pending or running:
        # Start new containers if slots available
        while pending and len(running) < max_parallel:
            exp = pending.pop(0)
            name = exp['name']
            
            # Create flat results directory
            exp_results_dir = results_dir / f"{name}"
            exp_results_dir.mkdir(parents=True, exist_ok=True)
            
            container_name = f"gcare-{name}-{int(time.time())}"
            
            # Build docker run command
            cmd = [
                "docker", "run",
                "--name", container_name,
                "--rm",
                "-e", f"DATASET_NAME={name}",
                "-e", f"DATASET_TYPE={exp['query_type']}",
                "-e", f"NUM_QUERIES={exp.get('num_queries', 100)}",
                "-e", f"RDF_TYPE_URI={exp['rdf_type_uri']}",
                "-v", f"{(gcare_data / 'data/queryset').absolute()}:/usr/src/app/gcare/data/queryset",
                "-v", f"{(gcare_data / 'data/dataset').absolute()}:/usr/src/app/gcare/data/dataset",
                "-v", f"{exp_results_dir.absolute()}:/usr/src/app/results",
                "gcare-benchmark:latest",
                "sh", "-c", "python3 ./build_statistics.py && python3 ./run_estimation.py"
            ]
            
            print(f"[{name}] Starting container {container_name}")
            
            try:
                proc = subprocess.Popen(cmd, cwd=gcare_root)
                running.append({
                    'name': name,
                    'container': container_name,
                    'process': proc,
                    'timeout': exp.get('timeout_hours', 2) * 3600,
                    'start_time': time.time()
                })
            except Exception as e:
                print(f"ERROR: Failed to start container for {name}: {e}")
                failed.append(name)
                # Cleanup and exit
                cleanup_containers([r['container'] for r in running])
                sys.exit(1)
        
        # Check running containers
        for run_info in running[:]:
            proc = run_info['process']
            poll_result = proc.poll()
            
            # Check timeout
            elapsed = time.time() - run_info['start_time']
            if elapsed > run_info['timeout']:
                print(f"ERROR: [{run_info['name']}] Timeout after {run_info['timeout']/3600:.1f}h")
                proc.kill()
                cleanup_containers([run_info['container']])
                failed.append(run_info['name'])
                running.remove(run_info)
                # Exit on first failure
                cleanup_containers([r['container'] for r in running])
                sys.exit(1)
            
            # Check if finished
            if poll_result is not None:
                running.remove(run_info)
                
                if poll_result == 0:
                    print(f"[{run_info['name']}] Completed successfully")
                    completed += 1
                else:
                    print(f"ERROR: [{run_info['name']}] Container failed with code {poll_result}")
                    cleanup_containers([run_info['container']])
                    failed.append(run_info['name'])
                    # Exit on first failure
                    cleanup_containers([r['container'] for r in running])
                    sys.exit(1)
        
        # Brief sleep to avoid busy loop
        if running:
            time.sleep(2)
    
    total_time = time.time() - start_time
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed: {completed}")
    print(f"Runtime: {total_time / 3600:.2f} hours")
    print(f"Results: {results_dir}")
    print("=" * 80)


def cleanup_containers(container_names):
    """Force remove Docker containers."""
    for name in container_names:
        try:
            subprocess.run(["docker", "rm", "-f", name], 
                         capture_output=True, timeout=10)
        except:
            pass


def build_docker_image(gcare_root):
    """Build the Docker image."""
    print("=" * 80)
    print("Building Docker Image")
    print("=" * 80)
    
    result = subprocess.run(
        ["docker", "build", "-t", "gcare-benchmark:latest", "."],
        cwd=gcare_root,
        capture_output=False
    )
    
    if result.returncode != 0:
        print("ERROR: Docker build failed")
        sys.exit(1)
    
    print("Docker image built successfully\n")


def main():
    parser = argparse.ArgumentParser(description="Run G-Care benchmark experiments")
    parser.add_argument("--config", type=Path, default=Path("experiments_config.yaml"),
                       help="Experiment configuration YAML file")
    parser.add_argument("--max-parallel", type=int, default=5,
                       help="Maximum parallel Docker containers (default: 5)")
    parser.add_argument("--workspace-root", type=Path,
                       default=Path(__file__).resolve().parent.parent.parent,
                       help="Workspace root directory")
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Setup
    workspace_root = args.workspace_root
    gcare_root = workspace_root / "benchmarks/gcare"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = gcare_root / "results" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory: {results_dir}\n")
    
    # Load experiments
    experiments = load_config(args.config)
    print(f"Loaded {len(experiments)} experiments\n")
    
    # Build Docker image
    build_docker_image(gcare_root)
    
    # Clear directories and run
    clear_gcare_directories(workspace_root)
    preprocess_graphs(experiments, workspace_root)
    run_experiments(experiments, results_dir, workspace_root, args.max_parallel)
    
    print("\n✓ All experiments completed successfully!")


if __name__ == "__main__":
    main()
