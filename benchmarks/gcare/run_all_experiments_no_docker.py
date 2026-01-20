#!/usr/bin/env python3
"""
G-Care Benchmark Experiment Runner (No Docker)

Runs G-Care cardinality estimation experiments sequentially using direct Python execution.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse common functions from Docker version
from run_all_experiments import load_config, clear_gcare_directories, preprocess_graphs


def run_experiments(experiments, results_dir, workspace_root):
    """Run experiments sequentially using direct Python execution."""
    print("=" * 80)
    print("PHASE 2: Running Experiments")
    print("=" * 80)
    
    gcare_root = workspace_root / "benchmarks/gcare"
    gcare_dir = gcare_root / "gcare"
    
    # Add gcare directory to path for imports
    sys.path.insert(0, str(gcare_root))
    
    completed = 0
    start_time = time.time()
    
    for exp in experiments:
        name = exp['name']
        
        # Create flat results directory
        exp_results_dir = results_dir / f"{name}"
        exp_results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{name}] Starting experiment")
        
        # Set environment variables
        os.environ['DATASET_NAME'] = name
        os.environ['DATASET_TYPE'] = exp['query_type']
        os.environ['NUM_QUERIES'] = str(exp.get('num_queries', 100))
        os.environ['RDF_TYPE_URI'] = exp['rdf_type_uri']
        os.environ['GCARE_HOME'] = str(gcare_dir.absolute())
        os.environ['DATA_ROOT'] = str((gcare_dir / 'data').absolute())
        os.environ['RESULTS_ROOT'] = str(exp_results_dir.absolute())
        
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to gcare directory
            os.chdir(gcare_dir)
            
            # Import and run build_statistics
            print(f"[{name}] Running build_statistics...")
            import build_statistics
            build_statistics.main()
            
            # Import and run run_estimation
            print(f"[{name}] Running estimation...")
            import run_estimation
            run_estimation.main()
            
            print(f"[{name}] Completed successfully")
            completed += 1
            
        except Exception as e:
            print(f"ERROR: [{name}] Failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed: {completed}")
    print(f"Runtime: {total_time / 3600:.2f} hours")
    print(f"Results: {results_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run G-Care benchmark experiments (no Docker)")
    parser.add_argument("--config", type=Path, default=Path("experiments_config.yaml"),
                       help="Experiment configuration YAML file")
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
    
    # Clear directories and run
    # clear_gcare_directories(workspace_root)
    # preprocess_graphs(experiments, workspace_root)
    run_experiments(experiments, results_dir, workspace_root)
    
    print("\n✓ All experiments completed successfully!")


if __name__ == "__main__":
    main()
