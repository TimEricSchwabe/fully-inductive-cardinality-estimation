#!/usr/bin/env python3
"""
LSS LOCO benchmark — run all experiments.

Usage (from project root):
    python benchmarks/lss/run_all_experiments.py

Reads:  benchmarks/experiments_config.yaml
Writes: benchmarks/lss/results/run_{timestamp}/{exp_id}/predictions.csv
"""
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

# Helper lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helper import (
    DATA_DIR,
    PROJECT_ROOT,
    LSS_DIR,
    build_mega_graph_files,
    compute_global_mappings,
    convert_dataset_to_lss,
    generate_prone_embeddings,
    load_config,
    print_summary,
    resolve_experiment_datasets,
    run_loco_fold,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: 80)")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Run only specific experiment IDs")
    cli = parser.parse_args()

    config = load_config()

    all_dataset_names = [ds["name"] for ds in config["datasets"]]
    dataset_by_name = {ds["name"]: ds for ds in config["datasets"]}
    queries_by_dataset = {}
    for q in config["queries"]:
        queries_by_dataset.setdefault(q["dataset_name"], []).append(q)

    # ── STEP 1: Convert all datasets to LSS format ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Convert FICE data to LSS format")
    print("=" * 60)

    per_ds_dir = DATA_DIR / "per_dataset"
    all_stats = {}

    for ds_cfg in config["datasets"]:
        name = ds_cfg["name"]
        stats_path = per_ds_dir / name / "stats.json"
        if stats_path.exists():
            print(f"  {name}: already converted")
            with open(stats_path) as f:
                all_stats[name] = json.load(f)
            continue

        print(f"\n  Converting {name}...")
        qcfgs = queries_by_dataset.get(name, [])
        stats = convert_dataset_to_lss(ds_cfg, qcfgs, PROJECT_ROOT, per_ds_dir)
        all_stats[name] = stats
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    # ── STEP 2: Compute global label mappings ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Compute global label mappings")
    print("=" * 60)

    global_map_path = DATA_DIR / "global_label_map.json"
    if global_map_path.exists():
        print("  Loading existing global label map")
        with open(global_map_path) as f:
            global_map = json.load(f)
    else:
        global_map = compute_global_mappings(all_stats, all_dataset_names)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(global_map_path, "w") as f:
            json.dump(global_map, f, indent=2)
        print(f"  Total entity types : {global_map['num_global_types']}")
        print(f"  Total relations    : {global_map['num_global_rels']}")
        print(f"  Total entities     : {global_map['num_global_entities']}")

    # ── STEP 3: Run LOCO experiments ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Run LOCO experiments")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = LSS_DIR / "results" / f"run_{timestamp}"

    errors = []

    if cli.experiments:
        config["experiments"] = [e for e in config["experiments"]
                                  if e["id"] in cli.experiments]

    for exp_cfg in config["experiments"]:
        exp_id = exp_cfg["id"]
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_id}")
        print("=" * 60)

        train_ds, left_out, eval_ds = resolve_experiment_datasets(
            exp_cfg, all_dataset_names
        )
        print(f"  Train datasets : {train_ds}")
        print(f"  Eval datasets  : {eval_ds}")

        mega_dir = DATA_DIR / f"mega_{exp_id}"

        # Build mega-graph files
        mega_graph_path = (
            mega_dir / "dataset" / f"train_{exp_id}" / f"train_{exp_id}.txt"
        )
        if mega_graph_path.exists():
            print("  Mega-graph files already exist, skipping build")
        else:
            print("  Building mega-graph files...")
            build_mega_graph_files(
                exp_id, all_dataset_names, train_ds, eval_ds,
                per_ds_dir, mega_dir, global_map, all_stats,
            )

        # Generate ProNE embeddings
        emb_path = mega_dir / "prone" / f"train_{exp_id}.emb.npy"
        if emb_path.exists():
            print("  ProNE embeddings already exist, skipping generation")
        else:
            print("  Generating ProNE embeddings...")
            generate_prone_embeddings(
                mega_graph_path,
                global_map["num_global_types"],
                emb_path,
            )

        # Run fold
        try:
            run_loco_fold(
                exp_id, train_ds, eval_ds,
                mega_dir, all_stats, global_map,
                results_dir / exp_id,
                epochs_override=cli.epochs,
            )
        except Exception as e:
            errors.append(f"{exp_id}: {e}")
            traceback.print_exc()
            print(f"  ERROR in {exp_id}: {e}")

    # ── STEP 4: Summary ───────────────────────────────────────────────────────
    print_summary(errors)


if __name__ == "__main__":
    main()
