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
import time
import traceback
from datetime import datetime
from pathlib import Path

# Helper lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helper import (
    DATA_DIR,
    PROJECT_ROOT,
    LSS_DIR,
    SHARED_MEGA_NAME,
    build_shared_mega_graph,
    build_experiment_manifests,
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
                        help="Run only specific experiment IDs (e.g. loco_swdf loco_lubm)")
    cli = parser.parse_args()

    config = load_config()

    all_dataset_names = [ds["name"] for ds in config["datasets"]]
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
        bundles_dir = per_ds_dir / name / "bundles"
        if stats_path.exists() and bundles_dir.exists() and any(bundles_dir.iterdir()):
            print(f"  {name}: already converted")
            with open(stats_path) as f:
                all_stats[name] = json.load(f)
            continue

        if stats_path.exists():
            print(f"  {name}: stats exist but bundles missing — re-converting")

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

    # ── STEP 3: Build shared mega-graph (once for all experiments) ───────────
    print("\n" + "=" * 60)
    print("STEP 3: Build shared mega-graph")
    print("=" * 60)

    shared_dir = DATA_DIR / "shared_mega"
    mega_graph_path = shared_dir / "dataset" / SHARED_MEGA_NAME / f"{SHARED_MEGA_NAME}.txt"
    done_sentinel = shared_dir / "dataset" / ".done"

    if done_sentinel.exists():
        print("  Shared mega-graph already built, skipping")
    else:
        print("  Building shared mega-graph (all datasets)...")
        build_shared_mega_graph(all_dataset_names, global_map, per_ds_dir, shared_dir)
        done_sentinel.touch()

    # ── STEP 4: Generate ProNE embeddings (once for all experiments) ──────────
    print("\n" + "=" * 60)
    print("STEP 4: Generate shared ProNE embeddings")
    print("=" * 60)

    emb_path = shared_dir / "prone" / f"{SHARED_MEGA_NAME}.emb.npy"
    prone_time_s = 0.0
    if emb_path.exists():
        print("  Shared ProNE embeddings already exist, skipping generation")
    else:
        print("  Generating ProNE embeddings on shared mega-graph...")
        _t0 = time.perf_counter()
        generate_prone_embeddings(
            mega_graph_path,
            global_map["num_global_types"],
            emb_path,
        )
        prone_time_s = time.perf_counter() - _t0
        print(f"  ProNE done in {prone_time_s:.2f}s")

    # ── STEP 5: Run LOCO experiments ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Run LOCO experiments")
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

        train_ds, _, eval_ds = resolve_experiment_datasets(
            exp_cfg, all_dataset_names
        )
        print(f"  Train datasets : {train_ds}")
        print(f"  Eval datasets  : {eval_ds}")

        # Build per-experiment query manifests (cheap JSON files)
        train_manifest = shared_dir / "queryset_homo" / f"train_{exp_id}.bundles.json"
        eval_manifest  = shared_dir / "queryset_homo" / f"eval_{exp_id}.bundles.json"
        if train_manifest.exists() and eval_manifest.exists():
            print("  Query manifests already exist, skipping")
        else:
            print("  Building query manifests...")
            build_experiment_manifests(
                exp_id, train_ds, eval_ds,
                shared_dir, per_ds_dir, global_map, all_stats,
            )

        # Run fold (training + evaluation share the same mega-graph)
        try:
            run_loco_fold(
                exp_id, train_ds, eval_ds,
                shared_dir, all_stats, global_map,
                results_dir / exp_id,
                epochs_override=cli.epochs,
                prone_time_s=prone_time_s,
            )
        except Exception as e:
            errors.append(f"{exp_id}: {e}")
            traceback.print_exc()
            print(f"  ERROR in {exp_id}: {e}")

    # ── STEP 6: Summary ───────────────────────────────────────────────────────
    print_summary(errors)


if __name__ == "__main__":
    main()
