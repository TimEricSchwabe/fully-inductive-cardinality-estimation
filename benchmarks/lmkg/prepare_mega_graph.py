#!/usr/bin/env python3
"""
Prepare LOCO (Leave-One-Graph-Out) mega-graph datasets for LMKG-S multi-graph training.

For each held-out dataset, creates:
  - Training files: remapped queries from all other datasets (global IDs)
  - Eval files:     remapped queries from the held-out dataset (global IDs)
  - Stats file:     mega_no_{eval_ds}_stats.json with mega_d and mega_b

Global ID assignment:
  - entity_offset[i]   = sum of num_entities for datasets 0..i-1
  - relation_offset[i] = sum of num_relations for datasets 0..i-1
  - All entity/relation IDs in query files are 1-based (FICE id + 1), so
    the remapped global ID = local_id + offset (offset 0 for first dataset).

Usage:
  python benchmarks/lmkg/prepare-mega-graph.py [--config benchmarks/lmkg/experiments_config.yaml]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LMKG_S_DIR = PROJECT_ROOT / "benchmarks/lmkg/LMKG-Code/lmkg-s"
_CONFIG_PATH = PROJECT_ROOT / "benchmarks/experiments_config.yaml"

# Datasets in a stable order for consistent global ID assignment — loaded from
# shared config so the list stays in sync with experiments_config.yaml.
with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)
DATASETS = [d["name"] for d in _cfg["datasets"]]


def load_stats(dataset_name):
    path = LMKG_S_DIR / "statistics" / f"{dataset_name}_stats.json"
    with open(path) as f:
        return json.load(f)


def compute_offsets(dataset_list):
    """Compute entity/relation offsets and mega d/b values.

    Returns (entity_offsets, relation_offsets, mega_d, mega_b).
    entity_offsets[ds] = number of entities in all preceding datasets.
    """
    stats = {ds: load_stats(ds) for ds in dataset_list}

    entity_offsets = {}
    relation_offsets = {}
    e_acc = 0
    r_acc = 0
    for ds in dataset_list:
        entity_offsets[ds] = e_acc
        relation_offsets[ds] = r_acc
        # stats["d"] = num_entities + 1 (1-based shift), so num_entities = d - 1
        e_acc += stats[ds]["d"] - 1
        r_acc += stats[ds]["b"] - 1

    # mega_d and mega_b follow the same convention: total_count + 1
    mega_d = e_acc + 1
    mega_b = r_acc + 1

    return entity_offsets, relation_offsets, mega_d, mega_b


def remap_star_line(line, entity_offset, relation_offset):
    """Remap a star query line by adding entity/relation offsets.

    Format: pred_id-obj_id,pred_id-obj_id,...:cardinality
    pred_id and obj_id may be integers (1-based) or '*' (variable).
    """
    line = line.strip()
    if not line:
        return line
    query_part, cardinality = line.rsplit(":", 1)
    triples = query_part.split(",")
    remapped = []
    for triple in triples:
        pred_str, obj_str = triple.split("-", 1)
        if pred_str != "*":
            pred_str = str(int(pred_str) + relation_offset)
        if obj_str != "*":
            obj_str = str(int(obj_str) + entity_offset)
        remapped.append(f"{pred_str}-{obj_str}")
    return f"{','.join(remapped)}:{cardinality}"


def remap_chain_line(line, entity_offset, relation_offset):
    """Remap a chain query line by adding entity/relation offsets.

    Format: node-pred-node-pred-...,cardinality
    Alternating node/pred tokens (starting with node); last token is cardinality.
    """
    line = line.strip()
    if not line:
        return line
    *chain_parts_and_comma, cardinality = line.rsplit(",", 1)
    chain_str = chain_parts_and_comma[0]
    tokens = chain_str.split("-")
    remapped = []
    for i, tok in enumerate(tokens):
        if tok == "*":
            remapped.append(tok)
        elif i % 2 == 0:
            # node position (even index)
            remapped.append(str(int(tok) + entity_offset))
        else:
            # predicate position (odd index)
            remapped.append(str(int(tok) + relation_offset))
    return f"{'-'.join(remapped)},{cardinality}"


def read_query_file(path: Path):
    """Read non-empty lines from a query file."""
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def remap_dataset_queries(dataset_name, entity_offset, relation_offset):
    """Load and remap all query files for one dataset.

    Returns {query_type: {size: [remapped_lines]}}.
    query_type is 'star' or 'path'.
    """
    result = {"star": defaultdict(list), "path": defaultdict(list)}

    gen_dir = LMKG_S_DIR / "final_datasets" / f"generated_{dataset_name}"

    for qtype, subdir, prefix in [
        ("star", "star", "star"),
        ("path", "path", "chain"),
    ]:
        type_dir = gen_dir / subdir
        if not type_dir.exists():
            continue

        # Find training files: star_{N}.txt or chain_{N}.txt
        for train_file in sorted(type_dir.glob(f"{prefix}_*.txt")):
            # Skip eval files
            if train_file.name.startswith("eval_"):
                continue
            # Extract size from filename (e.g. star_2.txt -> 2)
            try:
                size = int(train_file.stem.split("_")[-1])
            except ValueError:
                continue

            remap_fn = remap_star_line if qtype == "star" else remap_chain_line
            lines = read_query_file(train_file)
            remapped = [remap_fn(l, entity_offset, relation_offset) for l in lines if l]
            result[qtype][size].extend(remapped)

    return result


def read_eval_queries(dataset_name, entity_offset, relation_offset):
    """Load and remap eval query files for one dataset.

    Returns {query_type: {size: [remapped_lines]}}.
    """
    result = {"star": defaultdict(list), "path": defaultdict(list)}

    gen_dir = LMKG_S_DIR / "final_datasets" / f"generated_{dataset_name}"

    for qtype, subdir, prefix in [
        ("star", "star", "star"),
        ("path", "path", "chain"),
    ]:
        type_dir = gen_dir / subdir
        if not type_dir.exists():
            continue

        for eval_file in sorted(type_dir.glob(f"eval_600_{prefix}_*.txt")):
            try:
                size = int(eval_file.stem.split("_")[-1])
            except ValueError:
                continue

            remap_fn = remap_star_line if qtype == "star" else remap_chain_line
            lines = read_query_file(eval_file)
            remapped = [remap_fn(l, entity_offset, relation_offset) for l in lines if l]
            result[qtype][size].extend(remapped)

    return result


def write_fold(eval_ds, train_queries, eval_queries, mega_d, mega_b):
    """Write all files for one LOCO fold (mega_no_{eval_ds})."""
    fold_name = f"mega_no_{eval_ds}"
    fold_dir = LMKG_S_DIR / "final_datasets" / f"generated_{fold_name}"

    for qtype, subdir, prefix in [
        ("star", "star", "star"),
        ("path", "path", "chain"),
    ]:
        type_dir = fold_dir / subdir
        type_dir.mkdir(parents=True, exist_ok=True)

        # Write training files
        for size, lines in sorted(train_queries[qtype].items()):
            train_path = type_dir / f"{prefix}_{size}.txt"
            with open(train_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"  [train] {train_path.relative_to(LMKG_S_DIR)}: {len(lines)} queries")

        # Write eval files
        for size, lines in sorted(eval_queries[qtype].items()):
            n_eval = min(600, len(lines))
            eval_path = type_dir / f"eval_600_{prefix}_{size}.txt"
            with open(eval_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"  [eval]  {eval_path.relative_to(LMKG_S_DIR)}: {n_eval} queries")

    # Write stats
    stats_dir = LMKG_S_DIR / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{fold_name}_stats.json"

    star_sizes = sorted(train_queries["star"].keys())
    path_sizes = sorted(train_queries["path"].keys())
    eval_star_sizes = sorted(eval_queries["star"].keys())
    eval_path_sizes = sorted(eval_queries["path"].keys())
    stats_data = {
        "dataset": fold_name,
        "d": mega_d,
        "b": mega_b,
        "star_sizes": star_sizes,
        "path_sizes": path_sizes,
        "eval_star_sizes": eval_star_sizes,
        "eval_path_sizes": eval_path_sizes,
        "held_out": eval_ds,
        "note": "For a query of size N: n = N+1, e = N (both star and chain)",
    }
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"  stats: {stats_path.name}")


def write_star_path_folds(eval_ds, train_queries, eval_queries, mega_d, mega_b):
    """Write two LOCO folds for separate star/path evaluation.

    Creates mega_no_{eval_ds}_star and mega_no_{eval_ds}_path:
    - Both have identical training files (all star + path from N-1 graphs).
    - _star: eval files = only star eval files from held-out; eval_path_sizes=[].
    - _path: eval files = only path eval files from held-out; eval_star_sizes=[].
    """
    # _star variant: training = both types, eval = star only
    write_fold(
        eval_ds=eval_ds + "_star",
        train_queries=train_queries,
        eval_queries={"star": eval_queries["star"], "path": defaultdict(list)},
        mega_d=mega_d,
        mega_b=mega_b,
    )
    # _path variant: training = both types, eval = path only
    write_fold(
        eval_ds=eval_ds + "_path",
        train_queries=train_queries,
        eval_queries={"star": defaultdict(list), "path": eval_queries["path"]},
        mega_d=mega_d,
        mega_b=mega_b,
    )


def main():

    dataset_list = DATASETS
    print(f"Datasets ({len(dataset_list)}): {dataset_list}")

    # Compute stable global ID offsets
    entity_offsets, relation_offsets, mega_d, mega_b = compute_offsets(dataset_list)
    print(f"mega_d={mega_d}, mega_b={mega_b}")
    for ds in dataset_list:
        print(f"  {ds}: entity_offset={entity_offsets[ds]}, relation_offset={relation_offsets[ds]}")

    # Remap all datasets once
    print("\nRemapping query files...")
    all_train = {}
    all_eval = {}
    for ds in dataset_list:
        print(f"  {ds}...")
        all_train[ds] = remap_dataset_queries(ds, entity_offsets[ds], relation_offsets[ds])
        all_eval[ds] = read_eval_queries(ds, entity_offsets[ds], relation_offsets[ds])

    # Write one fold per held-out dataset
    for eval_ds in dataset_list:
        print(f"\n=== Fold: mega_no_{eval_ds} ===")

        # Aggregate training queries from all datasets except eval_ds
        merged_train = {
            "star": defaultdict(list),
            "path": defaultdict(list),
        }
        for ds in dataset_list:
            if ds == eval_ds:
                continue
            for qtype in ("star", "path"):
                for size, lines in all_train[ds][qtype].items():
                    merged_train[qtype][size].extend(lines)

        write_fold(
            eval_ds=eval_ds,
            train_queries=merged_train,
            eval_queries=all_eval[eval_ds],
            mega_d=mega_d,
            mega_b=mega_b,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
