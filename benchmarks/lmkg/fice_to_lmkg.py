#!/usr/bin/env python3
"""
Convert FICE RDF data to LMKG-S format.

LMKG-S needs:
  - final_datasets/generated_{dataset}/star/star_{N}.txt   (training queries)
  - final_datasets/generated_{dataset}/star/eval_600_star_{N}.txt
  - final_datasets/generated_{dataset}/path/chain_{N}.txt
  - final_datasets/generated_{dataset}/path/eval_600_chain_{N}.txt
  - statistics/{dataset}_stats.json  (d, b, n, e values for lmkgs.py)

Query formats:
  Star:  pred_id-obj_id,pred_id-obj_id,...:cardinality  (* = variable)
  Chain: node-pred-node-pred-node,...,cardinality        (* = variable)

IDs are 1-based (FICE id + 1) because LMKG's query_graph.py uses 0 to
represent both unbound variables and absent terms, so FICE's 0-based IDs
must be shifted to avoid that collision.

Usage:
  python benchmarks/lmkg/fice-to-lmkg.py [--config benchmarks/lmkg/experiments_config.yaml]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LMKG_S_DIR = PROJECT_ROOT / "benchmarks/lmkg/LMKG-Code/lmkg-s"

EVAL_NUM = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mappings(dataset_name):
    """Build (relation_to_id, entity_to_id) by parsing the raw .nt file.

    IDs are assigned in order of first appearance, matching FICE's own
    processing logic in datasets.py.
    """
    raw_dir = DATA_DIR / dataset_name / "raw"
    raw_files = [p for p in raw_dir.iterdir() if p.is_file()]
    if not raw_files:
        raise FileNotFoundError(f"No raw file found in {raw_dir}")
    raw_path = raw_files[0]

    ent_to_id: dict = {}
    rel_to_id: dict = {}

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")[:3]
            if len(parts) != 3:
                continue
            s, p, o = parts
            if s not in ent_to_id:
                ent_to_id[s] = len(ent_to_id)
            if o not in ent_to_id:
                ent_to_id[o] = len(ent_to_id)
            if p not in rel_to_id:
                rel_to_id[p] = len(rel_to_id)

    return rel_to_id, ent_to_id


def load_fice_queries(query_path):
    """Load FICE query JSON file, stripping any URL parameters."""
    path = query_path.split("?")[0]
    with open(path) as f:
        return json.load(f)


def parse_uri(token):
    """Strip angle brackets from a URI token."""
    return token.strip("<>")


def triple_elements(triple):
    """Return (s, p, o) from a FICE triple (which may have 4 elements including '.')."""
    return triple[0], triple[1], triple[2]


# ---------------------------------------------------------------------------
# Query conversion
# ---------------------------------------------------------------------------

def node_to_str(token, ent_to_id):
    """Convert a SPARQL node token to an LMKG node string.

    Variables (?...) → '*'
    Constant URIs    → 1-based integer string (FICE id + 1)

    IDs are offset by +1 because LMKG's query_graph.py maps both unbound
    variables and constant-id-0 to the integer 0 in the binary encoding,
    making them indistinguishable. Shifting to 1-based avoids that collision.

    Mapping keys may include angle brackets (e.g. '<http://...>'), so we try
    the raw token first, then the stripped version.
    """
    if token.startswith("?"):
        return "*"
    eid = ent_to_id.get(token) if token in ent_to_id else ent_to_id.get(parse_uri(token))
    return None if eid is None else str(eid + 1)


def pred_to_str(token, rel_to_id):
    """Convert a SPARQL predicate token to an LMKG predicate string.

    Variables (?...) → '*'
    Constant URIs    → 1-based integer string (FICE id + 1)

    See node_to_str for why IDs are offset by +1.
    """
    if token.startswith("?"):
        return "*"
    rid = rel_to_id.get(token) if token in rel_to_id else rel_to_id.get(parse_uri(token))
    return None if rid is None else str(rid + 1)


def convert_star_query(query_data, rel_to_id, ent_to_id):
    """Convert a FICE star query to LMKG-S format.

    Returns (lmkg_line, query_size) or None on failure.
    Format: pred_id-obj_id,pred_id-obj_id,...:cardinality
    """
    triples = query_data["triples"]
    cardinality = query_data["y"]
    parts = []
    for triple in triples:
        s, p, o = triple_elements(triple)
        p_str = pred_to_str(p, rel_to_id)
        if p_str is None:
            return None
        o_str = node_to_str(o, ent_to_id)
        if o_str is None:
            return None
        parts.append(f"{p_str}-{o_str}")
    return f"{','.join(parts)}:{cardinality}", len(triples)


def convert_path_query(query_data, rel_to_id, ent_to_id):
    """Convert a FICE path/chain query to LMKG-S format.

    Returns (lmkg_line, query_size) or None on failure.
    Format: node-pred-node-pred-node,...,cardinality
    """
    triples = query_data["triples"]
    cardinality = query_data["y"]
    chain_parts = []
    for i, triple in enumerate(triples):
        s, p, o = triple_elements(triple)
        p_str = pred_to_str(p, rel_to_id)
        if p_str is None:
            return None
        if i == 0:
            s_str = node_to_str(s, ent_to_id)
            if s_str is None:
                return None
            chain_parts.append(s_str)
        o_str = node_to_str(o, ent_to_id)
        if o_str is None:
            return None
        chain_parts.extend([p_str, o_str])
    return f"{'-'.join(chain_parts)},{cardinality}", len(triples)


# ---------------------------------------------------------------------------
# Group queries by size
# ---------------------------------------------------------------------------

def group_by_size(lmkg_lines):
    """Group converted LMKG query strings by query size (number of triples)."""
    by_size = defaultdict(list)
    for line, size in lmkg_lines:
        by_size[size].append(line)
    return by_size


# ---------------------------------------------------------------------------
# LMKG-S output
# ---------------------------------------------------------------------------

def write_lmkgs_files(by_size, dataset_name, query_type):
    """Write LMKG-S training and eval query files grouped by query size."""
    lmkgs_type = "star" if query_type == "star" else "chain"
    subdir = "star" if query_type == "star" else "path"

    base_dir = LMKG_S_DIR / "final_datasets" / f"generated_{dataset_name}" / subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for size, lines in sorted(by_size.items()):
        random.shuffle(lines)
        n_eval = min(EVAL_NUM, len(lines))
        eval_lines = lines[:n_eval]
        train_lines = lines[n_eval:]

        eval_path = base_dir / f"eval_{n_eval}_{lmkgs_type}_{size}.txt"

        with open(eval_path, "w") as f:
            f.write("\n".join(eval_lines) + "\n")

        if train_lines:
            train_path = base_dir / f"{lmkgs_type}_{size}.txt"
            with open(train_path, "w") as f:
                f.write("\n".join(train_lines) + "\n")

        stats[size] = {"train": len(train_lines), "eval": len(eval_lines)}
        print(f"  {lmkgs_type}_{size}: {len(train_lines)} train, {len(eval_lines)} eval")

    return stats


# ---------------------------------------------------------------------------
# Statistics file for lmkgs.py
# ---------------------------------------------------------------------------

def write_stats_file(dataset_name, num_entities, num_relations, star_sizes, path_sizes):
    """Write a stats JSON with d, b, n, e values needed by lmkgs.py.

    For a query of size N (both star and chain): n = N+1, e = N.
    """
    stats_dir = LMKG_S_DIR / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = stats_dir / f"{dataset_name}_stats.json"

    data = {
        "dataset": dataset_name,
        "d": num_entities + 1,  # +1 because entity IDs are shifted to 1-based
        "b": num_relations + 1,  # +1 because relation IDs are shifted to 1-based
        "star_sizes": sorted(star_sizes),
        "path_sizes": sorted(path_sizes),
        "note": "For a query of size N: n = N+1, e = N (both star and chain)",
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  stats: {out_path.name}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_experiment(exp: dict):
    name = exp["id"]
    dataset_name = exp["dataset_name"]
    query_type = exp["query_type"]  # "star" or "path"
    query_path = str(PROJECT_ROOT / exp["path"])

    print(f"\n=== {name} (dataset={dataset_name}, type={query_type}) ===")

    rel_to_id, ent_to_id = compute_mappings(dataset_name)
    num_entities = len(ent_to_id)
    num_relations = len(rel_to_id)
    print(f"  Entities: {num_entities}, Relations: {num_relations}")

    queries = load_fice_queries(query_path)
    print(f"  Loaded {len(queries)} queries")

    converter = convert_star_query if query_type == "star" else convert_path_query
    converted = []
    skipped = 0
    for q in queries:
        result = converter(q, rel_to_id, ent_to_id)
        if result is None:
            skipped += 1
            continue
        converted.append(result)

    print(f"  Converted {len(converted)} queries ({skipped} skipped)")

    by_size = group_by_size(converted)
    print(f"  Sizes: { {k: len(v) for k, v in sorted(by_size.items())} }")

    write_lmkgs_files(by_size, dataset_name, query_type)

    return by_size, num_entities, num_relations


def main():
    parser = argparse.ArgumentParser(description="Convert FICE data to LMKG-S format")
    parser.add_argument(
        "--config",
        default="benchmarks/experiments_config.yaml",
        help="Path to experiments_config.yaml",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Only run specific experiment names (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiments = config["queries"]
    if args.experiments:
        experiments = [e for e in experiments if e["id"] in args.experiments]
        print(f"Running {len(experiments)} experiment(s): {[e['id'] for e in experiments]}")

    # Accumulate per-dataset stats across star + path experiments
    dataset_stats = {}

    for exp in experiments:
        by_size, num_entities, num_relations = process_experiment(exp)
        dname = exp["dataset_name"]
        qtype = exp["query_type"]

        if dname not in dataset_stats:
            dataset_stats[dname] = {
                "num_entities": num_entities,
                "num_relations": num_relations,
                "star_sizes": [],
                "path_sizes": [],
            }
        key = "star_sizes" if qtype == "star" else "path_sizes"
        dataset_stats[dname][key] = sorted(by_size.keys())

    for dname, stats in dataset_stats.items():
        write_stats_file(
            dname,
            stats["num_entities"],
            stats["num_relations"],
            stats["star_sizes"],
            stats["path_sizes"],
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
