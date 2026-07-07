#!/usr/bin/env python3
"""G-Care Benchmark Experiment Runner

Runs G-Care cardinality estimation experiments using the locally compiled
G-Care binary (no Docker required).

Pipeline
--------
1. Preprocessing  – transform RDF graphs to G-Care text format and cache them.
2. Build stats     – run the G-Care build step (wj/cset summaries) per dataset.
3. Estimation      – run estimation in parallel for every experiment.

Query split
-----------
Each experiment loads at most `max_train_queries` queries from the FICE JSON
file, then uses the **last 20 %** as the evaluation set (consistent with the
80/20 split used by FICE and other baselines). G-Care has no training step so
the training portion is discarded.

Usage
-----
    python run_all_experiments.py [--max-parallel 5] [--gcare-home gcare/]
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Project root: three levels up from this file (benchmarks/gcare/run_all_experiments.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GCARE_BENCH = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> List[Dict]:
    """Load experiments from the shared experiments_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    datasets_by_name = {d["name"]: d for d in cfg["datasets"]}
    experiments = []
    for q in cfg["queries"]:
        ds = datasets_by_name[q["dataset_name"]]
        experiments.append({
            "name": q["id"],
            "dataset_name": q["dataset_name"],
            "query_type": q["query_type"],
            "query_path": PROJECT_ROOT / q["path"],
            "graph_path": PROJECT_ROOT / ds["graph_path"],
            "rdf_type_uri": ds["rdf_type_uri"],
            "num_queries": int(q.get("max_train_queries", 500)),
        })
    return experiments


def filter_experiments(
    experiments: List[Dict],
    only_datasets: Optional[List[str]],
    only_query_types: Optional[List[str]],
) -> List[Dict]:
    if not only_datasets and not only_query_types:
        return experiments
    ds_set = set(only_datasets or [])
    qt_set = set(only_query_types or [])
    return [
        e for e in experiments
        if (not ds_set or e["dataset_name"] in ds_set)
        and (not qt_set or e["query_type"] in qt_set)
    ]


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Phase 1: Graph preprocessing (transform to G-Care text format)
# ---------------------------------------------------------------------------

def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _cache_valid(cache_dir: Path, graph_path: Path, rdf_type_uri: str) -> bool:
    meta = cache_dir / "cache_meta.json"
    if not meta.exists():
        return False
    try:
        with open(meta) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return d.get("graph_path") == str(graph_path) and d.get("rdf_type_uri") == rdf_type_uri


def _write_cache_meta(cache_dir: Path, graph_path: Path, rdf_type_uri: str) -> None:
    with open(cache_dir / "cache_meta.json", "w") as f:
        json.dump({"graph_path": str(graph_path), "rdf_type_uri": rdf_type_uri}, f, indent=2)


def _ensure_graph_cache(
    dataset_name: str, graph_path: Path, rdf_type_uri: str, cache_root: Path
) -> Path:
    cache_dir = cache_root / dataset_name
    txt = cache_dir / f"{dataset_name}.txt"
    e_map = cache_dir / "id_to_id_mapping.json"
    p_map = cache_dir / "id_to_id_mapping_predicate.json"

    if _cache_valid(cache_dir, graph_path, rdf_type_uri) and txt.exists() and e_map.exists() and p_map.exists():
        print(f"[{dataset_name}] Using cached transformed graph")
        return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name}] Building transformed graph cache …")
    sys.path.insert(0, str(GCARE_BENCH))
    from transform_graph import transform_ttl_to_gcare  # noqa: PLC0415
    transform_ttl_to_gcare(
        graph_path=graph_path,
        rdf_type_uri=rdf_type_uri,
        dataset_name=dataset_name,
        out_path=cache_dir,
    )
    _write_cache_meta(cache_dir, graph_path, rdf_type_uri)
    return cache_dir


def _materialize_graph_inputs(
    exp_name: str, dataset_name: str, dataset_dir: Path, cache_root: Path
) -> None:
    cache_dir = cache_root / dataset_name
    exp_dir = dataset_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(cache_dir / f"{dataset_name}.txt", exp_dir / f"{exp_name}.txt")
    _link_or_copy(cache_dir / "id_to_id_mapping.json", exp_dir / "id_to_id_mapping.json")
    _link_or_copy(cache_dir / "id_to_id_mapping_predicate.json", exp_dir / "id_to_id_mapping_predicate.json")


def preprocess_graphs(experiments: List[Dict], gcare_home: Path) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing Graphs and Queries")
    print("=" * 60)

    dataset_dir = gcare_home / "data" / "dataset"
    cache_root = gcare_home / "data" / "cache" / "dataset_graphs"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Build (or reuse) one cached transformed graph per dataset.
    seen_datasets: Dict[str, Dict] = {}
    for exp in experiments:
        seen_datasets.setdefault(
            exp["dataset_name"],
            {"graph_path": exp["graph_path"], "rdf_type_uri": exp["rdf_type_uri"]},
        )

    for ds_name, ds_cfg in seen_datasets.items():
        gp = Path(ds_cfg["graph_path"])
        if not gp.exists():
            print(f"ERROR: Graph file not found: {gp}")
            sys.exit(1)
        try:
            _ensure_graph_cache(ds_name, gp, ds_cfg["rdf_type_uri"], cache_root)
        except Exception as exc:
            print(f"ERROR: Graph cache build failed for {ds_name}: {exc}")
            sys.exit(1)

    # Materialize per-experiment graph inputs (txt + ID maps).
    for exp in experiments:
        try:
            _materialize_graph_inputs(exp["name"], exp["dataset_name"], dataset_dir, cache_root)
            print(f"[{exp['name']}] Graph inputs ready")
        except Exception as exc:
            print(f"ERROR: {exp['name']}: {exc}")
            sys.exit(1)

    # Copy query JSON into the G-Care queryset tree.
    for exp in experiments:
        qdir = gcare_home / "data" / "queryset" / exp["name"]
        qdir.mkdir(parents=True, exist_ok=True)
        qp = Path(exp["query_path"])
        if not qp.exists():
            print(f"ERROR: Query file not found: {qp}")
            sys.exit(1)
        dest = qdir / f"{exp['name']}.json"
        if not dest.exists():
            shutil.copy(qp, dest)
            print(f"[{exp['name']}] Query file copied")
        else:
            print(f"[{exp['name']}] Query file already present")

    print("Preprocessing complete.\n")


# ---------------------------------------------------------------------------
# Phase 2: Build statistics (one per dataset)
# ---------------------------------------------------------------------------

def build_all_statistics(
    experiments: List[Dict], gcare_home: Path, methods: Optional[List[str]] = None
) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Building G-Care Statistics")
    print("=" * 60)

    sys.path.insert(0, str(GCARE_BENCH))
    from build_statistics import build_statistics  # noqa: PLC0415

    seen: set = set()
    for exp in experiments:
        exp_name = exp["name"]
        # Statistics are keyed on exp_name because the dataset directory is
        # named after the experiment (e.g. lubm_star) not the base dataset.
        if exp_name in seen:
            continue
        seen.add(exp_name)
        try:
            build_statistics(exp_name, gcare_home, methods)
        except Exception as exc:
            print(f"ERROR: build_statistics failed for {exp_name}: {exc}")
            sys.exit(1)

    print("All statistics built.\n")


# ---------------------------------------------------------------------------
# Phase 3: Run estimation (parallel)
# ---------------------------------------------------------------------------

def _run_one_experiment(
    exp: Dict,
    results_dir: Path,
    gcare_home: Path,
    methods: Optional[List[str]] = None,
) -> str:
    """Worker function executed in a subprocess pool."""
    # Import here so the worker process has the bench directory on its path.
    sys.path.insert(0, str(GCARE_BENCH))
    from run_estimation import run_estimation  # noqa: PLC0415

    exp_results = results_dir / exp["name"]
    exp_results.mkdir(parents=True, exist_ok=True)

    run_estimation(
        exp_name=exp["name"],
        dataset_name=exp["dataset_name"],
        query_type=exp["query_type"],
        query_path=Path(exp["query_path"]),
        rdf_type_uri=exp["rdf_type_uri"],
        num_queries=exp["num_queries"],
        results_dir=exp_results,
        gcare_home=gcare_home,
        methods=methods,
    )
    return exp["name"]


def run_experiments(
    experiments: List[Dict],
    results_dir: Path,
    gcare_home: Path,
    max_parallel: int = 5,
    methods: Optional[List[str]] = None,
) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Running Estimation")
    print("=" * 60)

    errors: List[str] = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(_run_one_experiment, exp, results_dir, gcare_home, methods): exp["name"]
            for exp in experiments
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"[{name}] Completed successfully")
            except Exception as exc:
                print(f"ERROR: [{name}] failed: {exc}")
                errors.append(name)

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed / 3600:.2f} h  |  Results: {results_dir}")
    return errors  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(errors: List[str]) -> None:
    print("\n" + "=" * 60)
    if errors:
        print(f"Done with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run G-Care benchmark experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "experiments_config.yaml",
        help="Shared experiments_config.yaml",
    )
    parser.add_argument(
        "--gcare-home",
        type=Path,
        default=GCARE_BENCH / "gcare",
        help="Path to the compiled G-Care directory (default: benchmarks/gcare/gcare/)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Max parallel estimation workers (default: 5)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip graph transformation and query copy steps",
    )
    parser.add_argument(
        "--skip-build-stats",
        action="store_true",
        help="Skip building G-Care statistics (use existing .graph files)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Comma-separated dataset names to run, e.g. yago,hetionet",
    )
    parser.add_argument(
        "--query-types",
        type=str,
        default=None,
        help="Comma-separated query types to run, e.g. path,star,cycle",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated estimators to run, e.g. sumrdf,wj,cset (default: all)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    gcare_home: Path = args.gcare_home.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = GCARE_BENCH / "results" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}\n")

    # Load and filter
    experiments = load_config(args.config)
    print(f"Loaded {len(experiments)} experiments from config")

    only_datasets = _parse_csv_list(args.dataset)
    only_query_types = _parse_csv_list(args.query_types)
    experiments = filter_experiments(experiments, only_datasets, only_query_types)

    if only_datasets:
        print(f"Filter --dataset:      {', '.join(only_datasets)}")
    if only_query_types:
        print(f"Filter --query-types:  {', '.join(only_query_types)}")

    if not experiments:
        print("ERROR: No experiments matched the filters")
        sys.exit(1)

    print(f"Selected {len(experiments)} experiments\n")

    only_methods = _parse_csv_list(args.methods) or None
    if only_methods:
        print(f"Filter --methods:      {', '.join(only_methods)}")

    if not args.skip_preprocess:
        preprocess_graphs(experiments, gcare_home)
    else:
        print("Skipping preprocessing.\n")

    if not args.skip_build_stats:
        build_all_statistics(experiments, gcare_home, only_methods)
    else:
        print("Skipping statistics build.\n")

    errors = run_experiments(experiments, results_dir, gcare_home, args.max_parallel, only_methods)
    print_summary(errors)


if __name__ == "__main__":
    main()
