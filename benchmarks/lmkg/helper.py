"""Shared utilities for LMKG-S benchmark runner scripts."""

import random
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_LMKG = PROJECT_ROOT / "benchmarks/lmkg"
LMKG_S_DIR = BENCHMARKS_LMKG / "LMKG-Code/lmkg-s"
CONFIG_PATH = BENCHMARKS_LMKG / "experiments_config.yaml"

sys.path.insert(0, str(BENCHMARKS_LMKG))

import run_lmkgs


def get_train_sizes(dataset_or_fold, qtype):
    """Discover query sizes from training files for a dataset or fold."""
    subdir = "star" if qtype == "star" else "path"
    prefix = "star" if qtype == "star" else "chain"
    type_dir = LMKG_S_DIR / "final_datasets" / f"generated_{dataset_or_fold}" / subdir
    sizes = []
    if type_dir.exists():
        for f in sorted(type_dir.glob(f"{prefix}_*.txt")):
            if f.name.startswith("eval_"):
                continue
            try:
                sizes.append(int(f.stem.split("_")[-1]))
            except ValueError:
                continue
    return sorted(sizes)


def classify_sizes(dataset_or_fold):
    """Compute combined/star-only/path-only size sets from training files.

    Returns a dict with keys "combined", "star", "path", each a sorted list.
    """
    star_sizes = set(get_train_sizes(dataset_or_fold, "star"))
    path_sizes = set(get_train_sizes(dataset_or_fold, "path"))
    result = {
        "combined": sorted(star_sizes & path_sizes),
        "star": sorted(star_sizes - path_sizes),
        "path": sorted(path_sizes - star_sizes),
    }
    print(f"  combined sizes: {result['combined']}")
    print(f"  star-only sizes: {result['star']}")
    print(f"  path-only sizes: {result['path']}")
    return result


def train_models(dataset, sizes_by_type, errors, *, epochs, batch_size, layers, neurons, run_id=None):
    """Train LMKG-S models for all query types and sizes."""
    type_to_qtype = {"combined": "combined", "star": "star", "path": "chain"}
    for type_key, qtype in type_to_qtype.items():
        for size in sizes_by_type[type_key]:
            args = [
                "--training",
                "--dataset", dataset,
                "--query-type", qtype,
                "--query-join", str(size),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--layers", str(layers),
                "--neurons", str(neurons),
            ]
            if run_id:
                args += ["--run-id", run_id]
            rc = run_lmkgs.run_lmkgs(args)
            if rc != 0:
                msg = f"Training failed: dataset={dataset}, type={qtype}, size={size}"
                print(f"WARNING: {msg}")
                errors.append(msg)


def _get_eval_test_mode(dataset, size):
    """Determine which eval files exist for a combined-trained size.

    Returns 'all', 'star', 'chain', or None if no eval files exist.
    """
    import glob
    base = LMKG_S_DIR / "final_datasets" / f"generated_{dataset}"
    has_star = bool(glob.glob(str(base / "star" / f"eval_*_star_{size}.txt")))
    has_chain = bool(glob.glob(str(base / "path" / f"eval_*_chain_{size}.txt")))
    if has_star and has_chain:
        return "all"
    if has_star:
        return "star"
    if has_chain:
        return "chain"
    return None


def eval_models(dataset, sizes_by_type, errors, *, layers, neurons, run_id=None):
    """Evaluate LMKG-S models for all query types and sizes."""
    type_to_qtype = {"combined": "combined", "star": "star", "path": "chain"}
    for type_key, qtype in type_to_qtype.items():
        for size in sizes_by_type[type_key]:
            args = [
                "--eval",
                "--dataset", dataset,
                "--query-type", qtype,
                "--query-join", str(size),
                "--layers", str(layers),
                "--neurons", str(neurons),
            ]
            if qtype == "combined":
                test_mode = _get_eval_test_mode(dataset, size)
                if test_mode is None:
                    msg = f"Eval skipped (no eval files): dataset={dataset}, type=combined, size={size}"
                    print(f"WARNING: {msg}")
                    errors.append(msg)
                    continue
                args += ["--test-mode", test_mode]
            if run_id:
                args += ["--run-id", run_id]
            rc = run_lmkgs.run_lmkgs(args)
            if rc != 0:
                msg = f"Eval failed: dataset={dataset}, type={qtype}, size={size}"
                print(f"WARNING: {msg}")
                errors.append(msg)


def train_model_alltypessizes(fold_name, errors, *, epochs, batch_size, layers, neurons, run_id=None):
    """Train one model per fold using alltypessizes (all query types + sizes)."""
    args = [
        "--training",
        "--dataset", fold_name,
        "--query-type", "alltypessizes",
        "--query-join", "2",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--layers", str(layers),
        "--neurons", str(neurons),
    ]
    if run_id:
        args += ["--run-id", run_id]
    rc = run_lmkgs.run_lmkgs(args)
    if rc != 0:
        msg = f"Training failed: dataset={fold_name}, type=alltypessizes"
        print(f"WARNING: {msg}")
        errors.append(msg)


def eval_model_alltypessizes(fold_name, errors, *, layers, neurons, run_id=None):
    """Evaluate the alltypessizes model for a fold."""
    args = [
        "--eval",
        "--dataset", fold_name,
        "--query-type", "alltypessizes",
        "--query-join", "2",
        "--layers", str(layers),
        "--neurons", str(neurons),
    ]
    if run_id:
        args += ["--run-id", run_id]
    rc = run_lmkgs.run_lmkgs(args)
    if rc != 0:
        msg = f"Eval failed: dataset={fold_name}, type=alltypessizes"
        print(f"WARNING: {msg}")
        errors.append(msg)


def train_model_star_only(fold_name, errors, *, epochs, batch_size, layers, neurons, run_id=None):
    """Train on mega_no_{ds}_star fold (both types in training, star eval only)."""
    args = [
        "--training",
        "--dataset", fold_name + "_star",
        "--query-type", "alltypessizes",
        "--query-join", "2",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--layers", str(layers),
        "--neurons", str(neurons),
    ]
    if run_id:
        args += ["--run-id", run_id]
    rc = run_lmkgs.run_lmkgs(args)
    if rc != 0:
        msg = f"Training failed: dataset={fold_name}_star, type=alltypessizes"
        print(f"WARNING: {msg}")
        errors.append(msg)


def train_model_path_only(fold_name, errors, *, epochs, batch_size, layers, neurons, run_id=None):
    """Train on mega_no_{ds}_path fold (both types in training, path eval only)."""
    args = [
        "--training",
        "--dataset", fold_name + "_path",
        "--query-type", "alltypessizes",
        "--query-join", "2",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--layers", str(layers),
        "--neurons", str(neurons),
    ]
    if run_id:
        args += ["--run-id", run_id]
    rc = run_lmkgs.run_lmkgs(args)
    if rc != 0:
        msg = f"Training failed: dataset={fold_name}_path, type=alltypessizes"
        print(f"WARNING: {msg}")
        errors.append(msg)


def eval_model_star_only(fold_name, errors, *, layers, neurons, run_id=None):
    """Evaluate the alltypessizes model for the _star variant of a fold."""
    args = [
        "--eval",
        "--dataset", fold_name + "_star",
        "--query-type", "alltypessizes",
        "--query-join", "2",
        "--layers", str(layers),
        "--neurons", str(neurons),
    ]
    if run_id:
        args += ["--run-id", run_id]
    rc = run_lmkgs.run_lmkgs(args)
    if rc != 0:
        msg = f"Eval failed: dataset={fold_name}_star, type=alltypessizes"
        print(f"WARNING: {msg}")
        errors.append(msg)


def eval_model_path_only(fold_name, errors, *, layers, neurons, run_id=None):
    """Evaluate the alltypessizes model for the _path variant of a fold."""
    args = [
        "--eval",
        "--dataset", fold_name + "_path",
        "--query-type", "alltypessizes",
        "--query-join", "2",
        "--layers", str(layers),
        "--neurons", str(neurons),
    ]
    if run_id:
        args += ["--run-id", run_id]
    rc = run_lmkgs.run_lmkgs(args)
    if rc != 0:
        msg = f"Eval failed: dataset={fold_name}_path, type=alltypessizes"
        print(f"WARNING: {msg}")
        errors.append(msg)


def convert_all_datasets(seed):
    """Convert all FICE datasets in config → LMKG-S per-dataset query files + stats.

    Returns a dict mapping dataset_name → {num_entities, num_relations, star_sizes, path_sizes}.
    """
    import fice_to_lmkg

    random.seed(seed)
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    dataset_stats = {}
    for exp in config["experiments"]:
        by_size, num_entities, num_relations = fice_to_lmkg.process_experiment(exp)
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
        fice_to_lmkg.write_stats_file(
            dname,
            stats["num_entities"],
            stats["num_relations"],
            stats["star_sizes"],
            stats["path_sizes"],
        )

    return dataset_stats


def print_summary(errors):
    """Print summary and exit with error code if there were failures."""
    print("\n" + "=" * 60)
    if errors:
        print(f"Done with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")
