"""Run G-Care cardinality estimation (wj + cset) for a single experiment.

Loads *num_queries* queries from the FICE JSON file, takes the last 20 % as the
evaluation split (discarding the training portion – G-Care has no training step),
and runs the G-Care binary for each query.

Can be used as a CLI script or imported and called as run_estimation().
"""
import argparse
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from transform_query import query_to_gcare

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_query_json(query_path: Path) -> Path:
    """Return a query JSON file given either a direct file path or a directory.

    Handles three cases:
    - explicit .json file   → returned as-is
    - file without extension (e.g. numeric IDs like 60037037) → returned as-is
    - directory            → first *.json match inside it
    """
    if query_path.is_file():
        return query_path
    if query_path.is_dir():
        candidates = sorted(query_path.glob("*.json"))
        if not candidates:
            candidates = sorted(query_path.rglob("*.json"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No query JSON file found at {query_path}")


def _eval_split(queries: list, num_queries: int, seed: int = 42) -> list:
    """Sample *num_queries* then return the last 20 % as the evaluation set.

    This mirrors the 80/20 split used by the other benchmarks.  G-Care has no
    training step so we only need the eval portion.
    """
    rng = random.Random(seed)
    sample_size = min(num_queries, len(queries))
    sampled = rng.sample(queries, sample_size)
    # Consistent deterministic shuffle so the split boundary is stable.
    rng.shuffle(sampled)
    split_idx = int(0.8 * len(sampled))
    return sampled[split_idx:]


def _predict(method: str, dataset: str, gcare_home: Path) -> Tuple[float, float]:
    """Call the G-Care binary for the single query currently in the queryset dir.

    Returns (predicted_cardinality, elapsed_seconds).
    """
    seed = 0
    result_dir = gcare_home / "data" / "result" / "accuracy" / "current"
    result_dir.mkdir(parents=True, exist_ok=True)

    p = 0.03
    repeat = 1 if method in ("cset", "bsk", "sumrdf") else 30

    if method == "bsk":
        os.environ["GCARE_BSK_BUDGET"] = "4096"

    command = (
        f"./run-exp.sh {method} {dataset} {p} {seed} {repeat} {result_dir}"
    )
    logging.info("Running: %s", command)

    start = time.time()
    proc = subprocess.run(
        f'bash -lc "{command}"',
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        cwd=str(gcare_home / "scripts"),
    )
    elapsed = time.time() - start

    if "GCARE_BSK_BUDGET" in os.environ:
        del os.environ["GCARE_BSK_BUDGET"]

    output = proc.stdout.decode().strip()
    values = output.split()
    predicted = np.mean(np.asarray(values[-repeat:], dtype=float))
    return float(predicted), elapsed


# ---------------------------------------------------------------------------
# Core estimation function
# ---------------------------------------------------------------------------

def run_estimation(
    *,
    exp_name: str,
    dataset_name: str,
    query_type: str,
    query_path: Path,
    rdf_type_uri: str,
    num_queries: int,
    results_dir: Path,
    gcare_home: Path,
) -> None:
    """Run wj + cset estimation for one experiment and write results to *results_dir*."""

    logging.info(
        "Starting estimation: exp=%s dataset=%s query_type=%s num_queries=%d",
        exp_name, dataset_name, query_type, num_queries,
    )

    gcare_dataset_dir = gcare_home / "data" / "dataset" / exp_name
    gcare_queryset_dir = gcare_home / "data" / "queryset" / exp_name

    id2id_path = gcare_dataset_dir / "id_to_id_mapping.json"
    id2id_pred_path = gcare_dataset_dir / "id_to_id_mapping_predicate.json"

    for p in (id2id_path, id2id_pred_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing ID mapping: {p}")

    # ------------------------------------------------------------------
    # Load queries and take the 20 % eval split
    # ------------------------------------------------------------------
    raw_json = _find_query_json(query_path)
    logging.info("[%s] Loading queries from %s", exp_name, raw_json)
    with raw_json.open() as f:
        all_queries = json.load(f)

    eval_queries = _eval_split(all_queries, num_queries)
    logging.info("[%s] Eval queries: %d (from %d total)", exp_name, len(eval_queries), len(all_queries))

    with id2id_path.open() as f:
        id_to_id = json.load(f)
    with id2id_pred_path.open() as f:
        id_to_id_pred = json.load(f)

    # ------------------------------------------------------------------
    # Run each method
    # ------------------------------------------------------------------
    methods = ["wj", "cset"]
    for method in methods:
        logging.info("[%s] Running method: %s", exp_name, method)
        preds: List[float] = []
        gts: List[int] = []
        sizes: List[int] = []
        pred_times: List[float] = []
        result_records = []

        method_dir = results_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        for i, query in enumerate(eval_queries):
            # Write this query as the single file in the queryset directory.
            for old in gcare_queryset_dir.glob("*.txt"):
                old.unlink()

            query_to_gcare(
                query["triples"],
                0,
                id_to_id_mapping=id_to_id,
                id_to_id_mapping_predicate=id_to_id_pred,
                dataset=exp_name,
                card=query["y"],
                rdf_type_uri=rdf_type_uri,
                gcare_home=gcare_home,
            )

            try:
                y_pred, pred_time = _predict(method, exp_name, gcare_home)
            except subprocess.CalledProcessError as exc:
                logging.error("[%s] %s query %d failed: %s", exp_name, method, i, exc)
                continue

            preds.append(y_pred)
            gts.append(query["y"])
            pred_times.append(pred_time)
            sizes.append(len(query["triples"]))
            result_records.append({
                **query,
                "y_pred": y_pred,
                "exec_time_total": pred_time,
            })

            logging.info(
                "[%s] %s %d/%d  size=%d  y=%s  y_pred=%.1f  t=%.2fs",
                exp_name, method, i + 1, len(eval_queries),
                len(query["triples"]), query["y"], y_pred, pred_time,
            )

        np.save(method_dir / "preds.npy", preds)
        np.save(method_dir / "gts.npy", gts)
        np.save(method_dir / "sizes.npy", sizes)
        np.save(method_dir / "pred_times.npy", pred_times)
        with (method_dir / "results.json").open("w") as f:
            json.dump(result_records, f, indent=2)

        logging.info("[%s] %s done – %d predictions written to %s", exp_name, method, len(preds), method_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run G-Care estimation for one experiment")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--query-type", required=True)
    parser.add_argument("--query-path", type=Path, required=True,
                        help="Path to the FICE query JSON file or its parent directory")
    parser.add_argument("--rdf-type-uri", required=True)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--gcare-home",
        type=Path,
        default=Path(__file__).resolve().parent / "gcare",
        help="Path to compiled G-Care directory (default: ./gcare/)",
    )
    args = parser.parse_args()

    run_estimation(
        exp_name=args.exp_name,
        dataset_name=args.dataset_name,
        query_type=args.query_type,
        query_path=args.query_path,
        rdf_type_uri=args.rdf_type_uri,
        num_queries=args.num_queries,
        results_dir=args.results_dir,
        gcare_home=args.gcare_home,
    )


if __name__ == "__main__":
    main()
