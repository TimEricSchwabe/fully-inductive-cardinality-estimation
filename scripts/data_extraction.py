"""Extract per-query decoder data from trained FICE models.

For every ``final_model.pt`` (or explicitly given checkpoint) and every factor
graph / query set in its training config, this script runs all queries through
the model once (eval mode, no gradients) and stores per query:

* the decoder (``query_gnn``) inputs, i.e. the embeddings computed by the
  embedding GNN on the factor graph:
    - ``node_embeddings``      [num_nodes, d]   entity embeddings (zeros for variables)
    - ``relation_embeddings``  [num_triples, d] relation embeddings per triple pattern
    - ``node_counts`` / ``edge_counts`` [.., 3]  count features as fed to the decoder (log1p)
* ``gnn_output``           raw output of ``query_gnn``
* ``fice_prediction``      value returned by ``FICE.forward`` (log-cardinality)
  and ``fice_prediction_relu`` (= what evaluation in train.py uses)
* ``true_cardinality`` and ``true_log_cardinality`` (= log1p, the training target)
* query structure: local ``edge_index``, global ``relation_ids`` / ``entity_ids``
  (-1 = variable), ``query_type``, ``query_idx`` (position in the query set)

With ``prediction_mode: log-cardinality`` ``gnn_output`` and ``fice_prediction``
are identical; the script checks this and reports the maximum difference.

The capture uses ``fice.decoder_capture.DecoderCapture`` (forward hooks on
``model.query_gnn``), so the stored embeddings are exactly what the decoder saw.
Afterwards ``verify_reproduction`` replays the decoder on the captured inputs.

Output per run (default ``output/extraction/<experiment>/<run>/``):
    extraction.pt   dict(format, meta, queries=[per-query dicts], summary={flat tensors})
    queries.csv     one row per query (scalars only) for a quick look

Usage (from the repository root, like train.py):
    python scripts/data_extraction.py                                  # all final models under output/ws2627
    python scripts/data_extraction.py --runs-root path/to/drive/runs   # e.g. Johannes' runs from Drive
    python scripts/data_extraction.py --checkpoint output/ws2627/fano_all_s2/run_.../checkpoints/final_model.pt
    python scripts/data_extraction.py --checkpoint-name best_model.pt  # other checkpoint file name

Loading:
    blob = torch.load("output/extraction/fano_all_s2/run_.../extraction.pt", weights_only=False)
    q = blob["queries"][0]; q["relation_embeddings"], q["gnn_output"], q["true_cardinality"]
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_geometric.loader import DataLoader  # noqa: E402

from fice.decoder_capture import DecoderCapture, verify_reproduction  # noqa: E402
from fice.models import FICE  # noqa: E402
from train import initialize_model_components, prepare_batch  # noqa: E402
from utils.config import Config, load_config_yaml  # noqa: E402
from utils.data_helper import get_or_build_graph, load_data, preprocess_data  # noqa: E402
from utils.helper import get_device, process_seed, set_random_seeds  # noqa: E402
from utils.models import TrainingContext  # noqa: E402
from utils.neighborhood_sampler import MultigraphNeighborSampler  # noqa: E402

logger = logging.getLogger("data_extraction")

FORMAT = "fice_data_extraction_v1"
NUM_COUNT_FEATURES = 3
# attributes of the query batch stored by DecoderCapture (at decoder-call time)
EXTRA_ATTRS = ("y", "original_edge_embedding_indices", "original_node_embedding_indices")


# --------------------------------------------------------------------------- helpers
def find_checkpoints(runs_roots: List[Path], checkpoint_name: str) -> List[Path]:
    found = []
    for root in runs_roots:
        if root.is_file():
            found.append(root)
            continue
        found.extend(sorted(root.glob(f"**/checkpoints/{checkpoint_name}")))
    return found


def load_run_config(checkpoint: Dict[str, Any], run_dir: Path) -> Config:
    """Prefer the config stored in the checkpoint (exactly what was trained)."""
    if checkpoint.get("config"):
        return Config(**checkpoint["config"])
    for name in ("config.yaml", "config.yml"):
        if (run_dir / name).exists():
            return load_config_yaml(run_dir / name)
    raise FileNotFoundError(f"No config in checkpoint and no config.yaml in {run_dir}")


def unique_query_sets(config: Config):
    """(dataset, queries, batch_size, num_queries) per distinct factor graph / query set.
    train.py lists the same dataset as train AND validation; extract it only once."""
    seen, result = set(), []
    for dc in config.data:
        key = (dc.dataset, dc.queries)
        if key in seen:
            continue
        seen.add(key)
        result.append(dc)
    return result


def as_list(value, n: int) -> List[Any]:
    """Batch attributes of non-tensor type (type, query_set_name) collate to lists."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


def split_record(rec, d: int) -> List[Dict[str, Any]]:
    """Split one decoder call (= one mini-batch) into per-query dicts.

    Node- and edge-level tensors are split explicitly via ``batch`` and the
    edge -> query map (DecoderRecord.split_queries guesses the level from the
    tensor length, which is ambiguous e.g. for cycles with #nodes == #edges)."""
    n_queries = rec.num_queries
    edge_batch = rec.batch[rec.edge_index[0]]
    out = rec.output.view(-1) if rec.output is not None else None
    pred = rec.prediction.view(-1) if rec.prediction is not None else None
    y = rec.extras["y"].view(-1).to(torch.float64)
    rel_ids = rec.extras.get("original_edge_embedding_indices")
    ent_ids = rec.extras.get("original_node_embedding_indices")

    items = []
    for g in range(n_queries):
        node_idx = (rec.batch == g).nonzero(as_tuple=False).view(-1)
        emask = edge_batch == g
        remap = torch.full((rec.x.size(0),), -1, dtype=torch.long)
        remap[node_idx] = torch.arange(node_idx.numel())
        x_g, e_g = rec.x[node_idx], rec.edge_attr[emask]
        true_card = float(y[g])
        fice_pred = float(pred[g]) if pred is not None else math.nan
        items.append({
            "num_nodes": int(node_idx.numel()),
            "num_triples": int(emask.sum()),
            "edge_index": remap[rec.edge_index[:, emask]],
            "relation_ids": rel_ids[emask].clone() if rel_ids is not None else None,
            "entity_ids": ent_ids[node_idx].clone() if ent_ids is not None else None,
            "node_counts": x_g[:, :NUM_COUNT_FEATURES].clone(),
            "edge_counts": e_g[:, :NUM_COUNT_FEATURES].clone(),
            "node_embeddings": x_g[:, -d:].clone(),
            "relation_embeddings": e_g[:, -d:].clone(),
            "gnn_output": float(out[g]) if out is not None else math.nan,
            "fice_prediction": fice_pred,
            "fice_prediction_relu": max(fice_pred, 0.0),
            "true_cardinality": true_card,
            "true_log_cardinality": math.log1p(true_card),
        })
    return items


# --------------------------------------------------------------------------- core
@torch.no_grad()
def extract_query_set(model, config: Config, data_config, device, verify: bool) -> Dict[str, Any]:
    dataset, (_eo, _ro, entity_mapping, relation_mapping), queries = load_data(
        dataset_name=data_config.dataset, queries_name=data_config.queries
    )
    # same filtering/truncation as in training (prepare_data)
    queries = preprocess_data(queries, min_cardinality=config.min_cardinality,
                              num_queries=data_config.num_queries)
    if config.relation_gnn.source == "factor":
        dataset.factor_graph = get_or_build_graph(dataset, data_config.dataset, "factor")
    else:
        dataset.bipartite_graph = get_or_build_graph(dataset, data_config.dataset, "bipartite")

    sampler = MultigraphNeighborSampler(num_neighbors=config.GNNNeighborhood)
    loader = DataLoader(queries, batch_size=data_config.batch_size, shuffle=False)
    d = config.embedding_dim

    records, items, offset = [], [], 0
    cap = DecoderCapture(model, extra_attrs=EXTRA_ATTRS)
    with cap:
        for query_batch in loader:
            n = query_batch.num_graphs
            prepared = prepare_batch(dataset, query_batch, config, sampler, type(model))
            graph = prepared.entity_graph.to(device) if prepared.entity_graph is not None else None
            qb = prepared.query_batch.to(device)
            graph_size = torch.tensor([prepared.graph_num_edges], device=device)

            cap.clear()
            model(data=graph, relation_graph=graph, batch=qb, graph_size=graph_size)
            if len(cap.records) != 1:
                raise RuntimeError(f"expected 1 decoder call per batch, got {len(cap.records)}")
            rec = cap.records[0]

            types = as_list(getattr(query_batch, "type", None), n)
            set_names = as_list(getattr(query_batch, "query_set_name", None), n)
            for g, item in enumerate(split_record(rec, d)):
                item.update({
                    "query_idx": offset + g,
                    "dataset": data_config.dataset,
                    "query_set_name": set_names[g],
                    "query_type": types[g],
                })
                items.append(item)
            if verify:
                records.append(rec)
            offset += n

    report = None
    if verify:
        report = verify_reproduction(model.query_gnn, records, raise_on_fail=False)
        if not report["ok"]:
            logger.warning(f"  reproduction check FAILED: {report}")

    return {
        "dataset": data_config.dataset,
        "queries_name": data_config.queries,
        "num_queries": len(items),
        "graph_num_edges": int(dataset.num_edges),
        "relation_to_id": relation_mapping,
        "entity_to_id": entity_mapping,
        "verify_report": report,
        "items": items,
    }


def summarize(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    col = lambda k, dt=torch.float64: torch.tensor([it[k] for it in items], dtype=dt)
    return {
        "query_idx": col("query_idx", torch.long),
        "num_triples": col("num_triples", torch.long),
        "gnn_output": col("gnn_output"),
        "fice_prediction": col("fice_prediction"),
        "fice_prediction_relu": col("fice_prediction_relu"),
        "true_cardinality": col("true_cardinality"),
        "true_log_cardinality": col("true_log_cardinality"),
    }


def write_csv(items: List[Dict[str, Any]], path: Path) -> None:
    cols = ["dataset", "query_set_name", "query_idx", "query_type", "num_nodes", "num_triples",
            "relation_ids", "true_cardinality", "true_log_cardinality",
            "gnn_output", "fice_prediction", "fice_prediction_relu"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for it in items:
            row = []
            for c in cols:
                v = it[c]
                if isinstance(v, torch.Tensor):
                    v = " ".join(str(int(t)) for t in v.tolist())
                row.append(v)
            w.writerow(row)


def process_checkpoint(ckpt_path: Path, out_root: Path, device, verify: bool, skip_existing: bool) -> Optional[Path]:
    run_dir = ckpt_path.parent.parent                 # <experiment>/<run>/checkpoints/<file>
    experiment, run_name = run_dir.parent.name, run_dir.name
    out_dir = out_root / experiment / run_name
    if ckpt_path.name != "final_model.pt":
        out_dir = out_dir / ckpt_path.stem
    out_file = out_dir / "extraction.pt"
    if skip_existing and out_file.exists():
        logger.info(f"skip (exists): {out_file}")
        return out_file

    logger.info(f"=== {experiment}/{run_name} ({ckpt_path.name})")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = load_run_config(checkpoint, run_dir)
    if config.model != "FICE":
        logger.warning(f"  model type {config.model} is not FICE, skipped")
        return None
    if config.prediction_mode != "log-cardinality":
        logger.warning(f"  prediction_mode={config.prediction_mode}: gnn_output != fice_prediction expected")

    set_random_seeds(process_seed(config.seed))
    context = TrainingContext(config=config, device=device, run_dir=run_dir,
                              log_dir=run_dir / "logs", checkpoint_dir=ckpt_path.parent)
    model = initialize_model_components(context, total_training_steps=1,
                                        model_state_dict=checkpoint["model_state_dict"]).model
    model.eval()
    if not isinstance(model, FICE):
        logger.warning("  not a FICE model, skipped")
        return None

    query_sets, all_items = [], []
    for dc in unique_query_sets(config):
        res = extract_query_set(model, config, dc, device, verify)
        all_items.extend(res.pop("items"))
        query_sets.append(res)

    s = summarize(all_items)
    diff = (s["gnn_output"] - s["fice_prediction"]).abs().max().item() if all_items else math.nan
    mae = (s["fice_prediction_relu"] - s["true_log_cardinality"]).abs().mean().item() if all_items else math.nan
    logger.info(f"  {len(all_items)} queries | max|gnn_output - fice_prediction| = {diff:.3g} "
                f"| MAE(log1p) = {mae:.4f}")

    meta = {
        "checkpoint": str(ckpt_path),
        "experiment": experiment,
        "run": run_name,
        "seed": config.seed,
        "conditioning": config.Conditioning,
        "prediction_mode": config.prediction_mode,
        "embedding_dim": config.embedding_dim,
        "config": config.model_dump(),
        "query_sets": query_sets,
        "max_abs_diff_gnn_output_vs_prediction": diff,
        "mae_log1p": mae,
        "device": str(device),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "meta": meta, "queries": all_items, "summary": s}, out_file)
    write_csv(all_items, out_dir / "queries.csv")
    logger.info(f"  saved {out_file}")
    return out_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, nargs="*", default=[REPO_ROOT / "output" / "ws2627"],
                    help="folders searched recursively for */checkpoints/<checkpoint-name>")
    ap.add_argument("--checkpoint", type=Path, nargs="*", default=None,
                    help="explicit checkpoint file(s); overrides --runs-root")
    ap.add_argument("--checkpoint-name", default="final_model.pt")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "extraction")
    ap.add_argument("--device", default="cpu", help="cpu (default, deterministic), cuda, mps or auto")
    ap.add_argument("--no-verify", action="store_true", help="skip the decoder reproduction check")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device(args.device)

    ckpts = args.checkpoint if args.checkpoint else find_checkpoints(args.runs_root, args.checkpoint_name)
    if not ckpts:
        logger.error(f"no checkpoints named {args.checkpoint_name} found under {args.runs_root}")
        sys.exit(1)
    logger.info(f"{len(ckpts)} checkpoint(s) found")

    failed = []
    for ckpt in ckpts:
        try:
            process_checkpoint(Path(ckpt), args.output_dir, device,
                               verify=not args.no_verify, skip_existing=args.skip_existing)
        except Exception as exc:  # keep going with the other runs
            logger.exception(f"failed: {ckpt}: {exc}")
            failed.append(str(ckpt))
    if failed:
        logger.error(f"{len(failed)} run(s) failed:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
