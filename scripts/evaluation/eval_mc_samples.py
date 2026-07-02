"""Evaluate a trained per_node checkpoint with varying test-time Monte Carlo
sample counts (mc_samples), without retraining.

For each M in --mc-samples the test split is re-encoded averaging M
neighborhoods per node (the fixed first M of the pool, as at eval time) and
q-error statistics are reported. Writes mc_sweep.csv next to the checkpoint.

Usage (from scripts/):
  python evaluation/eval_mc_samples.py --config ../configs/wn18rr_comparison/gt_mc32_16.yaml \
      --checkpoint <run_dir>/checkpoints/best_model.pt --mc-samples 1 2 4 8 16 32
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config_yaml
from utils.data_helper import prepare_data
from utils.helper import get_device, process_seed, set_random_seeds
from utils.models import TrainingContext
from train import initialize_model_components, make_prepare_fn, PrefetchLoader


def qerror(pred, true):
    pred = np.clip(pred, 1e-9, None)
    true = np.clip(true, 1e-9, None)
    return np.maximum(pred / true, true / pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mc-samples", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    config = load_config_yaml(args.config)
    device = get_device(config.device)
    # Same seeding as train.py main() so the train/val query split is identical
    set_random_seeds(process_seed(config.seed))

    # prepare_data BEFORE model init, mirroring train.py main() RNG order
    data_components = prepare_data(config)
    loader = data_components.test_loader

    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    context = TrainingContext(config=config, device=device, run_dir=Path("."),
                              log_dir=Path("."), checkpoint_dir=Path("."))
    model = initialize_model_components(
        context, total_training_steps=1,
        model_state_dict=ckpt["model_state_dict"],
    ).model
    model.eval()

    rows = []
    for m in args.mc_samples:
        if m > config.mc_pool_size:
            print(f"skipping M={m} (> mc_pool_size {config.mc_pool_size})")
            continue
        cfg_m = copy.deepcopy(config)
        cfg_m.mc_samples = m
        prepare_fn = make_prepare_fn(cfg_m, data_components.neighbor_sampler,
                                     type(model), training=False)
        preds, trues = [], []
        with torch.no_grad():
            for prepared in PrefetchLoader(loader, prepare_fn, prefetch=2):
                graph = prepared.entity_graph.to(device) if prepared.entity_graph is not None else None
                query_batch = prepared.query_batch.to(device)
                graph_size = torch.tensor([prepared.graph_num_edges], device=device)
                log_pred = model(data=graph, relation_graph=graph, batch=query_batch,
                                 graph_size=graph_size).to(torch.float32)
                preds.append(torch.expm1(log_pred).cpu().numpy())
                trues.append(query_batch.y.cpu().numpy())
        qe = qerror(np.concatenate(preds), np.concatenate(trues))
        row = {
            "mc_samples": m,
            "qerror_mean": float(qe.mean()),
            "qerror_p50": float(np.percentile(qe, 50)),
            "qerror_p90": float(np.percentile(qe, 90)),
            "qerror_p99": float(np.percentile(qe, 99)),
            "n_queries": int(len(qe)),
        }
        rows.append(row)
        print(row)

    out = Path(args.checkpoint).parent.parent / "mc_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
