#!/usr/bin/env python3
"""
LOCO evaluation for LSS — called via subprocess from helper.py.

Loads the held-out eval queryset, builds tensors using the training DataGraph
(so label space matches the trained model), then runs LSS's native evaluate().
Outputs a JSON predictions file.
"""
import sys
import json
import math
import time
import argparse
from pathlib import Path

LSS_DIR = Path(__file__).resolve().parent
LSS_CODE_DIR = LSS_DIR / "LSS"
for _p in [str(LSS_CODE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.optim as optim
from Queryset import Queryset
from QuerySampler import QueryDecompose
from active import ActiveLearner, _to_datasets
import cardnet
from util import load_model


def _build_args(exp_id, mega_dir, epochs, cuda):
    mega_dir = Path(mega_dir)
    train_name = f"train_{exp_id}"
    args = argparse.Namespace(
        num_layers=3,
        model_type="NNGINConcat",
        embed_type="prone",
        edge_embed_type="freq",
        num_g_hid=128,
        num_e_hid=32,
        out_g_ch=128,
        num_expert=64,
        num_att_hid=64,
        num_mlp_hid=128,
        pool_type="att",
        dropout=0.2,
        cumulative=False,
        num_fold=5,
        epochs=epochs,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=5e-4,
        decay_factor=0.1,
        decay_patience=50,
        weight_exp=1.0,
        num_workers=4,
        multi_task=True,
        max_classes=10,
        coeff=0.5,
        uncertainty="consist",
        biased_sample=True,
        active_iters=0,
        budget=50,
        active_epochs=50,
        ensemble_num=5,
        dataset=train_name,
        full_data_dir=str(mega_dir),
        save_res_dir=str(mega_dir / "result"),
        model_file=f"model_NNGINConcat_{epochs}_nocumu.pth",
        model_save_dir=str(mega_dir / "models"),
        matching="homo",
        k=3,
        verbose=False,
        mode="test",
    )
    args.queryset_homo_dir = str(mega_dir / "queryset_homo")
    args.queryset_iso_dir  = str(mega_dir / "queryset_iso")
    args.true_homo_dir     = str(mega_dir / "true_homo")
    args.true_iso_dir      = str(mega_dir / "true_iso")
    args.data_dir          = str(mega_dir / "dataset")
    args.prone_feat_dir    = str(mega_dir / "prone")
    args.n2v_feat_dir      = str(mega_dir / "n2v")
    args.nrp_feat_dir      = str(mega_dir / "nrp")
    args.embed_feat_dir    = args.prone_feat_dir
    args.queryset_dir      = args.queryset_homo_dir
    args.true_card_dir     = args.true_homo_dir
    args.cuda   = cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id",           required=True)
    parser.add_argument("--mega_dir",         required=True)
    parser.add_argument("--fold_results_dir", required=True)
    parser.add_argument("--eval_ds",          required=True, help="Held-out dataset name for CSV")
    parser.add_argument("--epochs",           type=int,   default=80)
    parser.add_argument("--prone_time_s",     type=float, default=0.0)
    parser.add_argument("--out",              required=True, help="Output JSON path")
    parser.add_argument("--no-cuda",          action="store_true", default=False)
    cli = parser.parse_args()

    cuda             = not cli.no_cuda
    fold_results_dir = Path(cli.fold_results_dir)
    train_name       = f"train_{cli.exp_id}"
    eval_name        = f"eval_{cli.exp_id}"

    args = _build_args(cli.exp_id, cli.mega_dir, cli.epochs, cuda)

    # ── Load eval queries ─────────────────────────────────────────────────────
    print(f"  Decomposing eval queries ({eval_name})...")
    _t0 = time.perf_counter()
    eval_qd = QueryDecompose(
        queryset_dir=args.queryset_dir,
        true_card_dir=args.true_card_dir,
        dataset=eval_name,
        k=args.k,
    )
    eval_qd.decomose_queries()
    if not eval_qd.all_subsets:
        raise RuntimeError(f"No eval queries found for {eval_name}")

    # Drop single-triple queries (CardNet squeeze() collapses them to 1-D)
    eval_subsets = {k: v for k, v in eval_qd.all_subsets.items() if k[1] >= 2}

    # Build Queryset with training DataGraph + ProNE (args.dataset = train_name)
    print(f"  Building eval Queryset using training DataGraph ({train_name})...")
    eval_qs = Queryset(args=args, all_subsets=eval_subsets)
    eval_qs.print_queryset_info()
    data_load_time_s = time.perf_counter() - _t0

    # ── Load checkpoint ───────────────────────────────────────────────────────
    args.model_save_dir = str(fold_results_dir / "checkpoint")
    ckpt_path = Path(args.model_save_dir) / train_name / args.model_file
    print(f"  Loading checkpoint: {ckpt_path}")

    model     = cardnet.CardNet(args, num_node_feat=eval_qs.num_node_feat,
                                num_edge_feat=eval_qs.num_edge_feat)
    optimizer = optim.Adam(model.parameters())
    load_model(args=args, model=model, device=args.device, optimizer=optimizer)

    criterion      = torch.nn.MSELoss()
    active_learner = ActiveLearner(args)

    # ── Evaluate per (pattern, size) bucket to preserve query_type ───────────
    print(f"  Evaluating on {eval_name}...")
    predictions = []
    query_id    = 0

    for (pattern, size) in sorted(eval_qs.all_subsets.keys()):
        bucket = eval_qs.all_subsets[(pattern, size)]
        if not bucket:
            continue
        bucket_datasets = _to_datasets([bucket], num_classes=args.max_classes)
        res = active_learner.evaluate(model, criterion, bucket_datasets, print_res=False)

        pairs, _, _, elapse_time = res[0]
        avg_eval_time = elapse_time / len(pairs) if pairs else 0.0

        for true_log2, pred_log2 in pairs:
            true_card = int(round(math.pow(2, true_log2)))
            pred_card = max(1, int(round(math.pow(2, pred_log2))))
            predictions.append({
                "query_id":          query_id,
                "query_type":        pattern,
                "dataset":           cli.eval_ds,
                "epoch":             cli.epochs,
                "ground_truth":      true_card,
                "prediction":        pred_card,
                "query_size":        size,
                "prone_time_s":      cli.prone_time_s,
                "data_load_time_s":  data_load_time_s,
                "eval_time_s":       avg_eval_time,
            })
            query_id += 1

    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"  Saved {len(predictions)} predictions → {out_path}")


if __name__ == "__main__":
    main()
