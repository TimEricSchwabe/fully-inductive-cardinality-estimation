"""Micro-benchmark for offline embedding generation.

Loads checkpoint + cached factor graph once, then runs the encoder forward pass
N times and reports median / p99 wall-clock.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.checkpoint_helper import load_checkpoint
from utils.config import Config
from utils.data_helper import get_or_build_graph, load_dataset
from prepare import (
    build_fice_from_config,
    compute_embeddings_full_graph,
    compute_embeddings_batched,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size from config. 0 = full-graph mode; >0 = batched neighbor sampling.")
    args = p.parse_args()

    with open(args.config) as f:
        inf_cfg = yaml.safe_load(f)

    device_str = inf_cfg.get("device", "cpu")
    device = torch.device(
        device_str if device_str != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    # Phase 1: checkpoint load
    t0 = time.perf_counter()
    ckpt = load_checkpoint(Path(inf_cfg["checkpoint"]))
    cfg = Config(**ckpt["config"])
    model = build_fice_from_config(cfg, state_dict=ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    t_ckpt = time.perf_counter() - t0
    print(f"checkpoint load       : {t_ckpt*1000:8.1f} ms")

    # Phase 2: graph load (cached factor/bipartite)
    t0 = time.perf_counter()
    graph = load_dataset(inf_cfg["graph_dataset"])
    mode = "factor" if cfg.relation_gnn.source == "factor" else "bipartite"
    graph_data = get_or_build_graph(graph, inf_cfg["graph_dataset"], mode)
    num_entities = graph_data.original_num_nodes
    num_relations = graph_data.original_num_relations
    t_graph = time.perf_counter() - t0
    print(f"graph load (cached)   : {t_graph*1000:8.1f} ms")
    print(f"  |V|={num_entities:>8d}   |R|={num_relations:>5d}   |E|={graph.num_edges:>10d}")

    batch_size = args.batch_size if args.batch_size is not None else inf_cfg.get("batch_size", 0)
    num_neighbors = cfg.GNNNeighborhood or [100, 100]

    def run_once():
        if batch_size <= 0:
            return compute_embeddings_full_graph(model, graph_data, device)
        return compute_embeddings_batched(
            model, graph_data, device, batch_size, num_neighbors,
        )

    mode_label = "full-graph" if batch_size <= 0 else f"batched (bs={batch_size}, fanout={num_neighbors})"
    print(f"mode                  : {mode_label}")

    # Warm up (first call pays one-time allocation + kernel JIT on torch)
    with torch.no_grad():
        _ = run_once()

    # Phase 3: timed forward passes
    times = []
    for i in range(args.n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ent_emb, rel_emb = run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  run {i+1}/{args.n_runs}            : {dt*1000:8.1f} ms")

    times.sort()
    n = len(times)
    median = times[n // 2]
    p99 = times[-1]  # for small N, max is the best tail estimate we have
    mean = sum(times) / n
    print()
    print(f"encoder forward pass  : median={median*1000:.1f} ms  mean={mean*1000:.1f} ms  max={p99*1000:.1f} ms  (N={n})")

    # Disk footprint
    ent_bytes = ent_emb.numel() * 4
    rel_bytes = rel_emb.numel() * 4
    occ_bytes = (num_entities + num_relations) * 3 * 4
    total = ent_bytes + rel_bytes + occ_bytes
    print()
    print("on-disk footprint (fp32):")
    print(f"  entity embeddings   : {ent_bytes/1024/1024:8.2f} MB  ({num_entities} x {ent_emb.shape[1]})")
    print(f"  relation embeddings : {rel_bytes/1024/1024:8.2f} MB  ({num_relations} x {rel_emb.shape[1]})")
    print(f"  occurrences (3 cols): {occ_bytes/1024/1024:8.2f} MB")
    print(f"  TOTAL               : {total/1024/1024:8.2f} MB")


if __name__ == "__main__":
    main()
