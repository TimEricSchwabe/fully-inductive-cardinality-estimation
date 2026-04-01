"""Benchmark C++ inference: timing + prediction quality.

Runs all queries through the C++ binary in batch mode, collecting per-query
timing (graph construction, model inference, total) and cardinality predictions.
Compares predictions against ground truth and produces figures:
  - Runtime boxplots by query size
  - Scatter plot: true vs predicted cardinality
  - Q-error boxplots by query size

Usage:
    cd scripts
    uv run python ../inference/benchmark_cpp.py \
        --queries ../data/fb237_v4_star_queries/raw/fb237_v4_star_queries \
        --cpp-binary ../inference/build/fice_inference \
        --artifacts-dir ../output/inference/ \
        --output-dir ../output/benchmark/
"""

import json
import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def write_bgp_file(queries, output_path):
    with open(output_path, "w") as f:
        for q in queries:
            parts = []
            for s, p, o in q["triples"]:
                parts.append(f"{s} {p} {o}")
            f.write(" . ".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark C++ FICE inference")
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--cpp-binary", type=str, default="inference/build/fice_inference")
    parser.add_argument("--artifacts-dir", type=str, default="output/inference/")
    parser.add_argument("--output-dir", type=str, default="output/benchmark/")
    parser.add_argument("--warmup", type=int, default=500,
                        help="Number of warmup queries prepended (results discarded)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    logger.info(f"Loading queries from {args.queries}")
    with open(args.queries) as f:
        all_queries = json.load(f)
    logger.info(f"Loaded {len(all_queries)} queries")

    # Write BGP file with warmup queries prepended
    bgp_path = output_dir / "queries.bgp"
    warmup_queries = all_queries[:args.warmup]
    logger.info(f"Prepending {len(warmup_queries)} warmup queries")
    write_bgp_file(warmup_queries + all_queries, bgp_path)

    # Run C++ binary
    logger.info("Running C++ binary...")
    result = subprocess.run(
        [args.cpp_binary, args.artifacts_dir, "--batch", str(bgp_path)],
        capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        logger.error(f"C++ binary failed:\n{result.stderr}")
        return
    logger.info(f"C++ stderr: {result.stderr.strip()}")

    # Parse results, skip warmup rows
    num_triples_list = []
    log_card_list = []
    build_us_list = []
    infer_us_list = []
    total_us_list = []

    lines = result.stdout.strip().split("\n")
    data_lines = [l for l in lines if not l.startswith("query_idx")]
    # Skip warmup entries
    data_lines = data_lines[len(warmup_queries):]

    for line in data_lines:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        num_triples_list.append(int(parts[1]))
        log_card_list.append(float(parts[2]))
        build_us_list.append(float(parts[3]))
        infer_us_list.append(float(parts[4]))
        total_us_list.append(float(parts[5]))

    num_triples = np.array(num_triples_list)
    log_card = np.array(log_card_list)
    build_us = np.array(build_us_list)
    infer_us = np.array(infer_us_list)
    total_us = np.array(total_us_list)

    # Ground truth cardinalities
    ground_truth = np.array([q["y"] for q in all_queries], dtype=np.float64)
    pred_card = np.expm1(np.clip(log_card, 0, None))

    # Q-error: max(pred/true, true/pred), clamp to avoid /0
    gt_clamped = np.maximum(ground_truth, 1.0)
    pred_clamped = np.maximum(pred_card, 1.0)
    q_errors = np.maximum(pred_clamped / gt_clamped, gt_clamped / pred_clamped)

    logger.info(f"Parsed {len(num_triples)} timed queries")

    # ---- Summary statistics ----
    print("\n" + "=" * 70)
    print("C++ Inference Benchmark Results")
    print(f"Queries: {len(num_triples)}")
    print("=" * 70)

    for name, arr in [("Graph construction", build_us),
                      ("Model inference", infer_us),
                      ("Total (build + infer)", total_us)]:
        print(f"\n  {name} (microseconds):")
        print(f"    Mean: {arr.mean():.1f}  Median: {np.median(arr):.1f}  "
              f"Std: {arr.std():.1f}")
        print(f"    Min: {arr.min():.0f}  Max: {arr.max():.0f}  "
              f"P95: {np.percentile(arr, 95):.0f}  "
              f"P99: {np.percentile(arr, 99):.0f}")

    # Q-error summary
    print("\n  Q-Error (predicted vs ground truth):")
    print(f"    Mean: {q_errors.mean():.2f}  Median: {np.median(q_errors):.2f}")
    print(f"    P90: {np.percentile(q_errors, 90):.2f}  "
          f"P95: {np.percentile(q_errors, 95):.2f}  "
          f"P99: {np.percentile(q_errors, 99):.2f}  "
          f"Max: {q_errors.max():.2f}")

    # Per query-size stats
    unique_sizes = sorted(np.unique(num_triples))
    print(f"\n  Query sizes present: {[int(s) for s in unique_sizes]}")
    print(f"\n  {'Size':>5}  {'Count':>6}  {'Build(med)':>10}  "
          f"{'Infer(med)':>10}  {'Total(med)':>10}  {'Q-err(med)':>10}")
    for sz in unique_sizes:
        mask = num_triples == sz
        print(f"  {sz:>5}  {mask.sum():>6}  "
              f"{np.median(build_us[mask]):>10.0f}  "
              f"{np.median(infer_us[mask]):>10.0f}  "
              f"{np.median(total_us[mask]):>10.0f}  "
              f"{np.median(q_errors[mask]):>10.2f}")

    # ---- Write raw TSV ----
    tsv_path = output_dir / "benchmark_results.tsv"
    with open(tsv_path, "w") as f:
        f.write("query_idx\tnum_triples\tground_truth\tpred_cardinality\t"
                "q_error\tgraph_build_us\tinference_us\ttotal_us\n")
        for i in range(len(num_triples)):
            f.write(f"{i}\t{num_triples[i]}\t{ground_truth[i]:.0f}\t"
                    f"{pred_card[i]:.2f}\t{q_errors[i]:.2f}\t"
                    f"{build_us[i]:.0f}\t{infer_us[i]:.0f}\t{total_us[i]:.0f}\n")
    logger.info(f"Raw results written to {tsv_path}")

    # ---- Figure 1: Runtime boxplots by query size ----
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    labels = [str(int(sz)) for sz in unique_sizes]
    for ax, (title, arr) in zip(axes1, [("Graph Construction", build_us),
                                         ("Model Inference", infer_us),
                                         ("Total", total_us)]):
        groups = [arr[num_triples == sz] for sz in unique_sizes]
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)
        ax.set_xlabel("Query Size (number of triples)")
        ax.set_ylabel("Time (\u00b5s)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig1.suptitle("C++ FICE Inference Runtime by Query Size", fontsize=14, y=1.02)
    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(output_dir / f"runtime_by_query_size.{ext}",
                     bbox_inches="tight", dpi=150)
    logger.info(f"Runtime figure saved")

    # ---- Figure 2: Scatter plot true vs predicted cardinality ----
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    # Color by query size
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(unique_sizes), vmax=max(unique_sizes))

    sc = ax2.scatter(ground_truth, pred_card, c=num_triples, cmap=cmap,
                     norm=norm, alpha=0.3, s=8, edgecolors="none")
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("Query Size")

    # Diagonal line
    lo = max(1, min(ground_truth.min(), pred_card.min()))
    hi = max(ground_truth.max(), pred_card.max()) * 1.1
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("True Cardinality")
    ax2.set_ylabel("Predicted Cardinality")
    ax2.set_title("True vs Predicted Cardinality (C++ Inference)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(output_dir / f"true_vs_predicted.{ext}",
                     bbox_inches="tight", dpi=150)
    logger.info("Scatter plot saved")

    # ---- Figure 3: Q-error boxplots by query size ----
    fig3, ax3 = plt.subplots(figsize=(10, 5))

    q_groups = [q_errors[num_triples == sz] for sz in unique_sizes]
    bp3 = ax3.boxplot(q_groups, tick_labels=labels, patch_artist=True,
                      showfliers=False, widths=0.6,
                      medianprops=dict(color="black", linewidth=1.5))
    for patch in bp3["boxes"]:
        patch.set_facecolor("#E8745A")
        patch.set_alpha(0.7)

    ax3.set_yscale("log")
    ax3.set_xlabel("Query Size (number of triples)")
    ax3.set_ylabel("Q-Error (log scale)")
    ax3.set_title("Q-Error Distribution by Query Size (C++ Inference)")
    ax3.grid(axis="y", alpha=0.3, which="both")
    fig3.tight_layout()
    for ext in ("pdf", "png"):
        fig3.savefig(output_dir / f"qerror_by_query_size.{ext}",
                     bbox_inches="tight", dpi=150)
    logger.info("Q-error figure saved")

    plt.close("all")

    print(f"\n  Figures saved to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
