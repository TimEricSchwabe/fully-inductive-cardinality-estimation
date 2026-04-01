"""Benchmark C++ inference timing and plot runtime by query size.

Runs all queries through the C++ binary in batch mode, collecting per-query
timing for graph construction, model inference, and total. Produces a
matplotlib figure with boxplots grouped by query size (number of triples).

Usage:
    cd scripts
    python ../inference/benchmark_cpp.py \
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
        build_us_list.append(float(parts[3]))
        infer_us_list.append(float(parts[4]))
        total_us_list.append(float(parts[5]))

    num_triples = np.array(num_triples_list)
    build_us = np.array(build_us_list)
    infer_us = np.array(infer_us_list)
    total_us = np.array(total_us_list)

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

    # Per query-size stats
    unique_sizes = sorted(np.unique(num_triples))
    print(f"\n  Query sizes present: {unique_sizes}")
    print(f"\n  {'Size':>5}  {'Count':>6}  {'Build(med)':>10}  "
          f"{'Infer(med)':>10}  {'Total(med)':>10}")
    for sz in unique_sizes:
        mask = num_triples == sz
        print(f"  {sz:>5}  {mask.sum():>6}  "
              f"{np.median(build_us[mask]):>10.0f}  "
              f"{np.median(infer_us[mask]):>10.0f}  "
              f"{np.median(total_us[mask]):>10.0f}")

    # ---- Write raw TSV ----
    tsv_path = output_dir / "benchmark_results.tsv"
    with open(tsv_path, "w") as f:
        f.write("query_idx\tnum_triples\tgraph_build_us\tinference_us\ttotal_us\n")
        for i in range(len(num_triples)):
            f.write(f"{i}\t{num_triples[i]}\t{build_us[i]:.0f}\t"
                    f"{infer_us[i]:.0f}\t{total_us[i]:.0f}\n")
    logger.info(f"Raw results written to {tsv_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    plot_data = [
        ("Graph Construction", build_us),
        ("Model Inference", infer_us),
        ("Total", total_us),
    ]

    for ax, (title, arr) in zip(axes, plot_data):
        # Group data by query size
        groups = []
        labels = []
        for sz in unique_sizes:
            mask = num_triples == sz
            groups.append(arr[mask])
            labels.append(str(sz))

        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))

        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)

        ax.set_xlabel("Query Size (number of triples)")
        ax.set_ylabel("Time (microseconds)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("C++ FICE Inference Runtime by Query Size", fontsize=14, y=1.02)
    fig.tight_layout()

    fig_path = output_dir / "runtime_by_query_size.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    fig_path_png = output_dir / "runtime_by_query_size.png"
    fig.savefig(fig_path_png, bbox_inches="tight", dpi=150)
    logger.info(f"Figure saved to {fig_path} and {fig_path_png}")

    print(f"\n  Figure: {fig_path_png}")
    print("=" * 70)


if __name__ == "__main__":
    main()
