"""Build G-Care statistical summaries (wj, cset, sumrdf) for a single dataset.

Can be used as a CLI script or imported and called as build_statistics().
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional


def _remove_files_except(directory_path: Path, exclude: List[str]) -> None:
    for child in directory_path.iterdir():
        if child.name in exclude:
            continue
        try:
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            print(f"Failed to delete {child}: {e}")


def build_statistics(dataset: str, gcare_home: Path, methods: Optional[List[str]] = None) -> None:
    """Build selected G-Care statistics for *dataset* inside *gcare_home*.

    Expects the graph file at:
        gcare_home/data/dataset/<dataset>/<dataset>.txt

    Writes compiled .graph files next to the dataset text file.
    Skips individual methods whose .graph files are already present.
    """
    graph_folder = gcare_home / "data" / "dataset" / dataset

    if not (graph_folder / f"{dataset}.txt").exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_folder / f'{dataset}.txt'}"
        )

    all_methods = methods if methods is not None else ["wj", "cset", "sumrdf"]
    p, seed = 0.03, 0
    needed = [
        m for m in all_methods
        if not (graph_folder / f"{dataset}.{m}.{p}.{seed}.graph").exists()
    ]

    if not needed:
        print(f"[{dataset}] Statistics already built for: {', '.join(all_methods)}, skipping.")
        return

    script_path = Path(__file__).resolve().parent / "custom-run-build.sh"
    env = {**os.environ, "GCARE_HOME": str(gcare_home), "GCARE_BUILD_METHODS": ",".join(needed)}

    start = time.time()
    print(f"[{dataset}] Building statistics for: {', '.join(needed)} …")
    subprocess.run(
        ["bash", str(script_path), dataset],
        check=True,
        env=env,
        cwd=str(gcare_home / "scripts"),
    )
    print(f"[{dataset}] Statistics built in {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G-Care statistics for a dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated build methods, e.g. sumrdf or wj,cset,sumrdf",
    )
    parser.add_argument(
        "--gcare-home",
        type=Path,
        default=Path(__file__).resolve().parent / "gcare",
        help="Path to compiled G-Care directory (default: ./gcare/)",
    )
    args = parser.parse_args()
    methods = [m.strip() for m in (args.methods or "").split(",") if m.strip()] or None
    build_statistics(args.dataset, args.gcare_home, methods)


if __name__ == "__main__":
    main()
