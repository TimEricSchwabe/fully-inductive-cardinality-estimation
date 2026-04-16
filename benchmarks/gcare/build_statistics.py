"""Build G-Care statistical summaries (wj, cset) for a single dataset.

Can be used as a CLI script or imported and called as build_statistics().
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Union, List


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


def build_statistics(dataset: str, gcare_home: Path) -> None:
    """Build wj/cset statistics for *dataset* inside *gcare_home*.

    Expects the graph file at:
        gcare_home/data/dataset/<dataset>/<dataset>.txt

    Writes compiled .graph files next to the dataset text file.
    Skips if .graph files are already present.
    """
    graph_folder = gcare_home / "data" / "dataset" / dataset
    existing_graphs = list(graph_folder.glob("*.graph"))
    if existing_graphs:
        print(f"[{dataset}] Statistics already built ({len(existing_graphs)} .graph files), skipping.")
        return

    if not (graph_folder / f"{dataset}.txt").exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_folder / f'{dataset}.txt'}"
        )

    # Remove stale build artefacts except the source txt and ID mappings.
    if graph_folder.is_dir():
        keep = [
            f"{dataset}.txt",
            "id_to_id_mapping.json",
            "id_to_id_mapping_predicate.json",
        ]
        _remove_files_except(graph_folder, keep)

    script_path = Path(__file__).resolve().parent / "custom-run-build.sh"
    env = {**os.environ, "GCARE_HOME": str(gcare_home)}

    start = time.time()
    print(f"[{dataset}] Building statistics …")
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
        "--gcare-home",
        type=Path,
        default=Path(__file__).resolve().parent / "gcare",
        help="Path to compiled G-Care directory (default: ./gcare/)",
    )
    args = parser.parse_args()
    build_statistics(args.dataset, args.gcare_home)


if __name__ == "__main__":
    main()
