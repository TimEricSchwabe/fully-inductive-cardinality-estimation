import subprocess
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
import json
from typing import Union, List

# Environment variables for Docker container paths
GCARE_HOME = os.environ.get('GCARE_HOME', '/usr/src/app/gcare')
DATA_ROOT = os.environ.get('DATA_ROOT', '/usr/src/app/gcare/data')
RESULTS_ROOT = os.environ.get('RESULTS_ROOT', '/usr/src/app/results')

def remove_files_in_directory(directory_path, exclude_file: Union[str, List[str]]='yago.txt'):
    if isinstance(exclude_file, str):
        exclude_file = [exclude_file]

    for filename in os.listdir(directory_path):
        if filename in exclude_file:
            continue  # Skip the file to be excluded
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def main():
    dataset = os.environ.get('DATASET_NAME', 'unknown_graph')
    hard_reset = True

    # check if build statistics is neccessary
    data_root = Path(f"{DATA_ROOT}")
    graph_folder = data_root / "dataset" / dataset
    if graph_folder.exists() and graph_folder.is_dir() and not hard_reset:
        graph_files = list(graph_folder.glob("*.graph"))
        if len(graph_files) > 0:
            print(f"Graph files already exist for dataset {dataset}. Skipping build statistics.")
            exit(0)

    # empty build fodler except dataset.txt
    if graph_folder.is_dir():
        exclude_files = ["id_to_id_mapping.json", "id_to_id_mapping_predicate.json", f"{dataset}.txt"]
        remove_files_in_directory(graph_folder, exclude_file=exclude_files)

    savetag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Remove all files in ../data/dataset/data
    # remove_files_in_directory(f"/home/tim/gcare/data/dataset/{dataset}", exclude_file=f'{dataset}.txt')

    starting_time = time.time()

    os.chdir(f"{GCARE_HOME}/scripts")
    # Execute the building for the specific method
    gcare_home_path = Path(GCARE_HOME)
    # script_path = gcare_home_path / "scripts" / "run-build.sh"
    script_path = Path(__file__).resolve().parent / "custom-run-build.sh"
    subprocess.run(["bash", str(script_path), dataset], check=True)
    end_time = time.time() - starting_time



if __name__ == '__main__':
    main()
