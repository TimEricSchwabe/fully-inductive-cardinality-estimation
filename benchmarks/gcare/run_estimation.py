import os
import subprocess
import time
from datetime import datetime
import numpy as np
import json
from transform_query import query_to_gcare
import random
from pathlib import Path
import glob
import shutil
import logging

GCARE_HOME = os.environ.get('GCARE_HOME', '/usr/src/app/gcare')
DATA_ROOT = os.environ.get('DATA_ROOT', '/usr/src/app/data')
RESULTS_ROOT = os.environ.get('RESULTS_ROOT', '/usr/src/app/results')

# configure logging so messages show in container logs
logging.basicConfig(level=logging.INFO)


def run_gcare(dataset, eval_folder, query_root, dataset_root, inductive):
    methods = ['wj', 'cset', 'sumrdf']
    #methods = ['impr']
    # Loading Queries and IDX transform mappings

    query_path = Path(glob.glob(str(query_root / '*.json'))[0])
    id2id_mapping_path = dataset_root / 'id_to_id_mapping.json'
    id2id_mapping_predicate_path = dataset_root / 'id_to_id_mapping_predicate.json'


    print('Loading Queries from:')
    print(str(query_path))
    with query_path.open() as f:
        raw_data = json.load(f)

    data = random.sample(raw_data, 100)
    # Same split as other methods
    if not inductive == 'full':
        random.Random(4).shuffle(data)
        data = data[int(0.8 * len(data)):]
    

    logging.info(f"Total number of queries loaded: {len(data)}")
    with id2id_mapping_path.open("r") as f:
        id_to_id_mapping = json.load(f)

    with id2id_mapping_predicate_path.open("r") as f:
        id_to_id_mapping_predicate = json.load(f)

    

    for method in methods:
        print(f"Starting {method}")
        preds = []
        gts = []
        sizes = []
        result_data = []
        pred_times = []
        result_dir = eval_folder / method
        result_dir.mkdir(parents=True, exist_ok=True)


        # Predicting top n queries of the testset
        queries = data[:]
        print(len(queries))
        for i, query in enumerate(queries):

            query_to_gcare(query["triples"], 0, id_to_id_mapping=id_to_id_mapping,
                           id_to_id_mapping_predicate=id_to_id_mapping_predicate, dataset=dataset, card=query["y"])
            try:
                y_pred, pred_time = predict(method=method, dataset=dataset)
            except subprocess.CalledProcessError as e:
                logging.error(str(e))
                continue

            preds.append(y_pred)
            gts.append(query["y"])
            pred_times.append(pred_time)
            sizes.append(len(query["triples"]))
            query["y_pred"] = y_pred
            query["exec_time_total"] = pred_time
            result_data.append(query)

            # log completion so container logs capture progress
            logging.info(f"{method} idx={i+1}/{len(queries)} size={len(query['triples'])} y={query['y']} y_pred={y_pred} time={pred_time}")
        np.save(result_dir / 'preds.npy', preds)
        np.save(result_dir / 'gts.npy', gts)
        np.save(result_dir / 'sizes.npy', sizes)
        np.save(result_dir / 'pred_times.npy', pred_times)

        with (result_dir / 'results.json').open('w') as file:
            json.dump(result_data, file, indent=4)

def predict(method: str, dataset: str):
    seed = 0
    now = 'current'
    result_dir = f'{GCARE_HOME}/data/result/accuracy/{now}'
    p = 0.03
    repeat = 1 if method in ['cset', 'bsk', 'sumrdf'] else 30
    if method == 'bsk':
        os.environ['GCARE_BSK_BUDGET'] = '4096'
    command = f'./run-exp.sh {method} {dataset} {p} {seed} {repeat} {result_dir}'
    start_time = time.time()
    proc = subprocess.run(
        # f'bash -lc "{command} | grep -A 2 \"Average Time\""',
        f'bash -lc "{command}"',
        check=True,
        shell=True,
        stdout=subprocess.PIPE,
        cwd=Path(f'{GCARE_HOME}/scripts')
    )
    end_time = time.time() - start_time
    result = proc.stdout.decode()

    lines = result.strip().split(' ')
    predicted_cardinalities = np.asarray(lines[-repeat:], dtype=float)
    prediction_time = end_time
    pred_cardinality = np.mean(predicted_cardinalities) 

    # predicted_cardinalities = np.asarray(lines[2].split(' ')[1:], dtype=float)
    # prediction_time = float(lines[0].split(" ")[4])
    # pred_cardinality = np.mean(predicted_cardinalities)

    # Clear the environment variable if it was set
    if 'GCARE_BSK_BUDGET' in os.environ:
        del os.environ['GCARE_BSK_BUDGET']

    return pred_cardinality, prediction_time


if __name__ == "__main__":
    dataset_name = os.environ.get('DATASET_NAME', 'unknown_graph')
    
    graph_path = Path('./gcare/data/dataset') / dataset_name
    queries_path = Path('./gcare/data/queryset') / dataset_name 
    print('Graph Path:', str(graph_path))
    print('Query Path:', str(queries_path))
    eval_folder = Path('/usr/src/app/results/evaluation/')

    if eval_folder.exists():
        for child in eval_folder.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except Exception as e:
                print(f"Failed to remove {child}: {e}")

    # command = f'./run-build.sh'
    # proc = subprocess.run(
    #     f'bash -lc "{command}"',
    #     check=True,
    #     shell=True,
    #     cwd=Path(f'{GCARE_HOME}/scripts')
    # )

    run_gcare(dataset=dataset_name,
              eval_folder=eval_folder,
              query_root=queries_path,
              dataset_root=graph_path,
              inductive='full')