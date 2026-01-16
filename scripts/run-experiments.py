import subprocess
from typing import Dict, Optional
import yaml
from pathlib import Path
import copy



EXPERIMENTS = [
    {
        'name': 'swdf_inductive',
        'validation': ['SWDF_Star_Queries', 'swdf_path_queries'],
    },
    {
        'name': 'lubm_inductive',
        'validation': ['LUBMQueries', 'lubm_path_queries'],
    },
    {
        'name': 'dbpedia100k',
        'validation': ['dbpedia100k_star_queries', 'dbpedia100k_path_queries'],
    },
    {
        'name': 'wikidata',
        'validation': ['WIKIDATA_Star_Queries', 'wikidata_path_queries'],
    },
    {
        'name': 'fb237_inductive',
        'validation': ['fb237_v4_ind_star_queries', 'fb237_v4_ind_path_queries'],
    },
    {
        'name': 'wn18rr_inductive',
        'validation': ['wn18rr_v4_ind_star_queries', 'wn18rr_v4_ind_path_queries'],
    },
    {
        'name': 'transductive',
        'validation': [],
    },
]

BASE_CONFIG_PATH = Path('./configs/train_full.yaml')


def prepare_config(experiment: Dict, config: Dict) -> Dict:
    exp_config = copy.deepcopy(config)
    exp_config['wandb_run_name'] = f"EXPERIMENT_{experiment['name']}"

    for d in exp_config['data']:
        query_dataset_name = d['queries']
        if query_dataset_name in experiment['validation']:
            d['type'] = 'validation'
        else:
            d['type'] = 'train-val'


    return exp_config



def save_temp_config(config: Dict, experiment: Dict, out: Path) -> Path:
    name = experiment['name']
    counter = 0
    config_path = out / f'{name}.yaml'
    while config_path.exists():
        counter += 1
        config_path = out / f'{name}_{counter}.yaml'
        
    with config_path.open('w') as f:
        yaml.dump(config, f)

    return config_path

def sbatch_run(config_path: Path, experiment_name: Optional[str] = None, w: Optional[str] = None):
    command = ['sbatch']

    # Handle optional arguments
    if w is not None:
        command.extend(['-w', w])
    
    if experiment_name is not None:
        command.extend(['-J', f'fice-{experiment_name}'])
    
    # Add the script path and the config argument
    # We pass config_path as an argument to the .sh script
    command.extend(['./scripts/experiment.sbatch', str(config_path.resolve())])

    try:
        # Run the command and capture output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Job submitted successfully:\n{result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job:\n{e.stderr}")
        return None

def main():
    base_config = yaml.safe_load(BASE_CONFIG_PATH.open('r'))
    out_dir = Path('./configs/')
    out_dir.mkdir(exist_ok=True, parents=True)
    
    running_jobs = {}
    for e in EXPERIMENTS:
        config = prepare_config(experiment=e, config=base_config)
        config_path = save_temp_config(config=config, experiment=e, out=out_dir)
        result = sbatch_run(config_path=config_path, experiment_name=e['name'])
        if result is not None:
            job_id = result.strip().split(' ')[-1]
            running_jobs[job_id] = e['name']
    
    print("\nSubmitted jobs:")
    for job_id, name in running_jobs.items():
        print(f"Job ID: {job_id}, Experiment: {name}")

    failed_jobs = set([e['name'] for e in EXPERIMENTS]) - set(running_jobs.values())
    if failed_jobs:
        print("\nFailed to submit jobs for experiments:")
        for name in failed_jobs:
            print(f"Experiment: {name}")
    else:
        print("\nAll jobs submitted successfully.")

        


if __name__ == "__main__":
    main()