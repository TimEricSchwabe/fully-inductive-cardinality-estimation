import subprocess
import yaml
import copy

def run_training():
    with open("configs/train_custom.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    datasets = [
        ("fano", "fano_queries"),
        ("ising", "ising_queries")
    ]
    dims = [128, 4]

    for ds, qs in datasets:
        for d in dims:
            config = copy.deepcopy(base_config)
            config["epochs"] = 40
            config["data"] = [
                {
                    "dataset": ds,
                    "queries": qs,
                    "batch_size": 64,
                    "num_queries": 20000,
                    "type": "train-val"
                }
            ]
            config["embedding_dim"] = d
            config["min_cardinality"] = 1
            config["wandb_run_name"] = f"transductive_{ds}_d{d}"
            
            # Save temp config
            tmp_config = f"configs/tmp_{ds}_d{d}.yaml"
            with open(tmp_config, "w") as f:
                yaml.dump(config, f)
            
            print(f"========================================")
            print(f"Training {ds} with dim {d}")
            print(f"========================================")
            subprocess.run(["uv", "run", "--extra", "pyg-extras", "python3", "scripts/train.py", "--config", tmp_config], check=True)

if __name__ == "__main__":
    run_training()
