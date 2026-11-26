import yaml
import copy
from pathlib import Path


BASE = Path("configs/ws2627/base.yaml")
OUT = Path("configs/ws2627/runs")

EPOCHS = 300
SEEDS = [2]
CONDITIONINGS = {"all": "all", "false": "false"}
GNN_NEIGHBORHOOD = [-1, -1, -1, -1]

# dataset -> (query set, Overrides)
DATASETS = {
    "fano":  ("fano_queries",  {}),
    "ising": ("ising_queries", {"embedding_dim": 32}),
    "fano_unequal": ("fano_unequal_queries", {}),
    "fano_fanout": ("fano_fanout_queries", {}),
    "orthogonal": ("orthogonal_queries", {}),
    "two_clusters": ("two_clusters_queries", {}),
    "implication": ("implication_queries", {}),
    "path_dataset": ("path_queries", {}),
    "cycle_dataset": ("cycle_queries", {}),
}

def data_entries(ds, qs, batch_size=8):
    entry = {"dataset": ds, "queries": qs, "batch_size": batch_size, "num_queries": 1000}
    return [{**entry, "type": "train"}, {**entry, "type": "validation"}]

def main():
    n=0;
    base = yaml.safe_load(BASE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for ds, (qs, overrides) in DATASETS.items():
        for cond_name, cond in CONDITIONINGS.items():
            for seed in SEEDS:
                name = f"{ds}_{cond_name}_s{seed}"
                config = copy.deepcopy(base)
                config.update(overrides)
                config.update({
                    "data": data_entries(ds, qs),
                    "Conditioning": cond,
                    "seed": seed,
                    "epochs": EPOCHS,
                    "GNNNeighborhood": GNN_NEIGHBORHOOD,
                    "output_dir": f"output/ws2627/{name}/",
                    "wandb_run_name": name,
                })
                file_config = OUT.joinpath(f"{name}.yaml")
                with open(file_config, "w") as f:
                    yaml.safe_dump(config, f, sort_keys=False)
                n+=1

                print(f"config for dataset {ds} with conditioning {cond} and seed {seed}")
    print(f"{n} configs created")

if __name__ == "__main__":
    main()

