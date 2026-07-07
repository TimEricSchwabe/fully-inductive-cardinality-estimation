import torch
import numpy as np
import sys
import os
import itertools
from pathlib import Path

# from symtorch import SymbolicModel  # backend = PySR

sys.path.append('.')
from utils.config import load_config_yaml
from utils.data_helper import make_factor_graph
from utils.models import TrainingContext
from scripts.train import initialize_model_components
from fice.datasets import FanoDataset, IsingDataset, FanoQueries, IsingQueries

def star_features(preds, R, occ):
    r = [R[i] for i in preds]
    f = {
        "indep": sum(np.log1p(occ[i]) for i in preds), # order 1
        "sumdot": sum(float(r[a] @ r[b]) for a in range(len(r)) for b in range(a+1, len(r))), # order 2
        "tri": sum(float((r[a]*r[b]*r[c]).sum()) for a in range(len(r)) for b in range(a+1,len(r)) for c in range(b+1,len(r))) # order 3
    }
    return f

def parse_query_data(query_dataset):
    queries_list = []
    for i in range(len(query_dataset)):
        query_graph = query_dataset[i]
        preds = []
        for e in query_graph.edge_type:
            preds.append(int(e))
        queries_list.append((tuple(preds), query_graph.y.item()))
    return queries_list

def main():
    device = torch.device('cpu')
    
    # Check if symtorch is available
    try:
        from symtorch import SymbolicModel
        import pysr
        print("Successfully imported symtorch and PySR")
    except ImportError:
        print("Warning: symtorch or PySR not found. Cannot run symbolic regression.")
        print("Please install them: uv pip install pysr torch-symbolic")
        # return

    configs_to_test = [
        ("fano", 128, 7),
        ("ising", 128, 10)
    ]
    
    for ds_name, dim, m in configs_to_test:
        print(f"\n==============================================")
        print(f" Symbolic Regression on {ds_name.upper()} (dim={dim})")
        print(f"==============================================")
        
        run_name = f"transductive_{ds_name}_d{dim}"
        runs = []
        for p in Path("output").iterdir():
            if p.is_dir() and (p / "config.yaml").exists():
                cfg = load_config_yaml(p / "config.yaml")
                if getattr(cfg, "wandb_run_name", None) == run_name:
                    runs.append(p)
                    
        if not runs:
            print(f"No completed run found for {run_name}")
            continue
        
        latest_run = sorted(runs, key=os.path.getmtime)[-1]
        model_path = latest_run / "checkpoints" / "final_model.pt"
        
        if ds_name == "fano":
            ds = FanoDataset()
            query_ds = FanoQueries()
        else:
            ds = IsingDataset()
            query_ds = IsingQueries()
            
        graph = make_factor_graph(ds).to(device)
        n_ent = graph.original_num_nodes
        n_rel = graph.original_num_relations
        
        config_path = f"configs/tmp_{ds_name}_d{dim}.yaml"
        config = load_config_yaml(config_path)
        context = TrainingContext(config=config, device=device, run_dir=Path("dummy"), log_dir=Path("dummy"), checkpoint_dir=Path("dummy"))
        components = initialize_model_components(context, total_training_steps=1)
        model = components.model
        
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            out_emb = model.embedding_model(graph).squeeze(0)
            rel_embs = out_emb[n_ent:n_ent+n_rel].cpu().numpy()
            
        # Get raw triples to count occurrences
        raw_nt_path = Path(f"data/{ds_name}/raw/{ds_name}.nt")
        occ = np.zeros(m)
        with open(raw_nt_path, "r") as f:
            for line in f.readlines():
                s, p, o = line.strip().split()[:3]
                p_idx = int(p.split('p')[-1].replace('>', ''))
                occ[p_idx] += 1
                
        queries = parse_query_data(query_ds)
        three_star_queries = [(p, c) for p, c in queries if len(p) == 3]
        
        X_features = []
        y_residual = []
        
        print("Extracting features for 3-star queries...")
        for preds, true_card in three_star_queries:
            f = star_features(preds, rel_embs, occ)
            
            # Recipe A target: log(1 + card) - indep_estimate
            indep_estimate = f["indep"]
            target = np.log1p(true_card) - indep_estimate
            
            # Features: [indep, sumdot, tri]
            X_features.append([f["indep"], f["sumdot"], f["tri"]])
            y_residual.append(target)
            
        X = np.array(X_features)
        y = np.array(y_residual)
        
        print(f"Dataset X shape: {X.shape}, y shape: {y.shape}")
        
        # Here we would run PySR
        print("\nReady for PySR fitting!")
        print("Target: y = log1p(card) - indep")
        print("Inputs: X = [indep, sumdot, tri]")
        
        try:
            from pysr import PySRRegressor
            model_sr = PySRRegressor(
                niterations=40,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "square", "sqrt"], # "relu(x) = max(x, 0f0)", "log1p(x) = log(1f0 + x)"
                extra_sympy_mappings={"relu": lambda x: max(x, 0), "log1p": lambda x: np.log1p(x)},
                model_selection="best",
            )
            print("Fitting PySR model...")
            model_sr.fit(X, y)
            print("\nBest symbolic model found:")
            print(model_sr.sympy())
        except ImportError:
            pass

if __name__ == "__main__":
    main()
