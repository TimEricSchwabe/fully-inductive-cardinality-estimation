import torch
import numpy as np
import sys
import os
import itertools
from collections import defaultdict
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score

sys.path.append('.')
from utils.config import load_config_yaml
from utils.data_helper import make_factor_graph
from utils.models import TrainingContext
from scripts.train import initialize_model_components
from fice.datasets import FanoDataset, IsingDataset, FanoQueries, IsingQueries
from scripts.generate_controlled import FANO

# ----------------------------------------------- data side: max-ent hierarchy
def occurrence_matrix(triples, m):
    """Binary occurrence matrix X (n_subjects x m), X[r,i]=1 iff subject r has p_i."""
    has = defaultdict(set)
    for s, p, _o in triples:
        has[s].add(int(p.split('p')[-1].replace('>', '')))
    subs = sorted(has)
    X = np.zeros((len(subs), m))
    for r, s in enumerate(subs):
        for i in has[s]:
            X[r, i] = 1
    return X

def maxent_hierarchy(X, max_order=3, ridge=1e-9):
    """Fit P_1..P_max_order by exact enumeration (m <= ~18). Returns the residual
    beyond pairwise D_KL(data||P_2) and the connected information per order."""
    n, m = X.shape
    states = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(float)
    
    terms = lambda K: [c for k in range(1, K + 1) for c in itertools.combinations(range(m), k)]
    feats = lambda rows, tm: np.stack([rows[:, list(t)].prod(1) for t in tm], 1)
    
    ent = lambda p: float(-(p[p > 1e-300] * np.log(p[p > 1e-300])).sum())
    q = np.bincount((X * (1 << np.arange(m))).sum(1).astype(int), minlength=2 ** m) / n
    S_data, S = ent(q), []
    
    for K in range(1, max_order + 1):
        tm = terms(K); F = feats(states, tm); dmean = feats(X, tm).mean(0)
        theta = np.zeros(len(tm))
        for _ in range(400):  # damped Newton on concave LL
            z = F @ theta; z -= z.max(); p = np.exp(z); p /= p.sum()
            g = dmean - F.T @ p - ridge * theta
            if np.abs(g).max() < 1e-11:
                break
            Fc = F - F.T @ p
            H = (Fc * p[:, None]).T @ Fc + ridge * np.eye(len(tm))
            step = np.linalg.solve(H + 1e-12 * np.eye(len(tm)), g)
            ll0 = theta @ dmean - np.logaddexp.reduce(F @ theta); a = 1.0
            for _bt in range(30):  # backtracking line search
                if (theta + a * step) @ dmean - np.logaddexp.reduce(F @ (theta + a * step)) >= ll0 - 1e-12:
                    break
                a *= 0.5
            theta = theta + a * step
        z = F @ theta; z -= z.max(); p = np.exp(z); p /= p.sum()
        S.append(ent(p))
        
    return {
        "S_data": S_data,
        "KL_pairwise": S[1] - S_data,  # D_KL(data||P_2)
        "I_connected": {k + 2: S[k] - S[k + 1] for k in range(len(S) - 1)}  # order k
    }

# ------------------------------------------ representation side: surrogate hierarchy
def _reduce(v):
    if len(v) == 0:
        return [0., 0., 0., 0., 0.]
    a = np.asarray(v, float)
    return [a.sum(), a.mean(), a.max(), a.min(), float(len(a))]

def query_features(preds, R, order):
    """Permutation-invariant, fixed-length features up to the given interaction order."""
    idx = list(preds)
    f = [float(len(idx))] + _reduce([np.linalg.norm(R[i]) for i in idx])  # order 1
    if order >= 2:
        f += _reduce([R[i] @ R[j] for i, j in itertools.combinations(idx, 2)])  # order 2: Gram
    if order >= 3:
        f += _reduce([(R[i] * R[j] * R[k]).sum() for i, j, k in itertools.combinations(idx, 3)])  # order 3: t_ijk
    return np.array(f)

def surrogate_hierarchy(queries, R, target, orders=(1, 2, 3)):
    """CV R^2 of predicting `target` (true or decoder log-card) from order-k features.
    The order-2 -> order-3 jump is the higher-order content."""
    y = np.asarray(target, float)
    cv = KFold(min(5, len(queries)), shuffle=True, random_state=0)
    rf = lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    
    return {k: round(float(cross_val_score(
            rf(), np.stack([query_features(p, R, k) for p, _c in queries]),
            y, cv=cv, scoring="r2").mean()), 3) for k in orders}

def matched_triple_auc(queries, blocks, score_fn):
    """AUC of `score_fn(preds)` separating line vs non-line 3-stars (matched on pairwise stats).
    score_fn = the decoder's prediction in practice."""
    lines = set(map(frozenset, blocks))
    three = [(p, c) for p, c in queries if len(p) == 3]
    y = np.array([1 if frozenset(p) in lines else 0 for p, _c in three])
    s = np.array([score_fn(p) for p, _c in three])
    return round(float(roc_auc_score(y, s)), 3) if len(set(y)) > 1 else float("nan")

# ------------------------------------ P_2 Baseline
def fit_p2(X, ridge=1e-9):
    """Pairwise max-entropy model P_2 over the 2^m states (m <= ~18)."""
    n, m = X.shape
    states = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(float)
    terms = [c for k in (1, 2) for c in itertools.combinations(range(m), k)]
    feats = lambda rows: np.stack([rows[:, list(t)].prod(1) for t in terms], 1)
    F = feats(states); dmean = feats(X).mean(0); theta = np.zeros(len(terms))
    
    for _ in range(400):
        z = F @ theta; z -= z.max(); p = np.exp(z); p /= p.sum()
        g = dmean - F.T @ p - ridge * theta
        if np.abs(g).max() < 1e-11: break
        Fc = F - F.T @ p; H = (Fc * p[:, None]).T @ Fc + ridge * np.eye(len(terms))
        step = np.linalg.solve(H + 1e-12 * np.eye(len(terms)), g)
        ll0 = theta @ dmean - np.logaddexp.reduce(F @ theta); a = 1.0
        for _bt in range(30):
            if (theta + a * step) @ dmean - np.logaddexp.reduce(F @ (theta + a * step)) >= ll0 - 1e-12:
                break
            a *= 0.5
        theta = theta + a * step
    z = F @ theta; z -= z.max(); p = np.exp(z); p /= p.sum()
    return states, p

def p2_star_cardinality(states, p, preds, n_subjects):
    """Best pairwise (P_2) prediction of a star's cardinality."""
    mask = np.all(states[:, list(preds)] == 1, axis=1)
    return p[mask].sum() * n_subjects

def parse_query_data(query_dataset):
    """Parses NTQueries to extract (preds_tuple, cardinality) format"""
    queries_list = []
    # This might need adjustment depending on how queries are stored in `query_dataset`
    # Here we assume `query_dataset.data.y` has true cardinality and we can extract predicates
    for i in range(len(query_dataset)):
        query_graph = query_dataset[i]
        preds = []
        for e in query_graph.edge_type:
            preds.append(int(e))
        queries_list.append((tuple(preds), query_graph.y.item()))
    return queries_list

def main():
    device = torch.device('cpu')
    
    # We will test dimension 128 as the primary target for probing
    configs_to_test = [
        ("fano", 128, 7),
        ("ising", 128, 10)
    ]
    
    for ds_name, dim, m in configs_to_test:
        print(f"\n==============================================")
        print(f" Probing FICE on {ds_name.upper()} (dim={dim})")
        print(f"==============================================")
        
        # Determine the latest run folder
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
        
        # We take the most recent
        latest_run = sorted(runs, key=os.path.getmtime)[-1]
        model_path = latest_run / "checkpoints" / "final_model.pt"
        
        if not model_path.exists():
            print(f"Model checkpoint not found at {model_path}")
            continue
        
        # Load dataset
        if ds_name == "fano":
            ds = FanoDataset()
            query_ds = FanoQueries()
        else:
            ds = IsingDataset()
            query_ds = IsingQueries()
            
        graph = make_factor_graph(ds).to(device)
        n_ent = graph.original_num_nodes
        n_rel = graph.original_num_relations
        
        # Load model
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
            
        print(f"Extracted {len(rel_embs)} relation embeddings.")
        
        # Get raw triples for data side metrics
        raw_nt_path = Path(f"data/{ds_name}/raw/{ds_name}.nt")
        with open(raw_nt_path, "r") as f:
            triples = [line.strip().split()[:3] for line in f.readlines()]
            
        # 1. Data side: maxent hierarchy
        X = occurrence_matrix(triples, m)
        print(f"\n--- Data-Side Ceiling (max-entropy hierarchy) ---")
        maxent_res = maxent_hierarchy(X)
        print(f"KL(data || P_2): {round(maxent_res['KL_pairwise'], 4)} nats")
        
        # 2. Extract queries
        queries = parse_query_data(query_ds)
        three_star_queries = [(p, c) for p, c in queries if len(p) == 3]
        
        # Approach 1b: order-resolved decodability on the encoder's embeddings
        print(f"\n--- Approach 1b: Encoder's Relation Embeddings (R = H_R) ---")
        if ds_name == "fano":
            target_1b = [1 if frozenset(p) in set(map(frozenset, FANO)) else 0 for p, c in three_star_queries]
            res_1b = surrogate_hierarchy(three_star_queries, rel_embs, target_1b)
            print(f"Fano line-membership prediction R^2 (order 1/2/3): {res_1b}")
        else:
            target_1b = [np.log1p(c) for p, c in queries]
            res_1b = surrogate_hierarchy(queries, rel_embs, target_1b)
            print(f"Ising log-cardinality prediction R^2 (order 1/2/3): {res_1b}")
        
        # Approach 2: Decoder computation probe
        def score_fn_decoder(preds):
            """Score function that runs FICE's query graph prediction"""
            # We need to construct a mini-batch with a single query containing these predicates
            # The simplest way is to find the query in `query_ds` that matches these predicates
            for i in range(len(query_ds)):
                q_graph = query_ds[i]
                q_preds = [int(e) for e in q_graph.edge_type]
                if set(q_preds) == set(preds):
                    # Found it
                    from torch_geometric.data import Batch
                    q_batch = Batch.from_data_list([q_graph.to(device)])
                    graph_size = torch.tensor([q_batch.num_nodes], dtype=torch.long, device=device)
                    with torch.no_grad():
                        pred_log_card = model(data=graph, relation_graph=graph, batch=q_batch, graph_size=graph_size).item()
                    return pred_log_card
            return 0.0
            
        print(f"\n--- Approach 2a: Pairwise-surrogate residual on decoder's output ---")
        target_2a = [score_fn_decoder(p) for p, c in queries]
        res_2a = surrogate_hierarchy(queries, rel_embs, target_2a)
        print(f"Decoder output prediction R^2 (order 1/2/3): {res_2a}")
        
        print(f"\n--- Approach 2b: Matched-triple AUC ---")
        if ds_name == "fano":
            auc = matched_triple_auc(queries, FANO, score_fn_decoder)
            print(f"Fano matched-triple AUC: {auc}")
        else:
            print("Skipped for Ising (no matched triples blocks).")
            
        # Approach 3: Behavioral Test (Comparison with P_2 baseline)
        print(f"\n--- Approach 3a: P_2 Baseline vs True vs Decoder ---")
        states, p2 = fit_p2(X)
        
        # Check a sample of 3-star queries
        print(f"Sample 3-star query comparisons:")
        sample_queries = three_star_queries[:5]
        for preds, true_card in sample_queries:
            pred_p2 = p2_star_cardinality(states, p2, preds, X.shape[0])
            pred_fice = score_fn_decoder(preds)
            print(f"  Preds {preds} | True: {true_card:6.1f} | P2: {pred_p2:6.1f} | FICE: {np.expm1(pred_fice):6.1f}")
            

if __name__ == "__main__":
    main()
