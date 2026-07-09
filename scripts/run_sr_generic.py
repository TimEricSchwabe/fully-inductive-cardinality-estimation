import argparse
import json
import re
import os
import glob
import torch
import numpy as np
import sympy
from collections import defaultdict
from symtorch import SymbolicModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generic Symbolic Regression using SymTorch.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., fano, ising, orthogonal, two_clusters)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pt)")
    return parser.parse_args()

def star_features(preds, R, occ):
    """Converts raw predicate embeddings into 3 mathematical features."""
    r = [R[i] for i in preds]
    f_indep = sum(np.log1p(occ[i]) for i in preds)
    f_sumdot = sum(float(r[a] @ r[b]) for a in range(len(r)) for b in range(a + 1, len(r)))
    f_tri = sum(float((r[a] * r[b] * r[c]).sum())
                for a in range(len(r))
                for b in range(a + 1, len(r))
                for c in range(b + 1, len(r)))
    return [f_indep, f_sumdot, f_tri]

def get_occurrences_from_nt(nt_path):
    occ = defaultdict(int)
    max_p = 0
    with open(nt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 3:
                p_uri = parts[1].strip('<>')
                # Extract integer from predicate (e.g., .../p0 -> 0)
                match = re.search(r'p(\d+)$', p_uri)
                if match:
                    p_idx = int(match.group(1))
                    occ[p_idx] += 1
                    if p_idx > max_p:
                        max_p = p_idx
    
    # Convert to array
    occ_arr = np.zeros(max_p + 1)
    for p_idx, count in occ.items():
        occ_arr[p_idx] = count
    return occ_arr

def main():
    args = parse_args()
    print(f"Running generic SR for dataset: {args.dataset}")
    
    # 1. Load the model and get embeddings
    print("Loading model and extracting embeddings...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    R = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()
    
    # 2. Get true occurrences from the raw dataset
    raw_nt_path = f"data/{args.dataset}_dataset/raw/raw_dataset.nt"
    if not os.path.exists(raw_nt_path):
        raise FileNotFoundError(f"Cannot find raw dataset at {raw_nt_path}")
    actual_occ = get_occurrences_from_nt(raw_nt_path)
    
    # 3. Load queries to get y and preds
    queries_path = f"data/{args.dataset}_queries/raw/{args.dataset}_queries.json"
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Cannot find queries at {queries_path}")
        
    with open(queries_path, 'r') as f:
        queries = json.load(f)
        
    X_list = []
    y_list = []
    
    for q in queries:
        # We only care about star queries (multiple predicates on same subject)
        if len(q['triples']) < 2:
            continue
            
        # Extract predicates
        preds = []
        for t in q['triples']:
            p_uri = t[1]
            match = re.search(r'p(\d+)$', p_uri)
            if match:
                preds.append(int(match.group(1)))
                
        if not preds:
            continue
            
        card = q['y']
        
        # Calculate features
        features = star_features(preds, R, actual_occ)
        X_list.append(features)
        
        indep_val = features[0]
        # Target = log1p(card) - indep_val
        y_val = np.log1p(card) - indep_val
        y_list.append(y_val)
        
    if not X_list:
        raise ValueError("No valid star queries found to generate features.")
        
    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    
    # 4. SYMBOLIC REGRESSION VIA SYMTORCH
    print("Setting up the SymTorch framework...")
    class RecipeABlock(torch.nn.Module):
        def __init__(self, targets):
            super().__init__()
            self.targets = targets
        def forward(self, x):
            return self.targets

    base_block = RecipeABlock(y)
    sr_model = SymbolicModel(base_block)
    _ = sr_model(X)
    
    print("Starting evolutionary symbolic regression...")
    binary_ops = ["+", "-", "*", "/"]
    unary_ops = [
        "exp", "log", "square", "sqrt",
        "relu(x) = max(x, 0f0)",
        "log1p(x) = x > -1f0 ? log(1f0 + x) : Float32(NaN)"
    ]
    
    custom_sympy_mappings = {
        "relu": lambda x: sympy.Max(x, 0),
        "log1p": lambda x: sympy.log(1 + x)
    }
    
    os.makedirs(f"SR_output/{args.dataset}_results", exist_ok=True)
    
    sr_model.distill(
        X,
        sr_params={
            "binary_operators": binary_ops,
            "unary_operators": unary_ops,
            "extra_sympy_mappings": custom_sympy_mappings
        },
        save_path=f"SR_output/{args.dataset}_results"
    )
    print("Search complete!")

if __name__ == "__main__":
    main()
