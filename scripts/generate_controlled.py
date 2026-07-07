import itertools
import numpy as np
from collections import defaultdict
import os
import json

# ---------------------------------------------------------------- designs
FANO = [(0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)] # S(2,3,7)

def design_kg(blocks, prefix="fano", n_per_block=40, fanout=1):
    """Higher-order KG: each block's subjects carry exactly that block's predicates.
    Returns a list of (subject, predicate, object) triples."""
    triples = []
    for bi, block in enumerate(blocks):
        for n in range(n_per_block):
            s = f"<http://fice/{prefix}/s_{bi}_{n}>"
            for p in block:
                for k in range(fanout):
                    o = f"<http://fice/{prefix}/o_s_{bi}_{n}_p{p}_{k}>"
                    triples.append((s, f"<http://fice/{prefix}/p{p}>", o))
    return triples

def cluster_coupling(m, clusters, J_in=1.5, base_rate=0.3):
    """Ising parameters (h, J) from a coupling graph: within-cluster edges = J_in,
    isolated predicates = nodes in no cluster. base_rate sets the field h."""
    J = np.zeros((m, m))
    for cl in clusters:
        for a, b in itertools.combinations(cl, 2):
            J[a, b] = J[b, a] = J_in
    h = np.full(m, np.log(base_rate / (1 - base_rate)))
    return h, J

def ising_kg(h, J, prefix="ising", n_subjects=12000, fanout=1, seed=0):
    """Pure-pairwise KG: sample subjects from P(x) \u221d exp(h\u00b7x + \u00bd x\u1d40Jx) by exact
    enumeration of 2^m states (fine for m \u2264 ~18). Returns triples."""
    rng = np.random.default_rng(seed)
    m = len(h)
    X = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(float)
    logp = X @ h + 0.5 * np.einsum("si,ij,sj->s", X, J, X)
    p = np.exp(logp - logp.max())
    p /= p.sum()
    draws = X[rng.choice(2 ** m, n_subjects, p=p)]
    triples = []
    for n, x in enumerate(draws):
        s = f"<http://fice/{prefix}/s_{n}>"
        for pi in np.nonzero(x)[0]:
            for k in range(fanout):
                o = f"<http://fice/{prefix}/o_s_{n}_p{int(pi)}_{k}>"
                triples.append((s, f"<http://fice/{prefix}/p{int(pi)}>", o))
    return triples

# ---------------------------------------------------------------- queries
def _object_counts(triples):
    c = defaultdict(lambda: defaultdict(int))
    for s, p, _o in triples:
        c[s][p] += 1
    return c

def star_cardinality(counts, preds):
    """Exact cardinality of the star ?x p_i ?y_i for i in preds."""
    total = 0
    for _s, cs in counts.items():
        if all(cs.get(p, 0) > 0 for p in preds):
            prod = 1
            for p in preds:
                prod *= cs[p]
            total += prod
    return total

def star_queries(triples, m, prefix, orders=(1, 2, 3), max_per_order=None, seed=0):
    """All star queries up to the given orders, each as (preds, cardinality)."""
    rng = np.random.default_rng(seed)
    counts = _object_counts(triples)
    out = []
    for k in orders:
        combos = list(itertools.combinations(range(m), k))
        rng.shuffle(combos)
        if max_per_order:
            combos = combos[:max_per_order]
        for preds_idx in combos:
            preds = [f"<http://fice/{prefix}/p{i}>" for i in preds_idx]
            out.append((preds, star_cardinality(counts, preds)))
    return out

def format_queries_json(queries_data):
    """Format queries for FICE's JSON structure"""
    json_data = []
    for preds, card in queries_data:
        query_triples = []
        for i, p in enumerate(preds):
            query_triples.append(["?x", p, f"?y_{i}"])
        json_data.append({
            "triples": query_triples,
            "y": card
        })
    return json_data

def save_kg_and_queries(dataset_name, triples, queries):
    """Save the NT file and queries JSON in the expected data/ folder structure"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", dataset_name, "raw")
    queries_dir = os.path.join(base_dir, "data", f"{dataset_name}_queries", "raw")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)
    
    nt_path = os.path.join(data_dir, f"{dataset_name}.nt")
    with open(nt_path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s} {p} {o} .\n")
            
    json_path = os.path.join(queries_dir, "queries.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(format_queries_json(queries), f, indent=2)
        
    print(f"[{dataset_name}] Saved {len(triples)} triples to {nt_path}")
    print(f"[{dataset_name}] Saved {len(queries)} queries to {json_path}")

def main():
    print("Generating Fano KG...")
    fano_triples = design_kg(FANO, prefix="fano", n_per_block=40)
    fano_queries = star_queries(fano_triples, m=7, prefix="fano")
    save_kg_and_queries("fano", fano_triples, fano_queries)

    print("Generating Ising KG...")
    # 10 predicates, two clusters [0,1,2] and [3,4,5,6], others isolated
    h, J = cluster_coupling(10, clusters=[[0, 1, 2], [3, 4, 5, 6]])
    ising_triples = ising_kg(h, J, prefix="ising", n_subjects=12000)
    ising_queries = star_queries(ising_triples, m=10, prefix="ising", max_per_order=120)
    save_kg_and_queries("ising", ising_triples, ising_queries)

if __name__ == "__main__":
    main()
