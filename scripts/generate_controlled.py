import argparse
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
    Returns a list of (subject, predicate, object) triples. n_per_block can be int (same size for all blocks). or list (one size per block)."""
    if isinstance(n_per_block, int):
        n_per_block = [n_per_block]*len(blocks)
    assert len(n_per_block) == len(blocks)

    triples = []
    for bi, (block, n_b) in enumerate(zip(blocks, n_per_block)):
        for n in range(n_b):
            s = f"<http://fice/{prefix}/s_{bi}_{n}>"
            for p in block:
                f = fanout if isinstance(fanout, int) else fanout[p]
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

#------------------------------------------------------------------- datasets


def gen_fano(seed, n=40, fanout=1, name="fano"):
    triples = design_kg(FANO, prefix = name, n_per_block=n, fanout=fanout)
    queries = star_queries(triples, m=7, prefix=name, seed=seed)
    return name, triples, queries

def gen_fano_unequal(seed):
    #block sizes dependent on seed
    rng = np.random.default_rng(seed)
    sizes = [int(x) for x in rng.integers(20,81,size = len(FANO))]
    print(f"  fano_unequal block sizes: {sizes}")
    return gen_fano(seed, n=sizes, name="fano_unequal")

def gen_fano_fanout(seed):
    return gen_fano(seed, fanout=2, name="fano_fanout")

def gen_ising(seed):
    # 10 predicates, two clusters [0,1,2] and [3,4,5,6], others isolated
    h, J = cluster_coupling(10, clusters=[[0, 1, 2], [3, 4, 5, 6]])
    ising_triples = ising_kg(h, J, prefix="ising", n_subjects=12000)
    ising_queries = star_queries(ising_triples, m=10, prefix="ising", max_per_order=120, seed=seed)
    return "ising", ising_triples, ising_queries

def gen_orthogonal(seed, m=7, n=120,fanout = 1):
    # every subject carries exactly one predicate -> no two subjects share same predicate
    blocks = [(p,) for p in range(m)]
    triples = design_kg(blocks, prefix="orthogonal", n_per_block=n, fanout=fanout)
    queries = star_queries(triples, m=m, prefix = "orthogonal", seed=seed)
    return "orthogonal", triples, queries

def gen_two_clusters(seed, clusters = ((0,1,2),(3,4,5,6)), n=120, fanout = 1):
    # every subject carries all predicates of its cluster; unequal cluster sizes so they are not 1-WL equivalent
    m = sum(len(c) for c in clusters)
    triples = design_kg(list(clusters), prefix="two_clusters", n_per_block=n, fanout=fanout)
    queries = star_queries(triples, m=m, prefix="two_clusters", seed=seed)
    return "two_clusters", triples, queries

def gen_implication(seed, fanout = 1):
    # (p0=>p1): no equivalence, (p2=>p3): near-equivalence, (p4=>p5=>p6): chain implication; Block (p,q) are subjects with both predicates, block (q,) only with q
    blocks = [(0,1), (1,), (2,3), (3,), (4,5,6), (5,6), (6,)]
    sizes = [40, 80, 100, 20, 40, 40, 40]
    triples = design_kg(blocks, prefix="implication",n_per_block=sizes, fanout=fanout)
    queries = star_queries(triples, m=7, prefix="implication", seed=seed)
    return "implication", triples, queries

GENERATORS = {
    "fano": gen_fano,
    "fano_unequal": gen_fano_unequal,
    "fano_fanout": gen_fano_fanout,
    "ising": gen_ising,
    "orthogonal": gen_orthogonal,
    "two_clusters": gen_two_clusters,
    "implication": gen_implication,
}


def main():
    ap = argparse.ArgumentParser(description="Generate controlled KGs and exact star queries.")
    ap.add_argument("datasets", nargs = "+", choices=[*GENERATORS, "all"])
    ap.add_argument("-s", "--seed", type = int, default = 0)
    args = ap.parse_args()
    names = list(GENERATORS) if "all" in args.datasets else args.datasets
    for name in names:
        print(f"Generating {name} (seed = {args.seed}) ...")
        save_kg_and_queries(*GENERATORS[name](args.seed))


if __name__ == "__main__":
    main()
