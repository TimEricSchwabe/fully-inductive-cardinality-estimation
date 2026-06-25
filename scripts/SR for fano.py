import torch
import numpy as np
import sympy
import itertools
from collections import defaultdict
from symtorch import SymbolicModel


# ==========================================
# 1. TIM'S FUNCTIONS (Feature Extraction)
# ==========================================
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


# ==========================================
# 2. TIM'S DATA GENERATORS (Fano Plane)
# ==========================================
FANO = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)]


def design_kg(blocks, n_per_block=40, fanout=1):
    triples = []
    for bi, block in enumerate(blocks):
        for n in range(n_per_block):
            s = f"s_{bi}_{n}"
            for p in block:
                for k in range(fanout):
                    triples.append((s, f"p{p}", f"o_{s}_{p}_{k}"))
    return triples


def _object_counts(triples):
    c = defaultdict(lambda: defaultdict(int))
    for s, p, o in triples:
        c[s][p] += 1
    return c


def star_cardinality(counts, preds):
    total = 0
    for s, cs in counts.items():
        if all(cs.get(f"p{i}", 0) > 0 for i in preds):
            prod = 1
            for i in preds:
                prod *= cs[f"p{i}"]
            total += prod
    return total


def star_queries(triples, m, orders=(1, 2, 3), max_per_order=None, seed=0):
    rng = np.random.default_rng(seed)
    counts = _object_counts(triples)
    out = []
    for k in orders:
        combos = list(itertools.combinations(range(m), k))
        rng.shuffle(combos)
        if max_per_order:
            combos = combos[:max_per_order]
        for preds in combos:
            out.append((preds, star_cardinality(counts, preds)))
    return out


def get_occurrences(triples, m):
    occ = np.zeros(m)
    for s, p, o in triples:
        p_idx = int(p[1:])
        occ[p_idx] += 1
    return occ


# ==========================================
# 3. DATA PREPARATION
# ==========================================
print("Loading model and extracting embeddings...")
model_path = "/Users/tomiris/PycharmProjects/fully-inductive-cardinality-estimation/output/run_2026-06-12_16-58-10/checkpoints/best_model.pt"
checkpoint = torch.load(model_path, map_location='cpu')
R = checkpoint['model_state_dict']['embedding_model.edge_encoder.weight'].detach().numpy()

print("Generating the Fano Plane Knowledge Graph...")
m = 7
fano_triples = design_kg(FANO, n_per_block=40)
queries = star_queries(fano_triples, m=m, orders=(3,))
actual_occ = get_occurrences(fano_triples, m)

X_list = []
y_list = []

for preds, card in queries:
    features = star_features(preds, R, actual_occ)
    X_list.append(features)

    indep_val = features[0]
    y_val = np.log1p(card) - indep_val
    y_list.append(y_val)

X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
# ==========================================
# ==========================================
# 4. SYMBOLIC REGRESSION VIA SYMTORCH
# ==========================================
print("Setting up the SymTorch framework...")

# Create a tiny custom PyTorch block that outputs our target data
class RecipeABlock(torch.nn.Module):
    def __init__(self, targets):
        super().__init__()
        self.targets = targets
    def forward(self, x):
        return self.targets

# Wrap our custom block inside the SymbolicModel constructor
base_block = RecipeABlock(y)
sr_model = SymbolicModel(base_block)

# Run a quick dummy forward pass to cache X and y data inside the wrapper
_ = sr_model(X)

print("Starting evolutionary symbolic regression...")

# Define operators using Tim's exact naming convention from the PDF
binary_ops = ["+", "-", "*", "/"]
unary_ops = [
    "exp",
    "log",
    "square",
    "sqrt",
    "relu(x) = max(x, 0f0)",
    "log1p(x) = x > -1f0 ? log(1f0 + x) : Float32(NaN)"  # Safe one-liner check
]
# We need to tell SymPy how to visually represent these custom operators in Python
import sympy
custom_sympy_mappings = {
    "relu": lambda x: sympy.Max(x, 0),
    "log1p": lambda x: sympy.log(1 + x)
}

# Call the official .distill() method with the updated mappings
sr_model.distill(
    X,
    sr_params={
        "binary_operators": binary_ops,
        "unary_operators": unary_ops,
        "extra_sympy_mappings": custom_sympy_mappings
    }
)

print("Search complete!")