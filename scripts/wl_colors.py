import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))

import fice.datasets as fds
from utils.data_helper import make_factor_graph

def load_factor_graph(name):
    cls = getattr(fds, name)
    ds = cls()
    return make_factor_graph(ds)

def wl_refine(num_nodes, edges, labels, init_colors, max_rounds, palette):
    inc = [[] for _ in range(num_nodes)]
    for (u,v),label in zip(edges, labels):
        inc[v].append((u, label))

    colors = list(init_colors)
    history = [colors]
    for _ in range(max_rounds):
        new = []
        for v in range(num_nodes):
            neighbors = []
            for (u,label) in inc[v]:
                neighbors.append((colors[u], label))
            neighbors = tuple(sorted(neighbors))
            sig = (colors[v], neighbors)
            palette.setdefault(sig, len(palette))
            color = palette[sig]
            new.append(color)
        stable = len(set(new)) == len(set(colors))
        colors = new
        history.append(colors)
        if stable:
            break
    return history

def graph_to_lists(graph):
    edge_u = graph.edge_index[0].tolist()
    edge_v = graph.edge_index[1].tolist()
    edges = list(zip(edge_u, edge_v))
    labels = [tuple(l) for l in graph.edge_attr.tolist()]
    N = graph.original_num_nodes
    R = graph.original_num_relations
    num_nodes = graph.num_nodes
    return edges, labels, N, R, num_nodes

def type_colors(num_nodes, N, R):
    colors = [0]*num_nodes
    for u in range(N):
        colors[u] = 0
    for u in range(R):
        colors[N+u] = 1
    for u in range(num_nodes-N-R):
        colors[N+R+u] = 2
    return colors

def load_relation_names(name):
    path = fds.DEFAULT_DATA_ROOT / name / "processed" / "relation_to_id.json"
    with open(path, encoding="utf-8") as f:
        rel_to_id = json.load(f)
        names = [None] * len(rel_to_id)
        for iri, rid in rel_to_id.items():
            short = iri.rstrip(">").split("/")[-1]
            names[rid] = short

    return names

def relation_classes(colors, N, R, rel_names):
    groups = defaultdict(list)
    for r in range(R):
        color = colors[N+r]
        name = rel_names[r]
        groups[color].append(name)
    return sorted(sorted(g) for g in groups.values())



def main():
    graph = load_factor_graph(sys.argv[1])
    edges, labels, N, R, num_nodes = graph_to_lists(graph)
    colors = type_colors(num_nodes, N, R)
    names = load_relation_names(sys.argv[1])
    history = wl_refine(num_nodes, edges, labels, colors, max_rounds=50, palette={})
    colors_stable = history[-1]
    colors_4 = history[min(3, len(history)-1)]
    classes_stable = relation_classes(colors_stable, N, R, names)
    classes_4 = relation_classes(colors_4, N, R, names)

    print(f"{sys.argv[1]}: stable after {len(history)-2} rounds")
    print(f"relation classes after 4 rounds: {classes_4}")
    print(f"relation classes stable: {classes_stable}")






if __name__ == "__main__":
    main()