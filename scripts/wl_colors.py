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

def wl_refine(num_nodes, edges, labels, init_colors, max_rounds, palette,stop_when_stable = True):
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
        if stop_when_stable and stable:
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

def seeded_colors(num_nodes, N, R, seed_rel_ids):
    colors = type_colors(num_nodes, N, R)
    for r in seed_rel_ids:
        colors[N+r] = 3
    return colors


def load_relation_names(name):
    path = fds.DEFAULT_DATA_ROOT / name / "processed" / "relation_to_id.json"
    with open(path, encoding="utf-8") as f:
        rel_to_id = json.load(f)
        names = [None] * len(rel_to_id)
        for iri, rid in rel_to_id.items():
            short = iri.rstrip(">").split("/")[-1]
            names[rid] = short

    return names, rel_to_id

def load_queries(name):
    qname = name.removesuffix("_dataset")+ "_queries"
    ordner = fds.DEFAULT_DATA_ROOT / qname / "raw"
    path = list(ordner.glob("*.json"))[0]
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)
    return queries

def query_rel_ids(query, rel_to_id):
    ids = []
    triples = query["triples"]
    for t in triples:
        p = t[1]
        rid = rel_to_id[p]
        ids.append(rid)
    return ids



def relation_classes(colors, N, R, rel_names):
    groups = defaultdict(list)
    for r in range(R):
        color = colors[N+r]
        name = rel_names[r]
        groups[color].append(name)
    return sorted(sorted(g) for g in groups.values())

def query_signatures(num_nodes, edges, labels, N, R, queries, rel_to_id, names, seeded, rounds=4):
    palette = {}
    groups = defaultdict(list)
    colors = []

    if not seeded:
        init_colors = type_colors(num_nodes, N, R)
        history = wl_refine(num_nodes, edges, labels, init_colors,rounds,palette,False)
        colors = history[rounds]

    for q in queries:
        ids = query_rel_ids(q, rel_to_id)
        if seeded:
            init_colors = seeded_colors(num_nodes, N, R, ids)
            history = wl_refine(num_nodes, edges, labels, init_colors,rounds,palette,False)
            colors = history[rounds]
        sig = []
        for r in ids:
            sig.append(colors[N+r])
        sig = tuple(sorted(sig))
        groups[sig].append(([names[r] for r in ids], q["y"]))
    return groups

def report(groups):
    by_order = defaultdict(list)
    for sig, members in groups.items():
        order = len(sig)
        by_order[order].append(members)
    for order in sorted(by_order):
        group_list = by_order[order]
        n_queries = 0
        for members in group_list:
            n_queries += len(members)
        mixed = []
        for members in group_list:
            ys = set(y for _, y in members)
            if len(ys) > 1:
                mixed.append(members)
        print(f"order {order}: {n_queries} queries, {len(group_list)} groups, {len(mixed)} mixed")
        for members in mixed:
            ys = Counter(y for _, y in members)
            print(f"   mixed group: {dict(ys)}")


def main():
    graph = load_factor_graph(sys.argv[1])
    edges, labels, N, R, num_nodes = graph_to_lists(graph)
    colors = type_colors(num_nodes, N, R)
    names, rel_to_id = load_relation_names(sys.argv[1])
    history = wl_refine(num_nodes, edges, labels, colors, max_rounds=50, palette={})
    colors_stable = history[-1]
    colors_4 = history[min(4, len(history)-1)]
    classes_stable = relation_classes(colors_stable, N, R, names)
    classes_4 = relation_classes(colors_4, N, R, names)

    print(f"{sys.argv[1]}: stable after {len(history)-2} rounds")
    print(f"relation classes after 4 rounds: {classes_4}")
    print(f"relation classes stable: {classes_stable}")

    queries = load_queries(sys.argv[1])
    star_queries = []
    for q in queries:
        if len(q["triples"]) in (2,3) and len({t[0] for t in q["triples"]}) == 1:
            star_queries.append(q)

    for seeded in (False, True):
        print("seeded:" if seeded else "unseeded:")
        report(query_signatures(num_nodes, edges, labels, N, R, star_queries, rel_to_id, names, seeded))






if __name__ == "__main__":
    main()