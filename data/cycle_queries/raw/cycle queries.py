import json
#from rdflib import Graph
from collections import defaultdict

#g = Graph()
#g.parse("cycle_dataset.nt", format="nt")

succ = defaultdict(list)
nodes = set()
with open("cycle_dataset.nt", encoding="utf-8") as f:
    for line in f:
        s, p, o = line.strip().split(" ")[:3]
        succ[s].append(o)
        nodes.update((s,o))

main_pred = "https://project.org/data/friendsWith"

def count_chains(length):
    cur = {u: 1 for u in nodes}
    for _ in range(length):
        nxt = defaultdict(int)
        for u in cur:
            for v in succ[u]:
                nxt[v] += cur[u]
        cur = nxt
    return sum(cur.values())

def count_cycles(length):
    total = 0
    for s in nodes:
        cur = {s: 1}
        for _ in range(length):
            nxt = defaultdict(int)
            for u in cur:
                for v in succ[u]:
                    nxt[v] += cur[u]
            cur = nxt
        total+= cur.get(s, 0)
    return total

def count_stars(degree):
    total = 0
    for c in nodes:
        n = len(succ[c])** degree
        total += n
    return total

json_queries = []

# 1. Chains
for length in range(1, 16):
    triples = []
    #sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"

    for i in range(length):
        var1 = f"?v{i}"
        var2 = f"?v{i + 1}"
        triples.append([var1, f"<{main_pred}>", var2])
    #    sparql += f"  {var1} <{main_pred}> {var2} .\n"

    #sparql += "}"

    #card = 0
    #for row in g.query(sparql):
    #    card = int(row.c)

    card = count_chains(length)

    json_queries.append({
        "triples": triples,
        "y": card,
        "split": "train",
        "type": f"chain_{length}"
    })

# 2. Cycles
for length in range(3, 16):
    triples = []
    #sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"

    for i in range(length):
        var1 = f"?v{i}"
        var2 = "?v0" if i == length - 1 else f"?v{i + 1}"
        triples.append([var1, f"<{main_pred}>", var2])
    #    sparql += f"  {var1} <{main_pred}> {var2} .\n"

    #sparql += "}"

    #card = 0
    #for row in g.query(sparql):
    #    card = int(row.c)

    card =count_cycles(length)

    json_queries.append({
        "triples": triples,
        "y": card,
        "split": "train",
        "type": f"cycle_{length}"
    })

# 3. Stars
for degree in range(2, 11):
    triples = []
    #sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"

    for i in range(degree):
        var = f"?v{i + 1}"
        triples.append(["?center", f"<{main_pred}>", var])
    #    sparql += f"  ?center <{main_pred}> {var} .\n"

    #sparql += "}"

    #card = 0
    #for row in g.query(sparql):
    #   card = int(row.c)

    card =count_stars(degree);

    json_queries.append({
        "triples": triples,
        "y": card,
        "split": "train",
        "type": f"star_{degree}"
    })

# 4. Mismatched
fake_pred = main_pred + "_FAKE"
for length in range(2, 15):
    triples = []

    for i in range(length):
        var1 = f"?v{i}"
        var2 = f"?v{i + 1}"
        pred_to_use = fake_pred if i == length - 1 else main_pred
        triples.append([var1, f"<{pred_to_use}>", var2])

    json_queries.append({
        "triples": triples,
        "y": 0,
        "split": "train",
        "type": f"mismatched_{length}"
    })

with open("cycle_queries.json", "w") as f:
    json.dump(json_queries, f, indent=2)