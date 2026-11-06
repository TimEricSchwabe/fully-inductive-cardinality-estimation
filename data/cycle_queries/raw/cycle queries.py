import json
from rdflib import Graph

g = Graph()
g.parse("cycle_dataset.nt", format="nt")


main_pred = "http://project.org/data/friendsWith"

json_queries = []

# 1. Chains
for length in range(1, 8):
    triples = []
    sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"

    for i in range(length):
        var1 = f"?v{i}"
        var2 = f"?v{i + 1}"
        triples.append([var1, main_pred, var2])
        sparql += f"  {var1} <{main_pred}> {var2} .\n"

    sparql += "}"

    card = 0
    for row in g.query(sparql):
        card = int(row.c)

    json_queries.append({
        "triples": triples,
        "y": card,
        "split": "train",
        "type": f"chain_{length}"
    })

# 2. Cycles
for length in range(3, 8):
    triples = []
    sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"

    for i in range(length):
        var1 = f"?v{i}"
        var2 = "?v0" if i == length - 1 else f"?v{i + 1}"
        triples.append([var1, main_pred, var2])
        sparql += f"  {var1} <{main_pred}> {var2} .\n"

    sparql += "}"

    card = 0
    for row in g.query(sparql):
        card = int(row.c)

    json_queries.append({
        "triples": triples,
        "y": card,
        "split": "train",
        "type": f"cycle_{length}"
    })

# 3. Stars
for degree in range(2, 6):
    triples = []
    sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"

    for i in range(degree):
        var = f"?v{i + 1}"
        triples.append(["?center", main_pred, var])
        sparql += f"  ?center <{main_pred}> {var} .\n"

    sparql += "}"

    card = 0
    for row in g.query(sparql):
        card = int(row.c)

    json_queries.append({
        "triples": triples,
        "y": card,
        "split": "train",
        "type": f"star_{degree}"
    })

# 4. Mismatched
fake_pred = main_pred + "_FAKE"
for length in range(2, 6):
    triples = []

    for i in range(length):
        var1 = f"?v{i}"
        var2 = f"?v{i + 1}"
        pred_to_use = fake_pred if i == length - 1 else main_pred
        triples.append([var1, pred_to_use, var2])

    json_queries.append({
        "triples": triples,
        "y": 0,
        "split": "train",
        "type": f"mismatched_{length}"
    })

with open("cycle_queries.json", "w") as f:
    json.dump(json_queries, f, indent=2)