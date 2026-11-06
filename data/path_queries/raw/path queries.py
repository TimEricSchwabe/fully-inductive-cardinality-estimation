import json
from rdflib import Graph, Namespace

g = Graph()
g.parse("data/path_dataset/raw/raw_dataset.nt", format="nt")
EX = Namespace("https://project.org/data/")

predicates = [EX.p_1, EX.p_2, EX.p_3, EX.p_4, EX.p_5]
json_queries = []


for length in range(1, 6):
    current_preds = predicates[:length]

    sparql = "SELECT (COUNT(*) AS ?c) WHERE {\n"
    triples = []

    for i, p in enumerate(current_preds):
        var1 = f"?v{i}"
        var2 = f"?v{i + 1}"
        sparql += f"  {var1} <{p}> {var2} .\n"
        triples.append([var1, str(p), var2])
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

output_file = "data/path_queries/raw/path_queries.json"
with open(output_file, "w") as f:
    json.dump(json_queries, f, indent=2)

