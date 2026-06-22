import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from rdflib import Graph, URIRef, Namespace

g = Graph()
EX = Namespace("https://project.org/data/")

predicates = [EX.p_1, EX.p_2, EX.p_3, EX.p_4, EX.p_5]
num_chains = 150

node_id = 0


for _ in range(num_chains):
    current_node = URIRef(f"{EX}person_{node_id}")

    for p in predicates:
        next_node = URIRef(f"{EX}person_{node_id + 1}")
        g.add((current_node, p, next_node))
        current_node = next_node
        node_id += 1

    node_id += 1

g.serialize(destination="raw_dataset.nt", format="nt")
