#!/usr/bin/env python3
"""Count triples, entities, and relations for each graph in data/."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

GRAPHS = ["swdf", "fb237_v4", "fb237_v4_ind", "wn18rr_v4", "wn18rr_v4_ind", "dbpedia100k", "lubm", "wikidata"]


def main():
    for name in sorted(GRAPHS):
        graph_dir = DATA_DIR / name

        raw_files = list((graph_dir / "raw").glob("*"))
        n_triples = sum(1 for f in raw_files for _ in open(f))

        with open(graph_dir / "processed" / "entity_to_id.json") as f:
            n_entities = len(json.load(f))
        with open(graph_dir / "processed" / "relation_to_id.json") as f:
            n_relations = len(json.load(f))

        print(
            f"{name:20s}  triples={n_triples:>8,}  entities={n_entities:>7,}  relations={n_relations:>5,}"
        )


if __name__ == "__main__":
    main()
