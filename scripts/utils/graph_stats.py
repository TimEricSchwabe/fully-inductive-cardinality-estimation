#!/usr/bin/env python3
"""Count triples, entities, and relations for each graph in data/ by parsing raw N-Triples files."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

GRAPHS = [
    "swdf",
    "fb237_v4",
    "fb237_v4_ind",
    "wn18rr_v4",
    "wn18rr_v4_ind",
    "dbpedia100k",
    "lubm",
    "wikidata",
    "hetionet",
    "yago",
    "codex_l",
    "aids",
]


def find_raw_files(raw_dir: Path) -> list[Path]:
    nt_files = list(raw_dir.glob("*.nt"))
    if nt_files:
        return nt_files
    return [f for f in raw_dir.iterdir() if f.is_file() and not f.suffix]


def count_graph(raw_files: list[Path]) -> tuple[int, int, int]:
    entities: set[str] = set()
    relations: set[str] = set()
    n_triples = 0
    for f in raw_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n\r")
                if not line or line.startswith("#"):
                    continue
                body = line.rsplit(" .", 1)[0]
                tokens = body.split(None, 2)
                if len(tokens) != 3:
                    continue
                s, p, o = tokens
                entities.add(s)
                relations.add(p)
                entities.add(o)
                n_triples += 1
    return n_triples, len(entities), len(relations)


def main():
    for name in sorted(GRAPHS):
        graph_dir = DATA_DIR / name
        raw_files = find_raw_files(graph_dir / "raw")
        n_triples, n_entities, n_relations = count_graph(raw_files)
        print(
            f"{name:20s}  triples={n_triples:>10,}  entities={n_entities:>9,}  relations={n_relations:>5,}"
        )


if __name__ == "__main__":
    main()
