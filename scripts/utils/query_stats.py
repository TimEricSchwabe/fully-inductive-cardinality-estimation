#!/usr/bin/env python3
"""Count the number of queries in each query dataset under data/."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main():
    for query_dir in sorted(DATA_DIR.iterdir()):
        if "quer" not in query_dir.name:
            continue
        raw_dir = query_dir / "raw"
        if not raw_dir.exists():
            continue
        raw_files = list(raw_dir.iterdir())
        if not raw_files:
            continue
        queries = json.loads(raw_files[0].read_text())
        print(f"{query_dir.name:40s}  queries={len(queries):>8,}")


if __name__ == "__main__":
    main()
