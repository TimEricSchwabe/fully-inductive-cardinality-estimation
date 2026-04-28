"""Download CoDEx-Large and convert to a single .nt file."""

import urllib.request
import os

BASE_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l"
SPLITS = ["train.txt", "valid.txt", "test.txt"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "codex_l", "raw")


def download_and_convert():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "codex_l.nt")

    with open(out_path, "w") as out:
        for split in SPLITS:
            url = f"{BASE_URL}/{split}"
            print(f"Downloading {url} ...")
            response = urllib.request.urlopen(url)
            for line in response:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                s, p, o = parts
                out.write(
                    f"<http://www.wikidata.org/entity/{s}> "
                    f"<http://www.wikidata.org/prop/direct/{p}> "
                    f"<http://www.wikidata.org/entity/{o}> .\n"
                )

    # Count lines
    with open(out_path) as f:
        n = sum(1 for _ in f)
    print(f"Wrote {n} triples to {out_path}")


if __name__ == "__main__":
    download_and_convert()
