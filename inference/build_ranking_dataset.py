"""Build a sub-BGP ranking dataset for one (KG, shape) pair.

Chain-sampling: each sub-BGP carries a ``prefix_mask`` so the ranking
loss can be filtered to *sibling* pairs — sub-BGPs that share a prefix
and differ by the next extension triple, exactly the decision the
greedy / DP plan enumerator makes.

For each parent query:
  1. Include all n singletons with ``prefix_mask = 0`` (siblings of
     each other — listwise at size 1, free n=1 supervision).
  2. Pick one singleton as the seed. At each depth, enumerate all
     connected one-triple extensions of the current prefix, cap via
     ``--max-siblings-per-step``, record all candidates with
     ``prefix_mask = current``. Pick one uniformly, continue to the
     next depth until the full BGP is reached.
  3. For each kept sub-BGP, query the configured SPARQL endpoint for
     the true cardinality (and the endpoint's own estimate when
     available).
  4. Stream a record per parent to ``<out>/queries.jsonl`` (resumable)
     and repack to a JSON array ``<out>/queries.json`` at the end so
     that ``NTRankingQueries`` can load it.

Each sub-BGP record now carries both ``mask`` and ``prefix_mask``.
Two sub-BGPs are siblings iff they share the same ``prefix_mask`` and
the same ``size``.

Example:
    .venv/bin/python inference/build_ranking_dataset.py \\
        --raw data/dbpedia100k_path_queries/raw/59898644 \\
        --out data/dbpedia100k_path_queries_ranking/raw \\
        --endpoint-url http://localhost:7020/ \\
        --max-queries 100 \\
        --max-siblings-per-step 6 \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _endpoint_local import get_endpoint_action  # noqa: E402

_ENDPOINT_ACTION = get_endpoint_action()


# ---------------------------------------------------------------------------
# Join-graph enumeration (duplicated from dp_subquery_eval for standalone use)
# ---------------------------------------------------------------------------

def _triple_vars(triple):
    return {t for t in triple if isinstance(t, str) and t.startswith("?")}


def build_join_adj(triples):
    n = len(triples)
    tvars = [_triple_vars(t) for t in triples]
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if tvars[i] & tvars[j]:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def connected_subsets(n, adj):
    out = []
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        start = lsb.bit_length() - 1
        visited = 1 << start
        stack = [start]
        while stack:
            u = stack.pop()
            nbrs = adj[u] & mask & ~visited
            while nbrs:
                v_bit = nbrs & -nbrs
                v = v_bit.bit_length() - 1
                visited |= v_bit
                stack.append(v)
                nbrs &= ~v_bit
        if visited == mask:
            out.append(mask)
    return out


def sample_chain(n, adj, max_siblings_per_step, rng):
    """Sample sub-BGPs along a prefix chain so siblings are explicit.

    Structure (one chain per parent):
      1. Depth 1: include all n singletons with ``prefix_mask = 0`` —
         siblings of each other (listwise at size 1).
      2. Pick one singleton uniformly as the seed.
      3. At depth k (current prefix has size k), enumerate every connected
         one-triple extension. If there are more than
         ``max_siblings_per_step`` of them, subsample uniformly. Record
         each kept candidate with ``prefix_mask = current``.
      4. Pick one candidate uniformly, make it the new current, repeat
         until current = full BGP.

    Returns a list of ``(mask, prefix_mask)`` tuples, deduplicated by mask
    (first occurrence wins). Returns ``None`` if the chain cannot reach
    the full BGP (disconnected join graph).
    """
    full = (1 << n) - 1
    items: List[Tuple[int, int]] = []

    # Depth 1: all singletons share the empty prefix.
    for i in range(n):
        items.append((1 << i, 0))

    current = 1 << rng.randrange(n)
    while current != full:
        candidates = []
        for i in range(n):
            bit = 1 << i
            if current & bit:
                continue
            if adj[i] & current:
                candidates.append(current | bit)
        if not candidates:
            return None
        if max_siblings_per_step and len(candidates) > max_siblings_per_step:
            candidates = rng.sample(candidates, max_siblings_per_step)
        prefix_for_these = current
        for m in candidates:
            items.append((m, prefix_for_these))
        current = rng.choice(candidates)

    # Deduplicate, preserving first occurrence of each mask.
    seen = set()
    unique: List[Tuple[int, int]] = []
    for m, pm in items:
        if m not in seen:
            seen.add(m)
            unique.append((m, pm))
    return unique


# ---------------------------------------------------------------------------
# Endpoint COUNT(*) (send=0 avoids materialising results)
# ---------------------------------------------------------------------------

def triples_to_bgp(triples):
    return " .\n  ".join(f"{s} {p} {o}" for s, p, o in triples)


def endpoint_count(endpoint_url, triples, timeout=120):
    """Return (true_card, endpoint_est_card). Raises on HTTP/network failure."""
    sparql = f"SELECT * WHERE {{\n  {triples_to_bgp(triples)}\n}}"
    url = endpoint_url.rstrip("/") + "/?" + urlencode({
        "query": sparql, "action": _ENDPOINT_ACTION, "send": "0"})
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    tree = payload.get("runtimeInformation", {}).get("query_execution_tree")
    if tree is None:
        return -1, float("nan")
    return int(tree.get("result_rows", 0)), float(tree.get("estimated_size", 0))


# ---------------------------------------------------------------------------
# Per-parent processing
# ---------------------------------------------------------------------------

def process_parent(
    qi: int,
    q: dict,
    endpoint_url: str,
    max_siblings_per_step: int,
    max_triples: int,
    workers: int,
    rng: random.Random,
    timeout: int,
) -> dict | None:
    triples = [tuple(t) for t in q["triples"]]
    n = len(triples)
    if n == 0 or n > max_triples:
        return None
    adj = build_join_adj(triples)

    # Require the full BGP to be connected.
    full = (1 << n) - 1
    if full not in set(connected_subsets(n, adj)):
        return None

    chain = sample_chain(n, adj, max_siblings_per_step, rng)
    if chain is None:
        return None
    prefix_by_mask = {m: pm for m, pm in chain}
    masks = [m for m, _ in chain]

    def _call(m):
        sub = [triples[i] for i in range(n) if m & (1 << i)]
        try:
            true_c, est_c = endpoint_count(endpoint_url, sub, timeout=timeout)
        except (HTTPError, URLError, TimeoutError, Exception) as e:
            logger.warning("qi=%d mask=%d endpoint failed: %s", qi, m, e)
            return m, sub, -1, float("nan")
        return m, sub, true_c, est_c

    sub_records = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_call, m) for m in masks]
            for fut in as_completed(futures):
                m, sub, true_c, est_c = fut.result()
                if true_c < 0:
                    continue
                sub_records.append({
                    "triples": [list(t) for t in sub],
                    "y": true_c,
                    "endpoint_estimate": est_c,
                    "mask": int(m),
                    "prefix_mask": int(prefix_by_mask[m]),
                    "size": bin(m).count("1"),
                })
    else:
        for m in masks:
            m, sub, true_c, est_c = _call(m)
            if true_c < 0:
                continue
            sub_records.append({
                "triples": [list(t) for t in sub],
                "y": true_c,
                "endpoint_estimate": est_c,
                "mask": int(m),
                "prefix_mask": int(prefix_by_mask[m]),
                "size": bin(m).count("1"),
            })

    if not sub_records:
        return None
    sub_records.sort(key=lambda r: r["mask"])

    return {
        "triples": q["triples"],
        "y": q.get("y"),
        "x": q.get("x"),
        "n_triples": n,
        "parent_qi": qi,
        "subqueries": sub_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="Path to existing raw queries JSON (list of {triples,y}).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for queries.jsonl (+ queries.json).")
    ap.add_argument("--endpoint-url", default="http://localhost:7020/")
    ap.add_argument("--max-queries", type=int, default=100,
                    help="Cap on parent queries to process (random subsample).")
    ap.add_argument("--max-siblings-per-step", type=int, default=6,
                    help="Cap on sibling extensions recorded at each depth of the "
                         "prefix chain. Set to 0 or a huge number to disable.")
    ap.add_argument("--max-triples", type=int, default=10,
                    help="Skip parents with more triples than this.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent endpoint calls per parent.")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Endpoint request timeout (s).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "queries.jsonl"
    json_path = args.out / "queries.json"

    with open(args.raw) as f:
        raw_queries = json.load(f)
    logger.info("loaded %d raw parents from %s", len(raw_queries), args.raw)

    rng = random.Random(args.seed)
    # Deterministic subsample of parents (identical to the evaluation split
    # behaviour in dp_subquery_eval.py).
    indices = list(range(len(raw_queries)))
    rng.shuffle(indices)
    selected = indices[: args.max_queries]
    logger.info("processing %d parents (seed=%d)", len(selected), args.seed)

    # Resume: skip parent_qi already in jsonl.
    done_qi: set[int] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done_qi.add(int(json.loads(line)["parent_qi"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        if done_qi:
            logger.info("resume: %d parents already in %s", len(done_qi), jsonl_path)

    # Per-parent RNG that depends only on qi + seed so stratified sampling is
    # reproducible and resume-safe.
    n_ok = 0
    n_skip = 0
    t0 = time.perf_counter()
    with open(jsonl_path, "a") as fout:
        for done_i, qi in enumerate(selected):
            if qi in done_qi:
                continue
            parent_rng = random.Random(args.seed * 1_000_003 + qi)
            rec = process_parent(
                qi, raw_queries[qi], args.endpoint_url,
                args.max_siblings_per_step, args.max_triples,
                args.workers, parent_rng, args.timeout,
            )
            if rec is None:
                n_skip += 1
                continue
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_ok += 1
            if (n_ok + n_skip) % 10 == 0:
                elapsed = time.perf_counter() - t0
                rate = (n_ok + n_skip) / max(elapsed, 1e-6)
                logger.info(
                    "progress: %d ok + %d skip / %d target  (%.1f parents/s, %.0fs elapsed)",
                    n_ok, n_skip, len(selected), rate, elapsed,
                )

    # Repack JSONL → JSON array for NTQueries-style loading.
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["parent_qi"])
    with open(json_path, "w") as f:
        json.dump(records, f)

    total_sub = sum(len(r["subqueries"]) for r in records)
    logger.info(
        "done: %d parents (%d sub-BGPs total), wrote %s and %s",
        len(records), total_sub, jsonl_path, json_path,
    )


if __name__ == "__main__":
    main()
