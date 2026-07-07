"""DP-style subquery evaluation: model prediction vs SPARQL endpoint ground truth.

For each input BGP query we:
  1. Enumerate all connected subsets of its triples (join graph: triples share a variable).
  2. For every subset, compute predicted cardinality via the TorchScript model and
     true cardinality via a local SPARQL endpoint (COUNT(*)).
  3. Run DP over the subsets with c-out cost to pick the best plan. c-out of the
     plan is reported twice — once using predicted sizes (the planner's view) and
     once by rebuilding the same plan with true sizes (what actually happens).

Usage (from scripts/):
    uv run python ../inference/dp_subquery_eval.py \
        --config ../inference/inference_config.yaml \
        --queries ../data/fb237_v4_star_queries/raw/fb237_v4_star_queries \
        --endpoint-url http://localhost:7020/ \
        --output-dir ../output/dp_subquery/ \
        --max-queries 200
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from _endpoint_local import get_endpoint_action  # noqa: E402

_ENDPOINT_ACTION = get_endpoint_action()


# ---------------------------------------------------------------------------
# Subset enumeration over the join graph
# ---------------------------------------------------------------------------

def triple_vars(triple):
    return {t for t in triple if isinstance(t, str) and t.startswith("?")}


def build_join_adjacency(triples):
    """Adjacency list: triple i and j are connected iff they share a variable."""
    n = len(triples)
    tvars = [triple_vars(t) for t in triples]
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if tvars[i] & tvars[j]:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def connected_subsets(n, adj):
    """Yield all connected subsets (bitmasks) of the join graph. At most 2^n - 1."""
    out = []
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        start = (lsb.bit_length() - 1)
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


def iter_submasks(mask):
    """Iterate over all non-empty proper sub-masks of `mask` (classic trick)."""
    sub = (mask - 1) & mask
    while sub > 0:
        yield sub
        sub = (sub - 1) & mask


# ---------------------------------------------------------------------------
# Build query graph tensors for scripted model (subset view)
# ---------------------------------------------------------------------------

def build_query_tensors(sub_triples, entity_emb_np, relation_emb_np,
                        entity_occ_np, relation_occ_np, uri_to_id,
                        feat_dim, graph_num_edges):
    entity_ids = uri_to_id["entities"]
    relation_ids = uri_to_id["relations"]

    node_map = {}
    src, dst, rel_ids = [], [], []
    for s, p, o in sub_triples:
        if s not in node_map:
            node_map[s] = len(node_map)
        if o not in node_map:
            node_map[o] = len(node_map)
        src.append(node_map[s])
        dst.append(node_map[o])
        rel_ids.append(relation_ids.get(p.strip("<>"), -1))

    num_nodes = len(node_map)
    num_edges = len(sub_triples)

    x = torch.zeros(num_nodes, feat_dim)
    for name, lid in node_map.items():
        if not name.startswith("?"):
            eid = entity_ids.get(name.strip("<>"), -1)
            if eid >= 0:
                x[lid, :3] = torch.from_numpy(entity_occ_np[eid].copy())
                x[lid, 3:] = torch.from_numpy(entity_emb_np[eid].copy())

    edge_attr = torch.zeros(num_edges, feat_dim)
    for e, rid in enumerate(rel_ids):
        if rid >= 0:
            edge_attr[e, :3] = torch.from_numpy(relation_occ_np[rid].copy())
            edge_attr[e, 3:] = torch.from_numpy(relation_emb_np[rid].copy())

    x[:, :3] = torch.log1p(x[:, :3])
    edge_attr[:, :3] = torch.log1p(edge_attr[:, :3])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    graph_size = torch.tensor([float(graph_num_edges)])
    return x, edge_index, edge_attr, batch, graph_size


# ---------------------------------------------------------------------------
# Endpoint COUNT(*)
# ---------------------------------------------------------------------------

def triples_to_bgp(triples):
    return " .\n  ".join(f"{s} {p} {o}" for s, p, o in triples)


def endpoint_bgp_stats(endpoint_url, triples, timeout=120):
    """Run `SELECT * WHERE { BGP }` (send=0) and return
    (endpoint_estimated_card, true_card) from the root of the execution tree.

    send=0 skips result materialization to the response, keeping the call cheap
    even for large BGP results.
    """
    sparql = f"SELECT * WHERE {{\n  {triples_to_bgp(triples)}\n}}"
    url = endpoint_url.rstrip("/") + "/?" + urlencode({
        "query": sparql, "action": _ENDPOINT_ACTION, "send": "0"})
    req = Request(url)
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Endpoint HTTP {e.code} for BGP:\n{sparql}\n-> {body}")
    tree = payload.get("runtimeInformation", {}).get("query_execution_tree")
    if tree is None:
        return float("nan"), 0
    return float(tree.get("estimated_size", 0)), int(tree.get("result_rows", 0))


def endpoint_plan_cout(endpoint_url, triples, timeout=120):
    """Run `SELECT * WHERE { BGP }` and sum estimated_size / result_rows over
    join-style operations of the endpoint's chosen plan.

    Returns (cout_pred, cout_true, n_ops).

    Excluded node kinds:
      - IndexScan: base relation, not an intermediate join result.
      - Sort: QLever's Sort operator's result_rows always equals its
        single child's result_rows (it reorders rows, doesn't change cardinality).
        Counting both double-counts that intermediate size and inflates c-out
        for the endpoint relative to FICE-DP's c-out (which sums one value per
        non-singleton subset). Skip Sort to keep the c-out metric comparable.
    """
    sparql = f"SELECT * WHERE {{\n  {triples_to_bgp(triples)}\n}}"
    url = endpoint_url.rstrip("/") + "/?" + urlencode({
        "query": sparql, "action": _ENDPOINT_ACTION, "send": "0"})
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    tree = payload.get("runtimeInformation", {}).get("query_execution_tree")
    if tree is None:
        return float("nan"), float("nan"), 0

    cout_pred = 0.0
    cout_true = 0.0
    n_ops = 0
    SKIP_PREFIXES = ("IndexScan", "Sort")

    def walk(node):
        nonlocal cout_pred, cout_true, n_ops
        desc = node.get("description", "")
        skip = any(desc.startswith(p) for p in SKIP_PREFIXES)
        if not skip:
            cout_pred += float(node.get("estimated_size", 0))
            cout_true += float(node.get("result_rows", 0))
            n_ops += 1
        for ch in node.get("children", []) or []:
            walk(ch)

    walk(tree)
    return cout_pred, cout_true, n_ops


# ---------------------------------------------------------------------------
# DP with c-out cost
# ---------------------------------------------------------------------------

def dp_best_plan(connected_masks, card):
    """DP over connected subsets with c-out cost.

    cost[S]       = minimum sum of intermediate sizes to produce S.
    plan[S]       = (left_mask, right_mask) or None if S is a singleton.
    Returns the full subset -> (cost, plan) mapping and the best full plan.
    """
    full_mask = 0
    for m in connected_masks:
        full_mask |= m
    mask_set = set(connected_masks)

    cost = {}
    plan = {}
    for m in sorted(connected_masks, key=lambda x: bin(x).count("1")):
        if bin(m).count("1") == 1:
            cost[m] = 0.0
            plan[m] = None
            continue
        best = None
        seen = set()
        for left in iter_submasks(m):
            if left in seen:
                continue
            right = m ^ left
            seen.add(left)
            seen.add(right)
            if left not in mask_set or right not in mask_set:
                continue
            c = cost[left] + cost[right] + float(card[m])
            if best is None or c < best[0]:
                best = (c, left, right)
        cost[m] = best[0] if best else float("inf")
        plan[m] = (best[1], best[2]) if best else None
    return cost, plan, full_mask


def plan_actual_cost(plan, full_mask, card):
    """Recompute c-out of the chosen plan using a different card table."""
    memo = {}

    def rec(m):
        if m in memo:
            return memo[m]
        p = plan[m]
        if p is None:
            memo[m] = 0.0
            return 0.0
        l, r = p
        v = rec(l) + rec(r) + float(card[m])
        memo[m] = v
        return v

    return rec(full_mask)


# ---------------------------------------------------------------------------
# Confidence-aware (UCB) cost
# ---------------------------------------------------------------------------

# Default size -> log10-stddev table, calibrated from the fb237 path-query run
# (FICE log-ratio std per subquery size). Overridable via --sigma-yaml.
DEFAULT_SIGMA_BY_SIZE = {
    1: 0.88, 2: 1.06, 3: 1.01, 4: 1.09, 5: 0.90,
    6: 1.05, 7: 1.27, 8: 1.20, 9: 1.46, 10: 1.55,
}
DEFAULT_SIGMA_FALLBACK = 1.5


def make_ucb_card(pred_card, sigma_by_size, k, fallback):
    """Inflate each subset's predicted card by 10**(k * sigma(|S|))."""
    ucb = {}
    for m, c in pred_card.items():
        n_bits = bin(m).count("1")
        sigma = sigma_by_size.get(n_bits, fallback)
        ucb[m] = float(c) * (10.0 ** (k * sigma))
    return ucb


def make_shift_card(pred_card, beta):
    """Apply a global log10 shift to FICE predictions: pred * 10**beta."""
    factor = 10.0 ** beta
    return {m: float(c) * factor for m, c in pred_card.items()}


def make_geom_card(pred_card, endpoint_est):
    """Per-subquery geometric mean of FICE and endpoint estimates:
    sqrt(max(fice, 1) * max(endpoint, 1)).  Undefined endpoint values fall
    back to FICE; undefined FICE values fall back to the endpoint.
    """
    out = {}
    for m, fice in pred_card.items():
        q = endpoint_est.get(m, float("nan"))
        f_ok = np.isfinite(fice) and fice > 0
        q_ok = np.isfinite(q) and q > 0
        if f_ok and q_ok:
            out[m] = float(np.sqrt(max(fice, 1.0) * max(q, 1.0)))
        elif f_ok:
            out[m] = float(fice)
        elif q_ok:
            out[m] = float(q)
        else:
            out[m] = float("inf")
    return out


def fit_beta(query_records):
    """Median log10(true / fice_pred) across all usable subqueries."""
    diffs = []
    for rec in query_records:
        for m, tc in rec["true_card"].items():
            pc = rec["pred_card"].get(m, 0.0)
            if tc is None or tc <= 0 or pc <= 0:
                continue
            diffs.append(np.log10(max(tc, 1.0) / max(pc, 1.0)))
    if not diffs:
        return 0.0
    return float(np.median(diffs))


# ---------------------------------------------------------------------------
# Left-linear greedy planner
# ---------------------------------------------------------------------------

def greedy_left_linear(n, adj, masks, card, seed_kind="singleton"):
    """Left-deep greedy: seed with cheapest singleton (`seed_kind='singleton'`)
    or cheapest connected 2-triple subset (`seed_kind='pair'`), then at each
    step append the connected triple that minimises card(prefix ∪ {t}).

    Pair-seeding avoids FICE's unreliable single-triple estimates on graphs
    it has not trained on (FICE was trained on multi-triple queries only).

    Returns (cost_pred, plan, full_mask) with the same shape as dp_best_plan
    so plan_actual_cost works unchanged.
    """
    mask_set = set(masks)
    full_mask = 0
    for m in masks:
        full_mask |= m

    plan = {}
    # Singletons
    singletons = [1 << i for i in range(n) if (1 << i) in mask_set]
    if not singletons:
        return {}, {}, 0
    for s in singletons:
        plan[s] = None

    cost_total = 0.0
    if seed_kind == "pair" and n >= 2:
        # Find the cheapest connected 2-triple subset.
        best_pair = None  # (card, i, j, mask)
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue  # disconnected
                pmask = (1 << i) | (1 << j)
                if pmask not in mask_set:
                    continue
                c = card.get(pmask, float("inf"))
                if best_pair is None or c < best_pair[0]:
                    best_pair = (c, i, j, pmask)
        if best_pair is not None:
            _, i, j, pmask = best_pair
            plan[pmask] = (1 << i, 1 << j)
            cost_total = float(card.get(pmask, 0.0))
            prefix = pmask
            remaining = [k for k in range(n) if not (prefix & (1 << k))]
        else:
            # Graph has no connected pair (shouldn't happen for connected BGPs
            # with n >= 2) — fall through to singleton seeding.
            seed = min(singletons, key=lambda s: card.get(s, float("inf")))
            prefix = seed
            remaining = [i for i in range(n) if not (prefix & (1 << i))]
    else:
        # Seed with cheapest singleton.
        seed = min(singletons, key=lambda s: card.get(s, float("inf")))
        prefix = seed
        remaining = [i for i in range(n) if not (prefix & (1 << i))]

    while remaining:
        best = None  # (card, idx, new_mask)
        for i in remaining:
            tbit = 1 << i
            new_mask = prefix | tbit
            # connected = prefix's bit-set shares a join-graph edge with i
            if not (adj[i] & prefix):
                continue
            if new_mask not in mask_set:
                continue
            c_new = card.get(new_mask, float("inf"))
            if best is None or c_new < best[0]:
                best = (c_new, i, new_mask)
        if best is None:
            # Disconnected → fall back to cheapest remaining (shouldn't happen
            # for a connected BGP, but be safe).
            for i in remaining:
                tbit = 1 << i
                new_mask = prefix | tbit
                if new_mask not in mask_set:
                    continue
                c_new = card.get(new_mask, float("inf"))
                if best is None or c_new < best[0]:
                    best = (c_new, i, new_mask)
        if best is None:
            break
        c_new, i, new_mask = best
        plan[new_mask] = (prefix, 1 << i)
        cost_total += float(c_new)
        prefix = new_mask
        remaining.remove(i)

    return {full_mask: cost_total}, plan, full_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--endpoint-url", default="http://localhost:7020/")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--max-triples", type=int, default=10,
                    help="Skip queries with more than this many triples "
                         "(2^n blow-up guard)")
    ap.add_argument("--planners", default="dp,greedy,greedy_pair",
                    help="Comma list: dp, greedy (singleton seed), "
                         "greedy_pair (2-triple seed)")
    ap.add_argument("--costs", default="cout,cout_ucb",
                    help="Comma list: cout, cout_ucb")
    ap.add_argument("--ucb-k", type=float, default=1.0,
                    help="Multiplier on log10-sigma in the UCB cost")
    ap.add_argument("--sigma-yaml", default=None,
                    help="YAML mapping {size: sigma}; overrides defaults")
    ap.add_argument("--beta", type=float, default=None,
                    help="Scalar log10-shift for FICE predictions "
                         "(fice_shift_card = pred * 10**beta). "
                         "If omitted, fitted as the median log10(true/pred) "
                         "over all usable subqueries of this run.")
    args = ap.parse_args()

    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    costs = [c.strip() for c in args.costs.split(",") if c.strip()]
    for p in planners:
        if p not in ("dp", "greedy", "greedy_pair"):
            ap.error(f"unknown planner: {p}")
    for c in costs:
        if c not in ("cout", "cout_ucb"):
            ap.error(f"unknown cost: {c}")
    combos = [(p, c) for p in planners for c in costs]

    import yaml
    with open(args.config) as f:
        inf_cfg = yaml.safe_load(f)
    artifacts_dir = Path(inf_cfg["output_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_by_size = dict(DEFAULT_SIGMA_BY_SIZE)
    if args.sigma_yaml:
        with open(args.sigma_yaml) as f:
            override = yaml.safe_load(f) or {}
        sigma_by_size.update({int(k): float(v) for k, v in override.items()})
    logger.info(f"Sigma table: {sigma_by_size} (k={args.ucb_k}, "
                f"fallback={DEFAULT_SIGMA_FALLBACK})")
    logger.info(f"Planner/cost combos: {combos}")

    # ---- Load queries (80/20 train/val split, same seed as compare.py) ----
    with open(args.queries) as f:
        raw_queries = json.load(f)
    n_total = len(raw_queries)
    random.seed(42)
    indices = list(range(n_total))
    random.shuffle(indices)
    split = int(n_total * 0.8)
    val_indices = indices[split:]
    queries = [raw_queries[i] for i in val_indices[:args.max_queries]]
    logger.info(f"Loaded {len(queries)} val queries "
                f"(out of {len(val_indices)} val / {n_total} total)")

    # ---- Load scripted model + artifacts ----
    scripted = torch.jit.load(str(artifacts_dir / "query_gnn_scripted.pt"))
    scripted.eval()

    with open(artifacts_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(artifacts_dir / "uri_to_id.json") as f:
        uri_to_id = json.load(f)

    ne, nr, ed = metadata["num_entities"], metadata["num_relations"], metadata["embedding_dim"]
    entity_emb = np.fromfile(str(artifacts_dir / "entity_embeddings.bin"),
                             dtype=np.float32).reshape(ne, ed)
    relation_emb = np.fromfile(str(artifacts_dir / "relation_embeddings.bin"),
                               dtype=np.float32).reshape(nr, ed)
    entity_occ = np.fromfile(str(artifacts_dir / "entity_occurrences.bin"),
                             dtype=np.float32).reshape(ne, 3)
    relation_occ = np.fromfile(str(artifacts_dir / "relation_occurrences.bin"),
                               dtype=np.float32).reshape(nr, 3)
    feat_dim = ed + 3
    graph_num_edges = int(metadata.get("num_edges", 33916))

    def predict_log(triples):
        with torch.no_grad():
            tensors = build_query_tensors(
                triples, entity_emb, relation_emb, entity_occ, relation_occ,
                uri_to_id, feat_dim, graph_num_edges)
            return float(scripted(*tensors).item())

    # ---- Output files ----
    sub_tsv = output_dir / "subqueries.tsv"
    plan_tsv = output_dir / "plans.tsv"
    plans_jsonl = output_dir / "plans_trees.jsonl"

    # Each combo is a (column_name, planner, cost, estimator) tuple.
    # `estimator` ∈ {"fice", "fice_shift", "fice_endpoint_geom", "endpoint"}.
    # The base set mirrors the existing --planners × --costs grid; the
    # "extra" set adds fice_shift and fice_endpoint_geom variants for DP and
    # greedy_pair only (per user request).
    combo_specs = []
    for p, c in combos:
        suffix = "ucb" if c == "cout_ucb" else "cout"
        combo_specs.append((f"{p}_{suffix}", p, c, "fice"))

    EXTRA_COMBOS = [
        ("dp_shift",           "dp",          "cout", "fice_shift"),
        ("dp_geom",            "dp",          "cout", "fice_endpoint_geom"),
        ("greedy_pair_shift",  "greedy_pair", "cout", "fice_shift"),
        ("greedy_pair_geom",   "greedy_pair", "cout", "fice_endpoint_geom"),
    ]
    combo_specs += EXTRA_COMBOS

    # Apples-to-apples: left-linear greedy driven by the endpoint's per-subquery
    # estimates (same planner, different estimator).
    ENDPOINT_GREEDY_COL = "endpoint_greedy"
    combo_specs.append((ENDPOINT_GREEDY_COL, "greedy", "cout", "endpoint"))

    combo_cols = [c[0] for c in combo_specs]

    # ---- Phase 1: measurement pass (FICE preds + endpoint stats + endpoint plan) ----
    logger.info("Phase 1/2: measuring subqueries (FICE + endpoint) ...")
    query_records = []  # in-memory cache for phase 2
    with open(sub_tsv, "w") as fs:
        fs.write("query_idx\tmask\tn_triples\tpred_log\tpred_card\ttrue_card\t"
                 "endpoint_est_card\tfice_q_error\tendpoint_q_error\n")
        for qi, q in enumerate(queries):
            triples = [tuple(t) for t in q["triples"]]
            n = len(triples)
            if n > args.max_triples:
                logger.info(f"[{qi}] skip: {n} triples")
                continue
            t0 = time.perf_counter()

            adj = build_join_adjacency(triples)
            masks = connected_subsets(n, adj)

            # Skip BGPs whose join graph (by shared variable) is disconnected —
            # e.g. path queries whose triples are linked only through a URI
            # constant. The planner has no meaningful full-BGP plan here.
            if n > 0 and ((1 << n) - 1) not in set(masks):
                logger.info(f"[{qi}] skip: join graph disconnected "
                            f"({n} triples, {len(masks)} connected subsets)")
                continue

            pred_card, true_card, endpoint_est = {}, {}, {}
            failed = 0
            for m in masks:
                sub = [triples[i] for i in range(n) if m & (1 << i)]
                plog = predict_log(sub)
                pc = float(np.expm1(max(plog, 0.0)))
                try:
                    qe_est, tc = endpoint_bgp_stats(args.endpoint_url, sub)
                except Exception:
                    failed += 1
                    qe_est, tc = float("nan"), -1
                pred_card[m] = pc
                true_card[m] = tc
                endpoint_est[m] = qe_est
                if tc < 0 or not np.isfinite(qe_est):
                    fice_qe, ql_qe = float("nan"), float("nan")
                else:
                    fice_qe = max(max(pc, 1.0) / max(tc, 1.0),
                                  max(tc, 1.0) / max(pc, 1.0))
                    ql_qe = max(max(qe_est, 1.0) / max(tc, 1.0),
                                max(tc, 1.0) / max(qe_est, 1.0))
                fs.write(f"{qi}\t{m}\t{bin(m).count('1')}\t{plog:.6f}\t"
                         f"{pc:.4f}\t{tc}\t{qe_est:.4f}\t"
                         f"{fice_qe:.4f}\t{ql_qe:.4f}\n")
            if failed:
                logger.info(f"  [{qi}] {failed}/{len(masks)} subqueries failed "
                            f"(likely endpoint OOM)")

            try:
                ql_pred, ql_true, ql_ops = endpoint_plan_cout(args.endpoint_url, triples)
            except Exception as e:
                logger.warning(f"[{qi}] endpoint plan fetch failed: {e}")
                ql_pred, ql_true, ql_ops = float("nan"), float("nan"), 0

            full_mask = 0
            for m in masks:
                full_mask |= m

            query_records.append({
                "qi": qi, "n": n, "triples": triples, "adj": adj,
                "masks": masks, "full_mask": full_mask,
                "pred_card": pred_card, "true_card": true_card,
                "endpoint_est": endpoint_est,
                "ql_pred_cout": ql_pred, "ql_true_cout": ql_true,
                "ql_n_ops": ql_ops,
                "phase1_elapsed": time.perf_counter() - t0,
            })
            fs.flush()

            if (qi + 1) % 10 == 0:
                logger.info(f"  [{qi+1}/{len(queries)}] n={n} "
                            f"subsets={len(masks)}")

    # ---- β fit (uses all measured subqueries; one scalar, no overfit risk) ----
    if args.beta is not None:
        beta = float(args.beta)
        logger.info(f"Using β = {beta:+.4f} from --beta")
    else:
        beta = fit_beta(query_records)
        logger.info(f"Fitted β = {beta:+.4f} "
                    f"(fice_shift = fice * 10**β = {10**beta:.3f}×)")

    # ---- Phase 2: planning pass with all combos / estimators ----
    logger.info("Phase 2/2: planning with all combos ...")
    plan_cols = ["query_idx", "n_triples", "n_subsets",
                 "pred_root_card", "true_root_card", "beta_used"]
    for col in combo_cols:
        plan_cols += [f"{col}_cout_pred", f"{col}_cout_true"]
    # Legacy columns for back-compat with older summary readers.
    plan_cols += ["cout_pred", "cout_true_same_plan", "cout_q_error",
                  "endpoint_cout_pred", "endpoint_cout_true", "endpoint_n_ops",
                  "elapsed_s"]
    with open(plan_tsv, "w") as fp, open(plans_jsonl, "w") as fj:
        fp.write("\t".join(plan_cols) + "\n")

        for rec in query_records:
            qi, n, adj, masks = rec["qi"], rec["n"], rec["adj"], rec["masks"]
            pred_card = rec["pred_card"]
            true_card = rec["true_card"]
            endpoint_est = rec["endpoint_est"]
            full_mask = rec["full_mask"]

            # Estimator card dicts.
            fice_card = pred_card
            fice_shift_card = make_shift_card(pred_card, beta)
            fice_geom_card = make_geom_card(pred_card, endpoint_est)
            endpoint_card = {m: (endpoint_est[m] if np.isfinite(endpoint_est[m])
                               else float("inf"))
                           for m in masks}
            est_cards = {
                "fice":             fice_card,
                "fice_shift":       fice_shift_card,
                "fice_endpoint_geom": fice_geom_card,
                "endpoint":           endpoint_card,
            }

            combo_results = {}
            tree_dump = {}
            for (col, p_name, c_name, est_name) in combo_specs:
                base = est_cards[est_name]
                if c_name == "cout_ucb":
                    ct = make_ucb_card(base, sigma_by_size,
                                       args.ucb_k, DEFAULT_SIGMA_FALLBACK)
                else:
                    ct = base
                if p_name == "dp":
                    cp, pl, fm = dp_best_plan(masks, ct)
                elif p_name == "greedy_pair":
                    cp, pl, fm = greedy_left_linear(n, adj, masks, ct,
                                                    seed_kind="pair")
                else:  # "greedy"
                    cp, pl, fm = greedy_left_linear(n, adj, masks, ct,
                                                    seed_kind="singleton")
                cout_pred_d = cp.get(fm, float("nan"))
                cout_true_d = plan_actual_cost(pl, fm, true_card)
                combo_results[col] = (cout_pred_d, cout_true_d)
                tree_dump[col] = {
                    "root": int(fm),
                    "plan": {str(k): (None if v is None
                                      else [int(v[0]), int(v[1])])
                             for k, v in pl.items()},
                }

            # Legacy "cout" baseline = dp_cout (unshifted FICE).
            legacy_pred, legacy_true = combo_results.get(
                "dp_cout", (float("nan"), float("nan")))
            legacy_qe = (max(max(legacy_pred, 1.0) / max(legacy_true, 1.0),
                             max(legacy_true, 1.0) / max(legacy_pred, 1.0))
                         if np.isfinite(legacy_pred)
                         and np.isfinite(legacy_true) else float("nan"))

            row = [str(qi), str(n), str(len(masks)),
                   f"{pred_card[full_mask]:.4f}", str(true_card[full_mask]),
                   f"{beta:.6f}"]
            for col in combo_cols:
                cp_v, ct_v = combo_results[col]
                row += [f"{cp_v:.4f}", f"{ct_v:.4f}"]
            row += [f"{legacy_pred:.4f}", f"{legacy_true:.4f}",
                    f"{legacy_qe:.4f}",
                    f"{rec['ql_pred_cout']:.4f}",
                    f"{rec['ql_true_cout']:.4f}",
                    str(rec["ql_n_ops"]),
                    f"{rec['phase1_elapsed']:.3f}"]
            fp.write("\t".join(row) + "\n")
            fj.write(json.dumps({"query_idx": qi, "plans": tree_dump}) + "\n")

    # ---- Summary ----
    logger.info("Done. Summarizing...")
    import pandas as pd
    sub_all = pd.read_csv(sub_tsv, sep="\t")
    plan_all = pd.read_csv(plan_tsv, sep="\t")

    # Drop subquery rows where endpoint failed (true_card=-1 or NaN estimate).
    sub = sub_all[(sub_all["true_card"] >= 0) &
                  np.isfinite(sub_all["endpoint_est_card"])].copy()
    dropped_sub = len(sub_all) - len(sub)
    if dropped_sub:
        logger.info(f"  Dropped {dropped_sub}/{len(sub_all)} subquery rows "
                    f"with failed endpoint calls")
    # Drop full-query plan rows with missing endpoint data.
    plan = plan_all[(plan_all["true_root_card"] >= 0) &
                    np.isfinite(plan_all["cout_true_same_plan"])].copy()
    dropped_plan = len(plan_all) - len(plan)
    if dropped_plan:
        logger.info(f"  Dropped {dropped_plan}/{len(plan_all)} plan rows "
                    f"with failed endpoint calls")

    def pct(x, p): return np.nanpercentile(x, p)

    print("\n===== SUBQUERY Q-ERROR: FICE vs the endpoint's own estimator =====")
    for name, col in [("FICE", "fice_q_error"), ("endpoint", "endpoint_q_error")]:
        qe = sub[col].values
        print(f"  {name}:  N={np.isfinite(qe).sum()}  "
              f"median={np.nanmedian(qe):.2f}  mean={np.nanmean(qe):.2f}  "
              f"p90={pct(qe,90):.2f}  p99={pct(qe,99):.2f}  "
              f"max={np.nanmax(qe):.2f}")

    print("\n  By subquery size (n_triples):")
    print(f"  {'n':>3} {'count':>6} | "
          f"{'FICE med':>9} {'FICE p99':>10} | "
          f"{'endpoint med':>11} {'endpoint p99':>12}")
    for n_ in sorted(sub["n_triples"].unique()):
        s = sub[sub["n_triples"] == n_]
        f_q = s["fice_q_error"].values
        q_q = s["endpoint_q_error"].values
        print(f"  {n_:>3} {len(s):>6} | "
              f"{np.nanmedian(f_q):>9.2f} {pct(f_q,99):>10.2f} | "
              f"{np.nanmedian(q_q):>11.2f} {pct(q_q,99):>12.2f}")

    # -------------- Systematic bias analysis --------------
    print("\n===== ENDPOINT ESTIMATOR: IS THE ERROR SYSTEMATIC? =====")
    safe_true = np.maximum(sub["true_card"].values.astype(float), 1.0)
    safe_ql = np.maximum(sub["endpoint_est_card"].values.astype(float), 1.0)
    safe_fice = np.maximum(sub["pred_card"].values.astype(float), 1.0)
    logratio_col_ql = np.log10(safe_ql / safe_true)
    logratio_col_fice = np.log10(safe_fice / safe_true)

    print("\n  log10(estimate / true) distribution (tight = systematic bias):")
    for name, lr in [("FICE", logratio_col_fice), ("endpoint", logratio_col_ql)]:
        print(f"    {name}: mean={np.nanmean(lr):+.3f}  "
              f"std={np.nanstd(lr):.3f}  median={np.nanmedian(lr):+.3f}  "
              f"IQR=[{np.nanpercentile(lr,25):+.3f},"
              f"{np.nanpercentile(lr,75):+.3f}]  "
              f"range=[{np.nanmin(lr):+.2f},{np.nanmax(lr):+.2f}]")

    print("\n  By subquery size — log10-ratio std (smaller = more systematic):")
    print(f"  {'n':>3} {'count':>6} | "
          f"{'FICE mean':>10} {'FICE std':>9} | "
          f"{'endpoint mean':>12} {'endpoint std':>11}")
    for n_ in sorted(sub["n_triples"].unique()):
        sel = (sub["n_triples"] == n_).values
        lr_f = logratio_col_fice[sel]
        lr_q = logratio_col_ql[sel]
        print(f"  {n_:>3} {sel.sum():>6} | "
              f"{np.nanmean(lr_f):>+10.3f} {np.nanstd(lr_f):>9.3f} | "
              f"{np.nanmean(lr_q):>+12.3f} {np.nanstd(lr_q):>11.3f}")

    # Rank correlation within each query: does the estimator preserve the
    # ordering of subqueries by cardinality? (Spearman rho of est vs true)
    from scipy.stats import spearmanr
    ql_rhos, fice_rhos = [], []
    for qi_, grp in sub.groupby("query_idx"):
        if len(grp) < 3:
            continue
        t = grp["true_card"].values.astype(float)
        if np.unique(t).size < 2:
            continue
        rho_q, _ = spearmanr(grp["endpoint_est_card"].values, t)
        rho_f, _ = spearmanr(grp["pred_card"].values, t)
        if np.isfinite(rho_q): ql_rhos.append(rho_q)
        if np.isfinite(rho_f): fice_rhos.append(rho_f)

    print(f"\n  Spearman rank correlation (est vs true) within each query")
    print(f"  — measures whether the ordering of subqueries is preserved:")
    if fice_rhos:
        a = np.array(fice_rhos)
        print(f"    FICE:   N={len(a)}  mean={a.mean():.3f}  "
              f"median={np.median(a):.3f}  "
              f"p10={pct(a,10):.3f}  min={a.min():.3f}")
    if ql_rhos:
        a = np.array(ql_rhos)
        print(f"    endpoint: N={len(a)}  mean={a.mean():.3f}  "
              f"median={np.median(a):.3f}  "
              f"p10={pct(a,10):.3f}  min={a.min():.3f}")

    print("\n===== ROOT (FULL) QUERY Q-ERROR =====")
    root = plan["pred_root_card"].clip(lower=1) / plan["true_root_card"].clip(lower=1)
    root = np.maximum(root, 1.0 / root).values
    print(f"  N={np.isfinite(root).sum()}  "
          f"median={np.nanmedian(root):.2f}  mean={np.nanmean(root):.2f}  "
          f"p90={pct(root,90):.2f}  p99={pct(root,99):.2f}")

    print("\n===== C-OUT Q-ERROR (planner view vs actual plan cost) =====")
    cqe = plan["cout_q_error"].values
    print(f"  median={np.nanmedian(cqe):.2f}  mean={np.nanmean(cqe):.2f}  "
          f"p90={pct(cqe,90):.2f}  p99={pct(cqe,99):.2f}  "
          f"max={np.nanmax(cqe):.2f}")

    # endpoint native planner comparison
    print("\n===== ENDPOINT NATIVE PLAN (its own planner) =====")
    ql_ok = (plan["endpoint_n_ops"] > 0) & np.isfinite(plan["endpoint_cout_true"])
    if ql_ok.any():
        p = plan[ql_ok]
        qe_ql = np.maximum(
            p["endpoint_cout_pred"].clip(lower=1) / p["endpoint_cout_true"].clip(lower=1),
            p["endpoint_cout_true"].clip(lower=1) / p["endpoint_cout_pred"].clip(lower=1)).values
        print(f"  N={len(p)}  endpoint planner's c-out q-error:")
        print(f"    median={np.nanmedian(qe_ql):.2f}  mean={np.nanmean(qe_ql):.2f}  "
              f"p90={pct(qe_ql,90):.2f}  p99={pct(qe_ql,99):.2f}  "
              f"max={np.nanmax(qe_ql):.2f}")

        print("\n  Actual c-out (true intermediate sizes) — FICE-DP plan vs endpoint plan:")
        fice_true = p["cout_true_same_plan"].clip(lower=1)
        ql_true = p["endpoint_cout_true"].clip(lower=1)
        ratio = (fice_true / ql_true).values
        print(f"    FICE/endpoint ratio  median={np.nanmedian(ratio):.2f}  "
              f"mean={np.nanmean(ratio):.2f}  p90={pct(ratio,90):.2f}  "
              f"p99={pct(ratio,99):.2f}  max={np.nanmax(ratio):.2f}")
        print(f"    (ratio>1 = FICE's plan is more expensive; <1 = FICE's plan is better)")
        print(f"    FICE-DP plan better on "
              f"{int(np.nansum(ratio < 1))}/{len(p)} queries")

    # ---- Combo comparison (DP vs greedy, c-out vs UCB) ----
    print("\n===== ALL PLANNER/COST COMBOS vs endpoint (true c-out ratio) =====")
    print(f"  {'combo':<14} {'N':>4} {'median':>8} {'mean':>10} "
          f"{'p90':>8} {'p99':>10} {'max':>10} {'wins':>6}")
    p = plan[ql_ok] if ql_ok.any() else plan
    ql_true_vec = p["endpoint_cout_true"].clip(lower=1)
    for col in combo_cols:
        cn = f"{col}_cout_true"
        if cn not in p.columns:
            continue
        ftrue = p[cn].clip(lower=1)
        r = (ftrue / ql_true_vec).values
        print(f"  {col:<14} {len(p):>4} "
              f"{np.nanmedian(r):>8.2f} {np.nanmean(r):>10.2f} "
              f"{pct(r,90):>8.2f} {pct(r,99):>10.2f} "
              f"{np.nanmax(r):>10.2f} "
              f"{int(np.nansum(r < 1)):>3}/{len(p):<2}")

    # ---- "FICE-DP loses" diagnostic: dominant subquery error per query ----
    print("\n===== WHERE FICE-DP LOSES TO ENDPOINT (top 10 by ratio) =====")
    if ql_ok.any():
        # Build per-query subquery dicts for fast lookup.
        sub_by_q = {}
        for qi_, grp in sub.groupby("query_idx"):
            sub_by_q[int(qi_)] = {
                int(r.mask): (float(r.pred_card), int(r.true_card),
                              float(r.fice_q_error)
                              if np.isfinite(r.fice_q_error) else float("nan"))
                for r in grp.itertuples()}

        # Load chosen plans.
        plans_by_q = {}
        with open(plans_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                plans_by_q[int(rec["query_idx"])] = rec["plans"]

        cand = p.assign(ratio_col=(p["cout_true_same_plan"].clip(lower=1) /
                                p["endpoint_cout_true"].clip(lower=1)))
        worse = cand[cand["ratio_col"] > 1].sort_values("ratio_col", ascending=False)
        worse = worse.head(10)
        if worse.empty:
            print("  (none — FICE-DP wins on every query)")
        else:
            print(f"  {'qi':>4} {'n':>3} {'fice/ql':>8} "
                  f"{'fice_cout':>11} {'ql_cout':>10} | "
                  f"{'worst-FICE-step (mask|n|pred|true|qerr)':<55}")
            for r in worse.itertuples():
                qi_ = int(r.query_idx)
                trees = plans_by_q.get(qi_, {}).get("dp_cout", {})
                plan_d = trees.get("plan", {})
                # Walk every internal node in the chosen plan and find the
                # subquery whose true_card * (multiplicative miss) contributes
                # most to plan blow-up.
                worst = None  # (qerr, mask, n, pred, true)
                for mask_str, lr in plan_d.items():
                    if lr is None:
                        continue
                    m = int(mask_str)
                    info = sub_by_q.get(qi_, {}).get(m)
                    if info is None:
                        continue
                    pc, tc, qerr = info
                    if not np.isfinite(qerr):
                        continue
                    if worst is None or qerr > worst[0]:
                        worst = (qerr, m, bin(m).count("1"), pc, tc)
                wstr = (f"m={worst[1]} n={worst[2]} pred={worst[3]:.1f} "
                        f"true={worst[4]} qerr={worst[0]:.1f}"
                        if worst else "(no info)")
                print(f"  {qi_:>4} {int(r.n_triples):>3} {r.ratio_col:>8.2f} "
                      f"{r.cout_true_same_plan:>11.0f} "
                      f"{r.endpoint_cout_true:>10.0f} | {wstr}")

    print(f"\nWrote: {sub_tsv}\nWrote: {plan_tsv}\nWrote: {plans_jsonl}")


if __name__ == "__main__":
    main()
