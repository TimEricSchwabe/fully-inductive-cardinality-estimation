"""
LSS LOCO experiment helpers.

Handles FICE→LSS conversion, mega-graph construction, ProNE embedding
generation, fold training and evaluation, and results export.
"""
import sys
import os
import json
import csv
import pickle
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LSS_DIR = Path(__file__).resolve().parent
LSS_CODE_DIR = LSS_DIR / "LSS"
PRONE_DIR = LSS_DIR / "ProNE"
# DATA_DIR = LSS_DIR / "data"
DATA_DIR = Path("/storage/nobackup/lukasketzer/fice/lss/data")

MIN_BUCKET_SIZE = 5  # skip query size buckets smaller than this

for _p in [str(LSS_CODE_DIR), str(PRONE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "benchmarks" / "experiments_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_experiment_datasets(exp_cfg, all_dataset_names):
    """Return (train_datasets, left_out, eval_datasets) as lists of names."""
    left_out = exp_cfg["left_out"]
    if isinstance(left_out, str):
        left_out = [left_out]

    eval_ds = exp_cfg.get("eval", left_out)
    if isinstance(eval_ds, str):
        eval_ds = [eval_ds]

    pool = exp_cfg.get("pool", "*")
    if pool == "*":
        train_ds = [ds for ds in all_dataset_names if ds not in left_out]
    else:
        if isinstance(pool, str):
            pool = [pool]
        train_ds = [ds for ds in pool if ds not in left_out]

    return train_ds, left_out, eval_ds


# ── FICE raw file helpers ─────────────────────────────────────────────────────

def _raw_file(dir_or_file):
    p = Path(dir_or_file)
    if p.is_file():
        return p
    files = [f for f in p.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No raw file in {p}")
    return files[0]


def load_fice_queries(query_path, max_queries=None):
    path = str(query_path).split("?")[0]
    with open(path) as f:
        data = json.load(f)
    # Single-triple queries decompose into single-edge subgraphs.  CardNet's
    # forward calls edge_index.squeeze() to strip the DataLoader batch dim,
    # which collapses shape [1, 2, 1] → [2] (1-D), causing a PyG error.
    # Filter them out before applying max_queries so the config limit is still
    # met by valid (multi-edge) queries.
    data = [q for q in data if len(q.get("triples", [])) >= 2]
    if max_queries:
        data = data[:max_queries]
    return data


# ── Per-dataset .nt parsing ───────────────────────────────────────────────────

def parse_nt_graph(graph_path, rdf_type_uri):
    """
    Parse a .nt RDF file.

    Returns:
      entity_to_id      {uri: int}  ordered by first appearance
      rel_to_id         {uri: int}  ordered by first appearance (excl. type triples)
      entity_type       {entity_uri: type_uri}
      type_to_local_id  {type_uri: int}  compact 0-based
      n_types           int
      n_rels            int
    """
    entity_to_id = {}
    rel_to_id = {}
    entity_type = {}
    seen_types = []
    use_types = rdf_type_uri != "0"

    with open(graph_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip(" .").split()
            if len(parts) < 3:
                continue
            s, p = parts[0], parts[1]
            o = " ".join(parts[2:]).rstrip(" .")

            if s not in entity_to_id:
                entity_to_id[s] = len(entity_to_id)

            if use_types and p == rdf_type_uri:
                if o not in seen_types:
                    seen_types.append(o)
                entity_type[s] = o
                # o is a type URI, not an entity
            else:
                if p not in rel_to_id:
                    rel_to_id[p] = len(rel_to_id)
                if o not in entity_to_id:
                    entity_to_id[o] = len(entity_to_id)

    if use_types:
        type_to_local_id = {t: i for i, t in enumerate(seen_types)}
        type_to_local_id["__unk__"] = len(type_to_local_id)  # for entities without a type
    else:
        type_to_local_id = {"__dummy__": 0}

    return (entity_to_id, rel_to_id, entity_type,
            type_to_local_id, len(type_to_local_id), len(rel_to_id))


# ── G-Care graph file writer ──────────────────────────────────────────────────

def write_gcare_graph(out_path, nt_path, rdf_type_uri,
                      entity_to_id, rel_to_id, entity_type, type_to_local_id,
                      entity_id_offset=0, type_id_offset=0, rel_id_offset=0):
    """
    Write a G-Care DataGraph .txt with optional global ID offsets.
    Each node gets one label: its global entity-type ID.
    Each edge gets one label: its global relation-type ID.
    """
    use_types = rdf_type_uri != "0"
    unk_local = type_to_local_id.get("__unk__", type_to_local_id.get("__dummy__", 0))

    written_nodes = {}   # uri → global_entity_id
    edges = []

    with open(nt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip(" .").split()
            if len(parts) < 3:
                continue
            s, p = parts[0], parts[1]
            o = " ".join(parts[2:]).rstrip(" .")

            if use_types and p == rdf_type_uri:
                if s in entity_to_id and s not in written_nodes:
                    written_nodes[s] = entity_to_id[s] + entity_id_offset
                continue

            if o not in entity_to_id:
                continue  # literal value, skip

            for uri in (s, o):
                if uri in entity_to_id and uri not in written_nodes:
                    written_nodes[uri] = entity_to_id[uri] + entity_id_offset

            src_id = entity_to_id[s] + entity_id_offset
            dst_id = entity_to_id[o] + entity_id_offset
            rel_id = rel_to_id[p] + rel_id_offset
            edges.append((src_id, dst_id, rel_id))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for uri, global_eid in sorted(written_nodes.items(), key=lambda x: x[1]):
            if use_types:
                type_uri = entity_type.get(uri)
                local_type = type_to_local_id.get(type_uri, unk_local)
            else:
                local_type = 0
            global_type = local_type + type_id_offset
            f.write(f"v {global_eid} {global_type}\n")
        for src, dst, rel in edges:
            f.write(f"e {src} {dst} {rel}\n")


# ── Query conversion ──────────────────────────────────────────────────────────

def _type_label(uri, entity_type, type_to_local_id, unk_local, type_id_offset):
    """Global type label for a URI, or -1 for variables."""
    if uri.startswith("?"):
        return -1
    type_uri = entity_type.get(uri)
    local = type_to_local_id.get(type_uri, unk_local)
    return local + type_id_offset


def _rel_id(p, rel_to_id, rel_id_offset):
    """Global relation ID for a predicate URI, or None if not found."""
    rid = rel_to_id.get(p)
    if rid is None:
        rid = rel_to_id.get(p.strip("<>"))
    return None if rid is None else rid + rel_id_offset


def convert_star_query(query, entity_type, type_to_local_id, unk_local,
                       rel_to_id, type_id_offset, rel_id_offset):
    """Convert a FICE star query to (gcare_lines, cardinality) or None."""
    triples = query["triples"]
    cardinality = query["y"]
    if not triples:
        return None

    anchor = triples[0][0]
    anchor_label = _type_label(anchor, entity_type, type_to_local_id, unk_local, type_id_offset)

    lines = [f"v 0 {anchor_label}"]
    for i, triple in enumerate(triples):
        s, p, o = triple[0], triple[1], triple[2]
        leaf_label = _type_label(o, entity_type, type_to_local_id, unk_local, type_id_offset)
        lines.append(f"v {i + 1} {leaf_label}")
        global_rid = _rel_id(p, rel_to_id, rel_id_offset)
        if global_rid is None:
            return None
        lines.append(f"e 0 {i + 1} {global_rid}")

    return lines, cardinality


def convert_path_query(query, entity_type, type_to_local_id, unk_local,
                       rel_to_id, type_id_offset, rel_id_offset):
    """Convert a FICE path query to (gcare_lines, cardinality) or None."""
    triples = query["triples"]
    cardinality = query["y"]
    if not triples:
        return None

    node_ids = {}
    node_lines = []
    edge_lines = []

    def get_node(token):
        if token not in node_ids:
            nid = len(node_ids)
            node_ids[token] = nid
            label = _type_label(token, entity_type, type_to_local_id, unk_local, type_id_offset)
            node_lines.append(f"v {nid} {label}")
        return node_ids[token]

    for triple in triples:
        s, p, o = triple[0], triple[1], triple[2]
        s_id = get_node(s)
        o_id = get_node(o)
        global_rid = _rel_id(p, rel_to_id, rel_id_offset)
        if global_rid is None:
            return None
        edge_lines.append(f"e {s_id} {o_id} {global_rid}")

    return node_lines + edge_lines, cardinality


# ── Per-dataset conversion ────────────────────────────────────────────────────

def convert_dataset_to_lss(dataset_cfg, query_cfgs, project_root, per_ds_dir):
    """
    Convert one FICE dataset to per-dataset LSS files.

    Writes:
      per_ds_dir/{name}/graph/{name}.txt
      per_ds_dir/{name}/queryset_homo/{qt}_{size}/q_{i}.txt
      per_ds_dir/{name}/true_homo/{qt}_{size}/q_{i}.txt

    Returns stats dict with entity/relation/type counts and query sizes.
    """
    name = dataset_cfg["name"]
    rdf_type_uri = dataset_cfg["rdf_type_uri"]
    graph_raw = _raw_file(project_root / dataset_cfg["graph_path"])

    print(f"  Parsing graph for {name}...")
    entity_to_id, rel_to_id, entity_type, type_to_local_id, n_types, n_rels = \
        parse_nt_graph(graph_raw, rdf_type_uri)
    print(f"    {len(entity_to_id)} entities, {n_rels} relations, {n_types} types")

    graph_out = per_ds_dir / name / "graph" / f"{name}.txt"
    write_gcare_graph(graph_out, graph_raw, rdf_type_uri,
                      entity_to_id, rel_to_id, entity_type, type_to_local_id)

    unk_local = type_to_local_id.get("__unk__", type_to_local_id.get("__dummy__", 0))
    query_sizes = {}  # {query_type: [sizes]}

    bundles_dir = per_ds_dir / name / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    for qcfg in query_cfgs:
        qt = qcfg["query_type"]
        max_q = qcfg.get("max_train_queries")
        query_raw = _raw_file(project_root / qcfg["path"])
        queries = load_fice_queries(query_raw, max_q)
        print(f"    Converting {len(queries)} {qt} queries...")

        converter = convert_star_query if qt == "star" else convert_path_query
        by_size = defaultdict(list)
        skipped = 0
        for q in queries:
            result = converter(q, entity_type, type_to_local_id, unk_local,
                               rel_to_id, 0, 0)
            if result is None:
                skipped += 1
                continue
            lines, card = result
            by_size[len(q["triples"])].append((lines, card))

        if skipped:
            print(f"    Skipped {skipped} queries (unknown predicates)")

        # One pickle per (qtype, size) bucket: [(query_lines, card), ...]
        # Offsets are kept at 0 here; global offsets are applied at load time
        # by the patched QueryDecompose (see LSS/QuerySampler.py).
        sizes_written = []
        for size, items in sorted(by_size.items()):
            if len(items) < MIN_BUCKET_SIZE:
                continue
            sizes_written.append(size)
            bundle_path = bundles_dir / f"{qt}_{size}.pkl"
            with open(bundle_path, "wb") as f:
                pickle.dump(items, f, protocol=4)

        query_sizes[qt] = sizes_written
        print(f"    Sizes written: {sizes_written}")

    return {
        "name": name,
        "n_entities": len(entity_to_id),
        "n_types": n_types,
        "n_rels": n_rels,
        "type_to_local_id": {k: v for k, v in type_to_local_id.items()},
        "rel_to_id": {k: v for k, v in rel_to_id.items()},
        "query_sizes": query_sizes,
    }


# ── Global label mapping ──────────────────────────────────────────────────────

def compute_global_mappings(all_stats, dataset_order):
    """
    Build compact global type/relation ID mappings.

    Type IDs:     [0, sum(n_types) - 1]  — offset-based, compact
    Relation IDs: [0, sum(n_rels) - 1]   — offset-based, compact
    Entity IDs:   per-dataset offsets     — for DataGraph node IDs

    Returns a dict saved to global_label_map.json.
    """
    type_offsets = {}
    rel_offsets = {}
    entity_offsets = {}
    t_acc = r_acc = e_acc = 0

    for name in dataset_order:
        s = all_stats[name]
        type_offsets[name] = t_acc
        rel_offsets[name] = r_acc
        entity_offsets[name] = e_acc
        t_acc += s["n_types"]
        r_acc += s["n_rels"]
        e_acc += s["n_entities"]

    return {
        "type_offsets": type_offsets,
        "rel_offsets": rel_offsets,
        "entity_offsets": entity_offsets,
        "num_global_types": t_acc,
        "num_global_rels": r_acc,
        "num_global_entities": e_acc,
        "dataset_order": dataset_order,
    }


# ── Mega-graph construction ───────────────────────────────────────────────────

def _append_gcare_graph(out_f, src_path, entity_id_offset, type_id_offset, rel_id_offset):
    """Append a (possibly already-offset-applied) DataGraph file, applying additional offsets."""
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("v"):
                parts = line.split()
                eid = int(parts[1]) + entity_id_offset
                type_id = int(parts[2]) + type_id_offset
                out_f.write(f"v {eid} {type_id}\n")
            elif line.startswith("e"):
                parts = line.split()
                src = int(parts[1]) + entity_id_offset
                dst = int(parts[2]) + entity_id_offset
                rel = int(parts[3]) + rel_id_offset
                out_f.write(f"e {src} {dst} {rel}\n")


def _write_bundle_manifest(manifest_path, per_ds_dir, datasets, global_map, all_stats):
    """
    Write a small JSON manifest listing which per-dataset pickled bundles to
    load and what type/relation offsets to apply at load time.

    Consumed by LSS/QuerySampler.py::QueryDecompose when it sees the manifest.
    """
    entries = []
    for ds in datasets:
        entries.append({
            "name": ds,
            "bundle_dir": str(Path(per_ds_dir) / ds / "bundles"),
            "type_offset": global_map["type_offsets"][ds],
            "rel_offset":  global_map["rel_offsets"][ds],
            "query_sizes": all_stats[ds]["query_sizes"],
        })
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"datasets": entries}, f)


def build_mega_graph_files(exp_id, train_datasets, eval_datasets,
                           per_ds_dir, mega_dir, global_map, all_stats):
    """
    Build mega-graph files for one LOCO fold.

    Produces:
      - mega_dir/dataset/train_{exp_id}/train_{exp_id}.txt        (DataGraph)
      - mega_dir/queryset_homo/train_{exp_id}.bundles.json        (manifest)
      - mega_dir/queryset_homo/eval_{exp_id}.bundles.json         (manifest)

    Query files are NOT duplicated per fold — the patched QueryDecompose
    loads from the per-dataset bundles and applies global offsets in memory.
    """
    mega_dir = Path(mega_dir)
    per_ds_dir = Path(per_ds_dir)

    train_name = f"train_{exp_id}"
    eval_name = f"eval_{exp_id}"

    # ── Mega DataGraph (training datasets only) ───────────────────────────────
    graph_out = mega_dir / "dataset" / train_name / f"{train_name}.txt"
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale pickle so DataGraph re-parses the new .txt file
    pkl_path = graph_out.with_suffix(".pkl")
    if pkl_path.exists():
        pkl_path.unlink()
    print(f"    Writing mega DataGraph ({len(train_datasets)} datasets)...")
    with open(graph_out, "w") as f:
        for ds in train_datasets:
            src = per_ds_dir / ds / "graph" / f"{ds}.txt"
            if not src.exists():
                raise FileNotFoundError(f"DataGraph missing: {src}")
            e_off = global_map["entity_offsets"][ds]
            t_off = global_map["type_offsets"][ds]
            r_off = global_map["rel_offsets"][ds]
            _append_gcare_graph(f, src, e_off, t_off, r_off)

        # Add one dummy node per global type ID so DataGraph.node_label_card
        # covers 0..num_global_types-1 even when a held-out eval dataset contains
        # node labels unseen in train datasets.
        num_global_types = global_map["num_global_types"]
        dummy_type_base = global_map["num_global_entities"]
        for type_id in range(num_global_types):
            f.write(f"v {dummy_type_base + type_id} {type_id}\n")

        # Dummy node pair with one edge per global relation ID.
        # LSS indexes edge features by raw label value, not position, so edge_attr
        # must be at least (max_global_rel_id + 1) wide. Adding these dummy edges
        # ensures DataGraph.edge_label_card covers 0..num_global_rels-1; the
        # held-out dataset's unseen relations get frequency 0 (correct semantics).
        num_global_rels = global_map["num_global_rels"]
        dummy_v0 = dummy_type_base + num_global_types
        dummy_v1 = dummy_v0 + 1
        f.write(f"v {dummy_v0} 0\n")
        f.write(f"v {dummy_v1} 0\n")
        for rel_id in range(num_global_rels):
            f.write(f"e {dummy_v0} {dummy_v1} {rel_id}\n")

    # ── Bundle manifests (no per-query file duplication) ──────────────────────
    qs_dir = mega_dir / "queryset_homo"
    print(f"    Writing train manifest ({len(train_datasets)} datasets)...")
    _write_bundle_manifest(
        qs_dir / f"{train_name}.bundles.json",
        per_ds_dir, train_datasets, global_map, all_stats,
    )
    print(f"    Writing eval manifest ({len(eval_datasets)} datasets)...")
    _write_bundle_manifest(
        qs_dir / f"{eval_name}.bundles.json",
        per_ds_dir, eval_datasets, global_map, all_stats,
    )


# ── ProNE embeddings ──────────────────────────────────────────────────────────

def generate_prone_embeddings(mega_graph_path, num_global_types, emb_out_path,
                               prone_dim=128, prone_step=10):
    """
    Generate ProNE embeddings for the global entity-type space.

    Builds a type co-occurrence graph (nodes = global type IDs), runs ProNE,
    saves result as numpy array of shape (num_global_types, prone_dim).
    """
    from proNE import ProNE  # from PRONE_DIR

    mega_graph_path = Path(mega_graph_path)
    emb_out_path = Path(emb_out_path)
    emb_out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"    Building type co-occurrence graph ({num_global_types} types)...")
    # type_adj[t1][t2] = 1 if any entity of type t1 is connected to entity of type t2
    type_edges = set()
    node_types = {}  # global_entity_id → global_type_id

    with open(mega_graph_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v"):
                parts = line.split()
                node_types[int(parts[1])] = int(parts[2])

    with open(mega_graph_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("e"):
                parts = line.split()
                src, dst = int(parts[1]), int(parts[2])
                t_src = node_types.get(src, 0)
                t_dst = node_types.get(dst, 0)
                if t_src != t_dst:
                    type_edges.add((t_src, t_dst))
                    type_edges.add((t_dst, t_src))
                else:
                    type_edges.add((t_src, t_src))

    # Add self-loops for isolated type nodes so ProNE can embed them.
    types_with_edges = set()
    for a, b in type_edges:
        types_with_edges.add(a)
        types_with_edges.add(b)
    for t in set(range(num_global_types)) - types_with_edges:
        type_edges.add((t, t))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".edgelist", delete=False) as tf:
        edge_file = tf.name
        for a, b in type_edges:
            tf.write(f"{a} {b}\n")

    emb1_path = str(emb_out_path).replace(".emb.npy", "_sparse.emb")
    emb2_path = str(emb_out_path).replace(".emb.npy", "_enhanced.emb")

    print(f"    Running ProNE (dim={prone_dim}, step={prone_step})...")
    model = ProNE(edge_file, emb1_path, emb2_path, prone_dim)
    features = model.pre_factorization(model.matrix0, model.matrix0)
    embeddings = model.chebyshev_gaussian(model.matrix0, features, prone_step)

    os.unlink(edge_file)

    # ProNE outputs shape (actual_nodes, dim) where actual_nodes = max_node_id + 1
    # Pad or trim to (num_global_types, dim)
    emb_array = np.zeros((num_global_types, prone_dim), dtype=np.float32)
    n = min(embeddings.shape[0], num_global_types)
    emb_array[:n] = embeddings[:n]

    np.save(emb_out_path, emb_array)
    print(f"    Saved embeddings: {emb_out_path} shape={emb_array.shape}")

    # Clean up text embedding files
    for p in [emb1_path, emb2_path]:
        if os.path.exists(p):
            os.unlink(p)


# ── Fold runner ───────────────────────────────────────────────────────────────

def run_loco_fold(exp_id, _train_datasets, eval_datasets,
                  mega_dir, _all_stats, _global_map, fold_results_dir,
                  epochs_override=None, prone_time_s=0.0):
    import subprocess
    import json

    mega_dir = Path(mega_dir)
    fold_results_dir = Path(fold_results_dir)
    fold_results_dir.mkdir(parents=True, exist_ok=True)

    train_name = f"train_{exp_id}"
    epochs = epochs_override if epochs_override is not None else 80
    model_save_dir = fold_results_dir / "checkpoint"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # ── Train via LSS native entry point ──────────────────────────────────────
    cmd_train = [
        sys.executable, "active_train.py",
        "--mode", "pretrain",
        "--dataset", train_name,
        "--embed_type", "prone",
        "--full_data_dir", str(mega_dir),
        "--model_save_dir", str(model_save_dir),
        "--epochs", str(epochs),
        "--matching", "homo",
    ]
    print(f"  Training CardNet via subprocess ({epochs} epochs)...")
    subprocess.run(cmd_train, cwd=str(LSS_CODE_DIR), check=True)

    # ── Eval via eval_loco.py ─────────────────────────────────────────────────
    ds_name = eval_datasets[0] if len(eval_datasets) == 1 else "eval"
    out_json = fold_results_dir / "predictions_raw.json"
    cmd_eval = [
        sys.executable, str(LSS_DIR / "eval_loco.py"),
        "--exp_id",           exp_id,
        "--mega_dir",         str(mega_dir),
        "--fold_results_dir", str(fold_results_dir),
        "--eval_ds",          ds_name,
        "--epochs",           str(epochs),
        "--prone_time_s",     str(prone_time_s),
        "--out",              str(out_json),
    ]
    print(f"  Evaluating via eval_loco.py...")
    subprocess.run(cmd_eval, cwd=str(LSS_DIR), check=True)

    # ── Collect results ───────────────────────────────────────────────────────
    with open(out_json) as f:
        predictions = json.load(f)

    save_predictions_csv(predictions, fold_results_dir)
    print(f"  Saved {len(predictions)} predictions → {fold_results_dir / 'predictions.csv'}")


# ── Results export ────────────────────────────────────────────────────────────

def save_predictions_csv(predictions, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "predictions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_type",
                "dataset",
                "epoch",
                "ground_truth",
                "prediction",
                "query_size",
                "prone_time_s",
                "data_load_time_s",
                "eval_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(errors):
    print("\n" + "=" * 60)
    if errors:
        print(f"Done with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")
