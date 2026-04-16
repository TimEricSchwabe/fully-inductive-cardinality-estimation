"""
LSS LOCO experiment helpers.

Handles FICE→LSS conversion, mega-graph construction, ProNE embedding
generation, fold training and evaluation, and results export.
"""
import sys
import os
import json
import csv
import math
import argparse
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LSS_DIR = Path(__file__).resolve().parent
LSS_CODE_DIR = LSS_DIR / "LSS"
PRONE_DIR = LSS_DIR / "ProNE"
DATA_DIR = LSS_DIR / "data"

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

        sizes_written = []
        for size, items in sorted(by_size.items()):
            if len(items) < MIN_BUCKET_SIZE:
                continue
            sizes_written.append(size)
            qs_dir = per_ds_dir / name / "queryset_homo" / f"{qt}_{size}"
            tc_dir = per_ds_dir / name / "true_homo" / f"{qt}_{size}"
            qs_dir.mkdir(parents=True, exist_ok=True)
            tc_dir.mkdir(parents=True, exist_ok=True)
            for i, (lines, card) in enumerate(items):
                qfile = qs_dir / f"q_{i}.txt"
                cfile = tc_dir / f"q_{i}.txt"
                with open(qfile, "w") as f:
                    f.write("t # 1\n")
                    for ln in lines:
                        f.write(ln + "\n")
                with open(cfile, "w") as f:
                    f.write(str(card) + "\n")

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


def _remap_query_files(src_dataset_dir, dest_qs_dir, dest_tc_dir,
                       entity_id_offset, type_id_offset, rel_id_offset,
                       query_sizes):
    """Copy query + cardinality files, remapping IDs with given offsets."""
    src_qs = Path(src_dataset_dir) / "queryset_homo"
    src_tc = Path(src_dataset_dir) / "true_homo"

    for qt, sizes in query_sizes.items():
        for size in sizes:
            bucket = f"{qt}_{size}"
            sq = src_qs / bucket
            st = src_tc / bucket
            if not sq.exists():
                continue

            dq = Path(dest_qs_dir) / bucket
            dt = Path(dest_tc_dir) / bucket
            dq.mkdir(parents=True, exist_ok=True)
            dt.mkdir(parents=True, exist_ok=True)

            # keep a running counter for destination query indices
            existing = len(list(dq.iterdir()))

            for qfile in sorted(sq.iterdir()):
                idx = existing
                existing += 1
                out_q = dq / f"q_{idx}.txt"
                out_c = dt / f"q_{idx}.txt"

                # Remap query file
                with open(qfile) as fin, open(out_q, "w") as fout:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("t"):
                            fout.write(line + "\n")
                        elif line.startswith("v"):
                            parts = line.split()
                            nid = parts[1]  # local node id, keep as-is (query-local)
                            label = int(parts[2])
                            if label != -1:
                                label += type_id_offset
                            fout.write(f"v {nid} {label}\n")
                        elif line.startswith("e"):
                            parts = line.split()
                            src, dst = parts[1], parts[2]
                            rel = int(parts[3]) + rel_id_offset
                            fout.write(f"e {src} {dst} {rel}\n")

                # Copy cardinality file unchanged
                card_src = st / qfile.name
                shutil.copy2(card_src, out_c)


def build_mega_graph_files(exp_id, all_dataset_names, train_datasets, eval_datasets,
                           per_ds_dir, mega_dir, global_map, all_stats):
    """
    Build mega-graph files for one LOCO fold.

    DataGraph:      all 8 datasets (for consistent feature space)
    Train queries:  N-1 datasets only
    Eval queries:   held-out dataset(s) only
    """
    mega_dir = Path(mega_dir)
    per_ds_dir = Path(per_ds_dir)

    train_name = f"train_{exp_id}"
    eval_name = f"eval_{exp_id}"

    # ── Mega DataGraph (ALL datasets) ─────────────────────────────────────────
    graph_out = mega_dir / "dataset" / train_name / f"{train_name}.txt"
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Writing mega DataGraph ({len(all_dataset_names)} datasets)...")
    with open(graph_out, "w") as f:
        for ds in all_dataset_names:
            src = per_ds_dir / ds / "graph" / f"{ds}.txt"
            if not src.exists():
                raise FileNotFoundError(f"DataGraph missing: {src}")
            e_off = global_map["entity_offsets"][ds]
            t_off = global_map["type_offsets"][ds]
            r_off = global_map["rel_offsets"][ds]
            _append_gcare_graph(f, src, e_off, t_off, r_off)

    # ── Training queries (N-1 datasets) ───────────────────────────────────────
    train_qs_dir = mega_dir / "queryset_homo" / train_name
    train_tc_dir = mega_dir / "true_homo" / train_name
    print(f"    Writing training queries from {len(train_datasets)} datasets...")
    for ds in train_datasets:
        t_off = global_map["type_offsets"][ds]
        r_off = global_map["rel_offsets"][ds]
        e_off = global_map["entity_offsets"][ds]
        _remap_query_files(
            per_ds_dir / ds, train_qs_dir, train_tc_dir,
            e_off, t_off, r_off, all_stats[ds]["query_sizes"]
        )

    # ── Eval queries (held-out datasets) ──────────────────────────────────────
    eval_qs_dir = mega_dir / "queryset_homo" / eval_name
    eval_tc_dir = mega_dir / "true_homo" / eval_name
    print(f"    Writing eval queries from {len(eval_datasets)} datasets...")
    for ds in eval_datasets:
        t_off = global_map["type_offsets"][ds]
        r_off = global_map["rel_offsets"][ds]
        e_off = global_map["entity_offsets"][ds]
        _remap_query_files(
            per_ds_dir / ds, eval_qs_dir, eval_tc_dir,
            e_off, t_off, r_off, all_stats[ds]["query_sizes"]
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


# ── LSS args builder ──────────────────────────────────────────────────────────

def build_lss_args(train_name, mega_dir, cuda=False):
    """Build an argparse.Namespace matching active_train.py's defaults."""
    mega_dir = Path(mega_dir)
    args = argparse.Namespace(
        # Model
        num_layers=3,
        model_type="NNGINConcat",
        embed_type="prone",
        edge_embed_type="freq",
        num_g_hid=128,
        num_e_hid=32,
        out_g_ch=128,
        num_expert=64,
        num_att_hid=64,
        num_mlp_hid=128,
        pool_type="att",
        dropout=0.2,
        # Training
        cumulative=False,
        num_fold=5,
        epochs=80,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=5e-4,
        decay_factor=0.1,
        decay_patience=50,
        weight_exp=1.0,
        num_workers=4,
        # Multi-task
        multi_task=True,
        max_classes=10,
        coeff=0.5,
        # Active learning (disabled)
        uncertainty="consist",
        biased_sample=True,
        active_iters=0,
        budget=50,
        active_epochs=50,
        ensemble_num=5,
        # Paths
        dataset=train_name,
        full_data_dir=str(mega_dir),
        save_res_dir=str(mega_dir / "result"),
        model_file=f"{train_name}.pth",
        model_save_dir=str(mega_dir / "models"),
        # Other
        matching="homo",
        k=3,
        verbose=False,
        mode="pretrain",
    )
    # Derived dirs (mirrors active_train.py logic)
    args.queryset_homo_dir = str(mega_dir / "queryset_homo")
    args.queryset_iso_dir = str(mega_dir / "queryset_iso")
    args.true_homo_dir = str(mega_dir / "true_homo")
    args.true_iso_dir = str(mega_dir / "true_iso")
    args.data_dir = str(mega_dir / "dataset")
    args.prone_feat_dir = str(mega_dir / "prone")
    args.n2v_feat_dir = str(mega_dir / "n2v")
    args.nrp_feat_dir = str(mega_dir / "nrp")
    args.embed_feat_dir = args.prone_feat_dir
    args.queryset_dir = args.queryset_homo_dir
    args.true_card_dir = args.true_homo_dir
    # Hardware
    import torch
    args.cuda = cuda and torch.cuda.is_available()
    args.device = __import__("torch").device("cuda" if args.cuda else "cpu")
    return args


# ── QuerysetEval ──────────────────────────────────────────────────────────────

class QuerysetEval:
    """
    Converts eval queries to tensors using an existing Queryset's label stats.
    Avoids re-loading the DataGraph or embedding file.
    """

    def __init__(self, train_qs, all_subsets):
        self._qs = train_qs
        # Use sorted key order so flat_list and metadata are always aligned.
        self._ordered_keys = sorted(all_subsets.keys())
        self.converted = self._convert(all_subsets)

    def _convert(self, all_subsets):
        converted = {}
        for key in self._ordered_keys:
            pairs = all_subsets[key]
            converted[key] = []
            for graphs, card in pairs:
                decomp_x, decomp_edge_index, decomp_edge_attr, _ = \
                    self._qs._get_decomposed_graph_data(graphs)
                converted[key].append((decomp_x, decomp_edge_index, decomp_edge_attr, card))
        return converted

    def to_flat_dataset(self):
        """Return all eval queries as a single QueryDataset (sorted key order)."""
        from Queryset import QueryDataset  # noqa: LSS module
        all_q = []
        for key in self._ordered_keys:
            all_q.extend(self.converted[key])
        return QueryDataset(queries=all_q, num_classes=self._qs.args.max_classes)

    def metadata(self):
        """Return list of (pattern, size) for each eval query (sorted key order)."""
        meta = []
        for (pattern, size) in self._ordered_keys:
            for _ in self.converted[(pattern, size)]:
                meta.append((pattern, size))
        return meta


# ── Fold runner ───────────────────────────────────────────────────────────────

def run_loco_fold(exp_id, train_datasets, eval_datasets,
                  mega_dir, all_stats, global_map, fold_results_dir,
                  epochs_override=None):
    """
    Full LOCO fold pipeline:
      1. Build Queryset for training (mega DataGraph + N-1 train queries)
      2. Build QuerysetEval for evaluation (same DataGraph + held-out queries)
      3. Train CardNet (pretrain, no active iterations)
      4. Evaluate on held-out queries
      5. Save predictions CSV
    """
    import torch
    import torch.optim as optim
    from Queryset import Queryset
    from QuerySampler import QueryDecompose
    import cardnet
    from active import ActiveLearner, _to_datasets

    mega_dir = Path(mega_dir)
    fold_results_dir = Path(fold_results_dir)
    fold_results_dir.mkdir(parents=True, exist_ok=True)

    train_name = f"train_{exp_id}"
    eval_name = f"eval_{exp_id}"

    cuda = torch.cuda.is_available()
    args = build_lss_args(train_name, mega_dir, cuda=cuda)
    if epochs_override is not None:
        args.epochs = epochs_override

    # ── Training Queryset ─────────────────────────────────────────────────────
    print(f"  Decomposing training queries ({train_name})...")
    train_qd = QueryDecompose(
        queryset_dir=args.queryset_dir,
        true_card_dir=args.true_card_dir,
        dataset=train_name,
        k=args.k,
    )
    train_qd.decomose_queries()

    if not train_qd.all_subsets:
        raise RuntimeError(f"No training queries found for {train_name}")

    print(f"  Loading training Queryset...")
    train_qs = Queryset(args=args, all_subsets=train_qd.all_subsets)
    train_qs.print_queryset_info()

    num_node_feat = train_qs.num_node_feat
    num_edge_feat = train_qs.num_edge_feat

    train_datasets_list = _to_datasets(train_qs.all_train_sets, args.max_classes)
    val_datasets_list = _to_datasets(train_qs.val_sets, args.max_classes)

    # ── Eval Queryset ─────────────────────────────────────────────────────────
    print(f"  Decomposing eval queries ({eval_name})...")
    eval_qd = QueryDecompose(
        queryset_dir=args.queryset_dir,
        true_card_dir=args.true_card_dir,
        dataset=eval_name,
        k=args.k,
    )
    eval_qd.decomose_queries()

    if not eval_qd.all_subsets:
        raise RuntimeError(f"No eval queries found for {eval_name}")

    # Filter single-triple queries (size == 1) — they decompose into single-edge
    # subgraphs that CardNet's squeeze() collapses to 1-D, causing a PyG error.
    # Mirrors the load_fice_queries filter applied during training.
    eval_subsets = {k: v for k, v in eval_qd.all_subsets.items() if k[1] >= 2}

    print(f"  Converting eval queries...")
    eval_qs = QuerysetEval(train_qs, eval_subsets)
    eval_dataset = eval_qs.to_flat_dataset()
    eval_meta = eval_qs.metadata()

    # ── Model and optimiser ───────────────────────────────────────────────────
    model = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
    criterion = torch.nn.MSELoss()
    criterion_cla = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"  Training CardNet for {args.epochs} epochs...")
    learner = ActiveLearner(args)
    learner.train(
        model=model,
        criterion=criterion,
        criterion_cal=criterion_cla,
        train_datasets=train_datasets_list,
        val_datasets=val_datasets_list,
        optimizer=optimizer,
        scheduler=scheduler,
        active=False,
    )

    # Save checkpoint
    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                Path(args.model_save_dir) / args.model_file)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"  Evaluating on held-out queries ({eval_name})...")
    model.eval()
    if args.cuda:
        model.to(args.device)

    from torch.utils.data import DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    predictions = []
    query_id = 0
    for decomp_x, decomp_edge_index, decomp_edge_attr, card, label in eval_loader:
        if args.cuda:
            from active.active_util import _to_cuda
            decomp_x = _to_cuda(decomp_x)
            decomp_edge_index = _to_cuda(decomp_edge_index)
            decomp_edge_attr = _to_cuda(decomp_edge_attr)

        with _torch.no_grad():
            output, _ = model(decomp_x, decomp_edge_index, decomp_edge_attr)
            output = output.reshape(-1)

        true_log2 = card.item()
        pred_log2 = output.item()
        true_card = int(round(math.pow(2, true_log2)))
        pred_card = max(1, int(round(math.pow(2, pred_log2))))

        pattern, size = eval_meta[query_id]
        # Determine dataset (from eval_datasets, use first if single)
        ds_name = eval_datasets[0] if len(eval_datasets) == 1 else "eval"
        predictions.append({
            "query_id": query_id,
            "query_type": pattern,
            "dataset": ds_name,
            "epoch": args.epochs,
            "ground_truth": true_card,
            "prediction": pred_card,
        })
        query_id += 1

    save_predictions_csv(predictions, fold_results_dir)
    print(f"  Saved {len(predictions)} predictions → {fold_results_dir / 'predictions.csv'}")
    return predictions


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
