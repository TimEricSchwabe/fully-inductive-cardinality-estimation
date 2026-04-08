import argparse
import json
import random
import shutil
import subprocess
import sys
import transform_graph
import transform_query
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "experiments_config.yaml"
LSS_PATH = Path(__file__).parent / "LSS"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_BASE_PATH = LSS_PATH / "data" / "dataset"
QUERYSET_PATH = LSS_PATH / "data" / "queryset_homo"
TRUE_CARD_PATH = LSS_PATH / "data" / "true_homo"
RESULTS_PATH = Path(__file__).parent / "results"


def _resolve_project_path(path_value):
    """Resolve config paths relative to the FICE project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _to_list(value):
    """Normalize a string or list config value to a list."""
    return value if isinstance(value, list) else [value]


def _get_eval_datasets(experiment):
    """Datasets to evaluate on — must be explicitly set via the 'eval' key.

    Independent of 'left_out': a dataset listed here is NOT excluded from the
    training pool unless it also appears in 'left_out'.
    Returns an empty list when 'eval' is absent or null (no evaluation runs).
    """
    raw = experiment.get("eval")
    if not raw:
        return []
    return _to_list(raw)


def _resolve_pool(experiment, all_dataset_names):
    """Resolve pool: '*' expands to all datasets except left_out."""
    left_out_set = set(_to_list(experiment["left_out"]))
    pool = experiment["pool"]
    if pool == "*":
        return [d for d in all_dataset_names if d not in left_out_set]
    return [d for d in pool if d not in left_out_set]


def _get_train_id(experiment):
    """Return the training ID for this experiment.

    Optional 'train_id' in the config lets multiple experiments share one
    mega-graph and one trained model (e.g. evaluating star vs. path separately
    without rebuilding or retraining). Defaults to the experiment's own 'id'.
    """
    return experiment.get("train_id", experiment["id"])


# ---------------------------------------------------------------------------
# Phase 1: Graph transformation
# ---------------------------------------------------------------------------

def _ensure_graph(ds_name, ds_config, dataset_mappings):
    """Load or transform a single dataset graph to G-Care/LSS format.

    Updates dataset_mappings in place and returns the mapping tuple.
    """
    if ds_name in dataset_mappings:
        return dataset_mappings[ds_name]

    out_path = DATASET_BASE_PATH / ds_name
    mapping_file = out_path / "id_to_id_mapping.json"
    pred_mapping_file = out_path / "id_to_id_mapping_predicate.json"
    type_label_mapping_file = out_path / "id_to_type_label_mapping.json"

    if mapping_file.exists() and pred_mapping_file.exists() and type_label_mapping_file.exists():
        print(f"[{ds_name}] Mappings already exist, loading from JSON...")
        with open(mapping_file) as f:
            id_to_id_mapping = json.load(f)
        with open(pred_mapping_file) as f:
            id_to_id_mapping_predicate = json.load(f)
        with open(type_label_mapping_file) as f:
            type_label_mapping = json.load(f)
    else:
        graph_path = _resolve_project_path(ds_config["graph_path"])
        rdf_type_uri = ds_config["rdf_type_uri"]
        print(f"[{ds_name}] Transforming graph...")
        _, _, id_to_id_mapping, id_to_id_mapping_predicate, type_label_mapping = transform_graph.transform_ttl_to_gcare(
            graph_path=graph_path,
            rdf_type_uri=rdf_type_uri,
            dataset_name=ds_name,
            out_path=out_path,
        )

    mapping = (id_to_id_mapping, id_to_id_mapping_predicate, type_label_mapping)
    dataset_mappings[ds_name] = mapping
    return mapping


# ---------------------------------------------------------------------------
# Phase 2: Query transformation
# ---------------------------------------------------------------------------

def _count_unique_nodes(triples):
    nodes = set()
    for triple in triples:
        s, p, o = triple[0], triple[1], triple[2]
        nodes.add(s)
        nodes.add(o)
    return len(nodes)


def _write_lss_query(query_path, query_idx, triples, id_to_id_mapping, id_to_id_mapping_predicate, rdf_type_uri, type_label_mapping):
    """Write a single query in LSS/G-Care text format."""
    vertices = set()
    vertex_labels = {}

    for tp in triples:
        vertices.add(tp[0])
        vertices.add(tp[2])
        if tp[1] == rdf_type_uri and "?" not in tp[2]:
            vertex_labels.setdefault(tp[0], []).append(tp[2])

    vertex_dict = {}
    for vid, vertex in enumerate(vertices):
        try:
            dvid = id_to_id_mapping[vertex] if "?" not in vertex else -1
        except KeyError:
            dvid = -1
        if vertex in vertex_labels:
            labels = [type_label_mapping.get(v, -1) for v in vertex_labels[vertex]]
        else:
            labels = [-1]
        vertex_dict[vertex] = [vid] + labels + [dvid]

    edge_list = []
    for tp in triples:
        try:
            edge_label = id_to_id_mapping_predicate[tp[1]] if "?" not in tp[1] else -1
        except KeyError:
            edge_label = -1
        edge_list.append([vertex_dict[tp[0]][0], vertex_dict[tp[2]][0], edge_label])

    with open(query_path, "w") as f:
        f.write(f"t # s {query_idx}\n")
        for v, entry in vertex_dict.items():
            label_str = " ".join(str(l) for l in entry[1:-1])
            f.write(f"v {entry[0]} {label_str} {entry[-1]}\n")
        for e in edge_list:
            f.write(f"e {e[0]} {e[1]} {e[2]}\n")


def _ensure_queries(ds_name, ds_config, query_configs, dataset_mappings):
    """Transform query files for a single dataset to LSS format (if not already done)."""
    id_to_id_mapping, id_to_id_mapping_predicate, type_label_mapping = dataset_mappings[ds_name]
    rdf_type_uri = ds_config["rdf_type_uri"]

    for q in query_configs:
        query_type = q["query_type"]
        query_path = _resolve_project_path(q["path"])

        print(f"[{ds_name}/{query_type}] Loading queries from {query_path}...")
        with open(query_path) as f:
            queries = json.load(f)

        for i, query in enumerate(queries):
            triples = query["triples"]
            if len(triples) < 2:
                continue
            cardinality = query["y"]
            num_nodes = _count_unique_nodes(triples)
            subdir = f"{query_type}_{num_nodes}"

            qs_dir = QUERYSET_PATH / ds_name / subdir
            tc_dir = TRUE_CARD_PATH / ds_name / subdir
            qs_dir.mkdir(parents=True, exist_ok=True)
            tc_dir.mkdir(parents=True, exist_ok=True)

            query_file = qs_dir / f"query_{i}.txt"
            card_file = tc_dir / f"query_{i}.txt"

            if query_file.exists() and card_file.exists():
                continue

            _write_lss_query(
                query_path=query_file,
                query_idx=i,
                triples=triples,
                id_to_id_mapping=id_to_id_mapping,
                id_to_id_mapping_predicate=id_to_id_mapping_predicate,
                rdf_type_uri=rdf_type_uri,
                type_label_mapping=type_label_mapping,
            )
            with open(card_file, "w") as f:
                f.write(f"{cardinality}\n")

        print(f"[{ds_name}/{query_type}] Done ({len(queries)} queries).")


# ---------------------------------------------------------------------------
# Phase 3: LOCO mega-graph and queryset construction
# ---------------------------------------------------------------------------

def _parse_gcare_graph(graph_file):
    """Parse a G-Care graph file. Returns (vertices, edges) as raw token lists."""
    vertices = []
    edges = []
    with open(graph_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v"):
                tokens = line.split()
                vid = int(tokens[1])
                labels = [int(t) for t in tokens[2:]]
                vertices.append((vid, labels))
            elif line.startswith("e"):
                tokens = line.split()
                src, dst = int(tokens[1]), int(tokens[2])
                edge_labels = [int(t) for t in tokens[3:]]
                edges.append((src, dst, edge_labels))
    return vertices, edges


def _compute_loco_offsets(train_datasets, dataset_mappings):
    """Compute per-dataset node, predicate, and type-label ID offsets for merging."""
    node_offsets = {}
    pred_offsets = {}
    label_offsets = {}
    cumulative_nodes = 0
    cumulative_preds = 0
    cumulative_labels = 0

    for ds in train_datasets:
        id_to_id_mapping, id_to_id_mapping_predicate, type_label_mapping = dataset_mappings[ds]
        node_offsets[ds] = cumulative_nodes
        pred_offsets[ds] = cumulative_preds
        label_offsets[ds] = cumulative_labels
        cumulative_nodes += len(id_to_id_mapping)
        cumulative_preds += len(id_to_id_mapping_predicate)
        cumulative_labels += len(type_label_mapping)

    return node_offsets, pred_offsets, label_offsets


def _build_mega_graph(train_datasets, dataset_mappings, out_path):
    """Merge N-1 G-Care graphs into one file with offset node/label/predicate IDs."""
    node_offsets, pred_offsets, label_offsets = _compute_loco_offsets(train_datasets, dataset_mappings)
    out_path.mkdir(parents=True, exist_ok=True)
    mega_name = out_path.name
    out_file = out_path / f"{mega_name}.txt"

    with open(out_file, "w") as f:
        f.write("t # 1\n")
        for ds in train_datasets:
            n_off = node_offsets[ds]
            p_off = pred_offsets[ds]
            l_off = label_offsets[ds]
            graph_file = DATASET_BASE_PATH / ds / f"{ds}.txt"
            vertices, edges = _parse_gcare_graph(graph_file)
            for vid, labels in vertices:
                # Label 0 = untyped sentinel, shared across all datasets — never shifted.
                # Real type labels (> 0) are shifted by the cumulative label offset.
                shifted_labels = [l + l_off if l > 0 else 0 for l in labels]
                f.write(f"v {vid + n_off} {' '.join(map(str, shifted_labels))}\n")
            for src, dst, edge_labels in edges:
                shifted_edge_labels = [l + p_off for l in edge_labels]
                f.write(f"e {src + n_off} {dst + n_off} {' '.join(map(str, shifted_edge_labels))}\n")

    print(f"  Mega-graph written to {out_file}")
    return node_offsets, pred_offsets, label_offsets


def _rewrite_query_with_offsets(src_file, dst_file, node_offset, pred_offset, label_offset):
    """Copy a query file, applying label/pred offsets for LOCO training set."""
    with open(src_file) as f:
        lines = f.readlines()

    with open(dst_file, "w") as f:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("v"):
                tokens = stripped.split()
                vid = tokens[1]
                dvid = tokens[-1]
                label_tokens = tokens[2:-1]
                shifted_labels = [
                    # -1 = wildcard (no type constraint), 0 = untyped sentinel — never shifted.
                    # Real type labels (> 0) are shifted by the cumulative label offset.
                    str(int(t) + label_offset) if int(t) > 0 else str(int(t))
                    for t in label_tokens
                ]
                f.write(f"v {vid} {' '.join(shifted_labels)} {dvid}\n")
            elif stripped.startswith("e"):
                tokens = stripped.split()
                src, dst = tokens[1], tokens[2]
                edge_label = int(tokens[3]) if len(tokens) > 3 else -1
                shifted_edge = edge_label + pred_offset if edge_label != -1 else -1
                f.write(f"e {src} {dst} {shifted_edge}\n")
            else:
                f.write(line)


def _is_single_edge(query_file):
    """Return True if a query file has exactly 1 edge (unsupported by LSS)."""
    with open(query_file) as f:
        return sum(1 for l in f if l.startswith("e ")) == 1


def _build_loco_queryset(train_datasets, dataset_mappings, node_offsets, pred_offsets, label_offsets, loco_name, dataset_caps=None):
    """Merge querysets from N-1 datasets into a single training queryset with offsets.

    dataset_caps: optional dict mapping dataset name -> max queries to include.
    """
    if dataset_caps:
        loco_qs_root = QUERYSET_PATH / loco_name
        loco_tc_root = TRUE_CARD_PATH / loco_name
        if loco_qs_root.exists():
            shutil.rmtree(loco_qs_root)
        if loco_tc_root.exists():
            shutil.rmtree(loco_tc_root)

    for ds in train_datasets:
        n_off = node_offsets[ds]
        p_off = pred_offsets[ds]
        l_off = label_offsets[ds]
        ds_qs_root = QUERYSET_PATH / ds
        ds_tc_root = TRUE_CARD_PATH / ds

        if not ds_qs_root.exists():
            print(f"  WARNING: No queryset found for {ds}, skipping.")
            continue

        all_query_files = [
            f for f in ds_qs_root.rglob("query_*.txt")
            if not _is_single_edge(f)
        ]

        cap = dataset_caps.get(ds) if dataset_caps else None
        if cap is not None and len(all_query_files) > cap:
            total = len(all_query_files)
            rng = random.Random(42)
            all_query_files = rng.sample(all_query_files, cap)
            print(f"  [{ds}] Sampled {cap} / {total} queries.")

        for query_file in all_query_files:
            subdir_name = query_file.parent.name
            dst_qs_dir = QUERYSET_PATH / loco_name / subdir_name
            dst_tc_dir = TRUE_CARD_PATH / loco_name / subdir_name
            dst_qs_dir.mkdir(parents=True, exist_ok=True)
            dst_tc_dir.mkdir(parents=True, exist_ok=True)

            dst_query = dst_qs_dir / f"{ds}_{query_file.name}"
            if not dst_query.exists():
                _rewrite_query_with_offsets(query_file, dst_query, n_off, p_off, l_off)

            tc_file = ds_tc_root / subdir_name / query_file.name
            if tc_file.exists():
                dst_card = dst_tc_dir / f"{ds}_{query_file.name}"
                if not dst_card.exists():
                    dst_card.write_text(tc_file.read_text())


def _build_padded_eval_graph(left_out, dataset_mappings, train_datasets):
    """Create a padded copy of the eval data graph that covers all mega-graph label IDs.

    The mega-graph (N-1 training datasets merged) has more unique predicate and type-label
    IDs than any single held-out dataset.  LSS's Queryset builds feature vectors sized by
    len(edge_label_card) / len(node_label_card) of the data graph — so without padding the
    eval graph the feature dimension would be smaller than the trained model expects.

    This function writes a copy of {left_out}.txt with dummy self-loop edges for every
    missing predicate ID and dummy isolated vertices for every missing type-label ID, so
    that Queryset naturally produces feature vectors of the correct (mega-graph) dimension.
    """
    eval_name = f"{left_out}_eval"
    eval_graph_dir = DATASET_BASE_PATH / eval_name
    eval_graph_dir.mkdir(parents=True, exist_ok=True)
    eval_graph_file = eval_graph_dir / f"{eval_name}.txt"

    if eval_graph_file.exists():
        return eval_name, eval_graph_file

    src_graph = DATASET_BASE_PATH / left_out / f"{left_out}.txt"
    vertices, edges = _parse_gcare_graph(src_graph)

    # Feature counts for the eval (held-out) dataset
    _, eval_pred_map, eval_type_map = dataset_mappings[left_out]
    eval_pred_count = len(eval_pred_map)   # pred IDs: 0 .. eval_pred_count-1
    eval_type_count = len(eval_type_map)   # type label IDs: 1 .. eval_type_count

    # Feature counts for the training mega-graph
    mega_pred_count = sum(len(dataset_mappings[ds][1]) for ds in train_datasets)
    mega_type_count = sum(len(dataset_mappings[ds][2]) for ds in train_datasets)

    existing_preds = {l for _, _, el in edges for l in el}
    max_vid = max((v for v, _ in vertices), default=-1)

    with open(eval_graph_file, "w") as f:
        f.write("t # 1\n")
        for vid, labels in vertices:
            f.write(f"v {vid} {' '.join(map(str, labels))}\n")
        for src, dst, edge_labels in edges:
            f.write(f"e {src} {dst} {' '.join(map(str, edge_labels))}\n")

        # Pad type-label IDs: add isolated dummy vertices for each missing label
        dummy_vid = max_vid + 1
        for label_id in range(eval_type_count + 1, mega_type_count + 1):
            f.write(f"v {dummy_vid} {label_id}\n")
            dummy_vid += 1

        # Pad predicate IDs: add dummy self-loops for each missing predicate
        for pred_id in range(eval_pred_count, mega_pred_count):
            if pred_id not in existing_preds:
                f.write(f"e 0 0 {pred_id}\n")

    print(f"[{eval_name}] Padded eval graph written: "
          f"pred {eval_pred_count}→{mega_pred_count}, "
          f"type {eval_type_count}→{mega_type_count}")
    return eval_name, eval_graph_file


def _build_capped_eval_queryset(left_out, max_eval_queries, dataset_mappings, train_datasets, seed=42):
    """Create a padded evaluation queryset under {left_out}_eval.

    Builds a padded data graph so feature dimensions match the training mega-graph,
    then samples min(available, max_eval_queries) non-single-edge query files (or all
    queries when max_eval_queries is None).
    """
    eval_name, eval_graph_file = _build_padded_eval_graph(left_out, dataset_mappings, train_datasets)
    dst_qs_root = QUERYSET_PATH / eval_name
    dst_tc_root = TRUE_CARD_PATH / eval_name

    all_query_files = [
        f for f in (QUERYSET_PATH / left_out).rglob("query_*.txt")
        if not _is_single_edge(f)
    ]

    if max_eval_queries is not None:
        target_count = min(len(all_query_files), max_eval_queries)
    else:
        target_count = len(all_query_files)

    existing_count = len(list(dst_qs_root.rglob("*.txt"))) if dst_qs_root.exists() else 0
    if existing_count == target_count:
        print(f"[{eval_name}] Eval queryset already built ({existing_count} queries), skipping.")
    else:
        if dst_qs_root.exists():
            shutil.rmtree(dst_qs_root)
        if dst_tc_root.exists():
            shutil.rmtree(dst_tc_root)

        rng = random.Random(seed)
        sampled = rng.sample(all_query_files, target_count)

        src_tc_root = TRUE_CARD_PATH / left_out
        for query_file in sampled:
            subdir_name = query_file.parent.name
            dst_qs_dir = dst_qs_root / subdir_name
            dst_tc_dir = dst_tc_root / subdir_name
            dst_qs_dir.mkdir(parents=True, exist_ok=True)
            dst_tc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(query_file, dst_qs_dir / query_file.name)
            tc_file = src_tc_root / subdir_name / query_file.name
            if tc_file.exists():
                shutil.copy2(tc_file, dst_tc_dir / query_file.name)

        print(f"[{eval_name}] Built eval queryset with {target_count} queries.")

    prone_dir = LSS_PATH / "data" / "prone"
    eval_emb = prone_dir / f"{eval_name}.emb.npy"
    if not eval_emb.exists():
        src_emb = prone_dir / f"{left_out}.emb.npy"
        if src_emb.exists():
            eval_emb.symlink_to(src_emb.resolve())

    return eval_name


def _ensure_mega_graph(train_id, train_datasets, dataset_mappings, dataset_caps=None):
    """Build the LOCO mega-graph for train_id (if not already on disk).

    Returns the path to the mega-graph .txt file.
    """
    mega_out = DATASET_BASE_PATH / train_id
    mega_file = mega_out / f"{train_id}.txt"

    if mega_file.exists():
        print(f"[{train_id}] Mega-graph already exists, skipping.")
        node_offsets, pred_offsets, label_offsets = _compute_loco_offsets(train_datasets, dataset_mappings)
    else:
        print(f"[{train_id}] Building mega-graph from: {train_datasets}")
        node_offsets, pred_offsets, label_offsets = _build_mega_graph(train_datasets, dataset_mappings, mega_out)

    print(f"[{train_id}] Syncing queryset and cardinalities (incremental)...")
    _build_loco_queryset(train_datasets, dataset_mappings, node_offsets, pred_offsets, label_offsets, train_id, dataset_caps=dataset_caps)
    print(f"[{train_id}] Queryset done.")

    return mega_file


# ---------------------------------------------------------------------------
# Phase 3.5: ProNE embeddings
# ---------------------------------------------------------------------------

PRONE_PATH = (Path(__file__).parent / "ProNE").resolve()


def gcare_to_prone(path):
    PRONE_DATASET_PATH = PRONE_PATH / "data"
    PRONE_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("e"):
                tokens = line.split()
                src, dst = int(tokens[1]), int(tokens[2])
                edge_labels = [int(t) for t in tokens[3:]]
                edges.append((src, dst, edge_labels))

    out_path = PRONE_DATASET_PATH / f"{path.stem}.prone.txt"
    with open(out_path, "w") as f:
        for src, dst, edge_labels in edges:
            f.write(f"{src} {dst}\n")

    return out_path


def _run_prone(graph_path, out_path):
    script_path = "proNE.py"
    result = subprocess.run([
        sys.executable, script_path,
        "-graph", str(graph_path),
        "-emb2", str(out_path),
        "-dimension", "128",
        "-step", "10",
        "-theta", "0.5",
        "-mu", "0.2",
    ], cwd=PRONE_PATH)
    return result.returncode == 0


def _load_prone_embedding(emb_file):
    """Parse ProNE word2vec-style text output into a float32 numpy array."""
    with open(emb_file) as f:
        n_nodes, dim = map(int, f.readline().split())
        emb = np.zeros((n_nodes, dim), dtype=np.float32)
        for line in f:
            tokens = line.split()
            emb[int(tokens[0])] = [float(v) for v in tokens[1:]]
    return emb


def compute_embeddings(graph_paths):
    """Generate ProNE embeddings for the given graphs (if not already on disk).

    Args:
        graph_paths: dict mapping name -> Path to G-Care .txt file.

    Saves results as LSS/data/prone/{name}.emb.npy.
    """
    prone_dir = LSS_PATH / "data" / "prone"
    prone_dir.mkdir(parents=True, exist_ok=True)
    (PRONE_PATH / "emb").mkdir(parents=True, exist_ok=True)

    for name, graph_file in graph_paths.items():
        emb_path = prone_dir / f"{name}.emb.npy"
        if emb_path.exists():
            print(f"[{name}] Embedding already exists, skipping.")
            continue

        if not graph_file.exists():
            print(f"[{name}] Graph file not found at {graph_file}, skipping.")
            continue

        print(f"[{name}] Converting graph to ProNE format...")
        prone_input = gcare_to_prone(graph_file)

        prone_output = PRONE_PATH / "emb" / f"{name}_enhanced.emb"
        print(f"[{name}] Running ProNE...")
        success = _run_prone(prone_input, prone_output)
        if not success:
            print(f"[{name}] ProNE failed, skipping.")
            continue

        emb = _load_prone_embedding(prone_output)
        np.save(emb_path, emb)
        print(f"[{name}] Saved embedding → {emb_path} shape={emb.shape}")


# ---------------------------------------------------------------------------
# Phase 4: Training and evaluation
# ---------------------------------------------------------------------------

_EXTRA_LSS_ARGS = []


def _run_lss(args, cwd):
    """Run active_train.py with given args as a subprocess."""
    cmd = [sys.executable, "active_train.py"] + args + _EXTRA_LSS_ARGS
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  WARNING: active_train.py exited with code {result.returncode}")
    return result.returncode


def _run_experiment(experiment, results_dir, trained_ids, dataset_mappings, train_datasets, max_eval_queries=None):
    """Train (if needed) and evaluate a single experiment."""
    train_id = _get_train_id(experiment)
    exp_id = experiment["id"]
    left_out = experiment["left_out"]
    fold_results = results_dir / exp_id

    print("=" * 70)
    print(f"Experiment '{exp_id}': model='{train_id}', eval on '{left_out}'")
    print("=" * 70)

    prone_dir = LSS_PATH / "data" / "prone"
    missing_emb = [
        name for name in [train_id, left_out]
        if not (prone_dir / f"{name}.emb.npy").exists()
    ]
    if missing_emb:
        print(f"[{exp_id}] Skipping: missing ProNE embeddings for: {missing_emb}")
        return

    if train_id not in trained_ids:
        model_path = LSS_PATH / "models" / train_id / "best.pth"
        if model_path.exists():
            print(f"[{train_id}] Model already exists on disk, skipping training.")
        else:
            train_res = fold_results / "train"
            train_res.mkdir(parents=True, exist_ok=True)
            _run_lss([
                "--dataset", train_id,
                "--full_data_dir", "./data/",
                "--mode", "train",
                "--model_save_dir", "./models/",
                "--save_res_dir", str(train_res),
                "--embed_type", "freq",
                "--edge_embed_type", "freq",
                "--matching", "homo",
            ], cwd=LSS_PATH)
            # Copy the saved checkpoint to a fixed name so the test phase can find it.
            if not model_path.exists():
                pth_files = list((LSS_PATH / "models" / train_id).glob("model_*.pth"))
                if pth_files:
                    latest = max(pth_files, key=lambda p: p.stat().st_mtime)
                    shutil.copy2(latest, model_path)
        trained_ids.add(train_id)
    else:
        print(f"[{train_id}] Already trained this session, skipping.")

    eval_dataset = _build_capped_eval_queryset(left_out, max_eval_queries, dataset_mappings, train_datasets)

    eval_res = fold_results / "eval"
    eval_res.mkdir(parents=True, exist_ok=True)
    _run_lss([
        "--dataset", eval_dataset,
        "--full_data_dir", "./data/",
        "--mode", "test",
        # Pass an absolute path so os.path.join in load_model ignores model_save_dir/dataset.
        "--model_file", str((LSS_PATH / "models" / train_id / "best.pth").resolve()),
        "--save_res_dir", str(eval_res),
        "--embed_type", "freq",
        "--edge_embed_type", "freq",
        "--matching", "homo",
        "--stage", "eval",
        "--dataset_name", left_out,
    ], cwd=LSS_PATH)

    print(f"[{exp_id}] Done. Results in {fold_results}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA and run on CPU (forwarded to active_train.py)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override active_train.py --epochs for faster/slower runs")
    parser.add_argument("--active-epochs", type=int, default=None,
                        help="Override active_train.py --active_epochs for active-learning rounds")
    parser.add_argument("--active-iters", type=int, default=None,
                        help="Override active_train.py --active_iters")
    parser.add_argument("--quick-test", action="store_true", default=False,
                        help="Run a smoke test configuration (epochs=2, active_epochs=1, active_iters=1)")
    cli_args = parser.parse_args()
    if cli_args.no_cuda:
        _EXTRA_LSS_ARGS.append("--no-cuda")
    if cli_args.quick_test:
        _EXTRA_LSS_ARGS.extend([
            "--epochs", "2",
            "--active_epochs", "1",
            "--active_iters", "1",
        ])
    else:
        if cli_args.epochs is not None:
            _EXTRA_LSS_ARGS.extend(["--epochs", str(cli_args.epochs)])
        if cli_args.active_epochs is not None:
            _EXTRA_LSS_ARGS.extend(["--active_epochs", str(cli_args.active_epochs)])
        if cli_args.active_iters is not None:
            _EXTRA_LSS_ARGS.extend(["--active_iters", str(cli_args.active_iters)])

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    all_dataset_names = [ds["name"] for ds in config["datasets"]]
    dataset_index = {ds["name"]: ds for ds in config["datasets"]}
    query_index = {}  # dataset_name -> list of query configs
    for q in config["queries"]:
        query_index.setdefault(q["dataset_name"], []).append(q)

    dataset_train_caps = {}
    for q in config["queries"]:
        if "max_train_queries" in q:
            ds = q["dataset_name"]
            dataset_train_caps[ds] = dataset_train_caps.get(ds, 0) + q["max_train_queries"]
    max_train_queries_per_dataset = dataset_train_caps if dataset_train_caps else None

    max_eval_queries = config.get("max_eval_queries", None)

    dataset_mappings = {}  # dataset_name -> (id_to_id, id_to_id_pred, type_label)
    trained_ids = set()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = RESULTS_PATH / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for experiment in config["experiments"]:
        left_out_list = _to_list(experiment["left_out"])
        eval_list = _get_eval_datasets(experiment)
        train_id = _get_train_id(experiment)
        train_datasets = _resolve_pool(experiment, all_dataset_names)
        # deduplicated list of all datasets that need graph/query transformation
        needed_datasets = list(dict.fromkeys(train_datasets + left_out_list + eval_list))

        print("=" * 70)
        print(f"Experiment '{experiment['id']}': preparing datasets {needed_datasets}")
        print("=" * 70)

        # Phase 1: transform graphs
        for ds_name in needed_datasets:
            _ensure_graph(ds_name, dataset_index[ds_name], dataset_mappings)

        # Phase 2: transform queries
        for ds_name in needed_datasets:
            if ds_name in query_index:
                _ensure_queries(ds_name, dataset_index[ds_name], query_index[ds_name], dataset_mappings)

        # Phase 3: build LOCO mega-graph and queryset
        mega_graph_file = _ensure_mega_graph(train_id, train_datasets, dataset_mappings, dataset_caps=max_train_queries_per_dataset)

        if not eval_list:
            print(f"[{experiment['id']}] No eval datasets specified, skipping phases 3.5 and 4.")
            continue

        # Phase 3.5: compute embeddings for train + all eval datasets
        emb_graphs = {train_id: mega_graph_file}
        for lo in eval_list:
            emb_graphs[lo] = DATASET_BASE_PATH / lo / f"{lo}.txt"
        compute_embeddings(emb_graphs)

        # Phase 4: train + evaluate on each eval dataset
        for lo in eval_list:
            _run_experiment({**experiment, "left_out": lo}, results_dir, trained_ids, dataset_mappings, train_datasets, max_eval_queries=max_eval_queries)

    print("\nAll experiments complete.")
    print(f"Results saved in: {results_dir}")
