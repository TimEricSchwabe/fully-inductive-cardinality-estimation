import os
from pathlib import Path
from typing import Literal, Optional


def query_to_gcare(
    query,
    query_idx: int,
    id_to_id_mapping: dict,
    id_to_id_mapping_predicate: dict,
    dataset: str,
    card,
    rdf_type_uri: str,
    mode: Literal["default", "label", "rdf"] = "default",
    gcare_home: Optional[Path] = None,
) -> None:
    """Transform a single query (list of [s, p, o] triples) to G-Care format.

    Writes the result to:
        <gcare_home>/data/queryset/<dataset>/<dataset>_<query_idx>.txt

    *gcare_home* defaults to the GCARE_HOME environment variable, then to
    the ``gcare/`` directory next to this file.
    """
    if gcare_home is None:
        env_home = os.environ.get("GCARE_HOME")
        if env_home:
            gcare_home = Path(env_home)
        else:
            gcare_home = Path(__file__).resolve().parent / "gcare"

    vertices: set = set()
    vertex_labels: dict = {}

    for tp in query:
        vertices.add(tp[0])
        vertices.add(tp[2])

        if tp[1] == rdf_type_uri and "?" not in tp[2]:
            if tp[0] in vertex_labels:
                vertex_labels[tp[0]].append(tp[2])
            else:
                vertex_labels[tp[0]] = [tp[2]]

    vertex_dict: dict = {}
    for vid, vertex in enumerate(vertices):
        try:
            dvid = id_to_id_mapping[vertex] if "?" not in vertex else -1
        except KeyError:
            dvid = 76711
        if vertex in vertex_labels:
            try:
                labels = [id_to_id_mapping[v] for v in vertex_labels[vertex]]
            except KeyError:
                labels = [-1]
        else:
            labels = [-1]
        vertex_dict[vertex] = [vid] + labels + [dvid]

    edge_list = []
    for tp in query:
        edge_label = id_to_id_mapping_predicate[tp[1]] if "?" not in tp[1] else -1
        edge_list.append([vertex_dict[tp[0]][0], vertex_dict[tp[2]][0], edge_label])

    query_path_folder = gcare_home / "data" / "queryset" / dataset
    query_path_folder.mkdir(parents=True, exist_ok=True)
    query_path = query_path_folder / f"{dataset}_{query_idx}.txt"

    with query_path.open("w") as f:
        f.write(f"t # s {query_idx}\n")
        for v in vertex_dict:
            label_str = " ".join(str(l) for l in vertex_dict[v][1:-1])
            f.write(f"v {vertex_dict[v][0]} {label_str} {vertex_dict[v][-1]}\n")
        for e in edge_list:
            f.write(f"e {e[0]} {e[1]} {e[2]}\n")
