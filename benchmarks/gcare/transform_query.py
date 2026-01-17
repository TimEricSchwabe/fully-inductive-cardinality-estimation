import json
from pathlib import Path
import os
from typing import Literal, Dict, List

# Environment variables for Docker container paths
GCARE_HOME = os.environ.get('GCARE_HOME', '/usr/src/app/gcare')


def query_to_gcare(query, query_idx, id_to_id_mapping, id_to_id_mapping_predicate, dataset, card, rdf_type_uri, mode: Literal['default', 'label', 'rdf'] = 'default'):
    """Transform a single query to G-Care format."""
    vertices = set()
    vertex_labels = {}
    
    # Get unique vertices and extract type labels
    for tp in query:
        vertices.add(tp[0])
        vertices.add(tp[2])

        if (tp[1] == rdf_type_uri) and ('?' not in tp[2]):
            if tp[0] in vertex_labels:
                vertex_labels[tp[0]] += [tp[2]]
            else:
                vertex_labels[tp[0]] = [tp[2]]

    # Creating Vertex Dict
    vertex_dict = {}
    vid = 0
    for vertex in vertices:
        try:
            dvid = id_to_id_mapping[vertex] if not "?" in vertex else -1
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
        vid += 1

    # Creating Edge List
    edge_list = []
    for tp in query:
        edge_label = id_to_id_mapping_predicate[tp[1]] if not "?" in tp[1] else -1
        edge_list.append([vertex_dict[tp[0]][0], vertex_dict[tp[2]][0], edge_label])

    # Writing the Query File
    query_path_folder = Path(f"{GCARE_HOME}/data/queryset/{dataset}/")
    query_path_folder.mkdir(parents=True, exist_ok=True)

    query_path = query_path_folder / f"{dataset}_{query_idx}.txt"

    with open(str(query_path), "w") as f:
        f.write("t # s " + str(query_idx))
        f.write("\n")
        for v in vertex_dict:
            label_str = ''
            for l in vertex_dict[v][1:-1]:
                label_str += str(l)
                label_str += ' '
            f.write("v " + str(vertex_dict[v][0]) + " " + label_str + str(vertex_dict[v][2]))
            f.write("\n")
        for e in edge_list:
            f.write("e " + str(e[0]) + " " + str(e[1]) + " " + str(e[2]))
            f.write("\n")






