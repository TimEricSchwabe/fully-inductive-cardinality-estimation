import json
import random
from pathlib import Path
import os
from typing import Literal
from rdf_types import *

# Environment variables for Docker container paths
GCARE_HOME = os.environ.get('GCARE_HOME', '/usr/src/app/gcare')


def remove_files_in_folder(folder_path):
    # List all files in the folder
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Folder {folder_path} not found. Skipping removal of files.")
        return

    # Loop through each file and remove it
    for file in files:
        file_path = os.path.join(folder_path, file)

        # Check if it's a file (not a folder)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"Skipping {file_path}, not a file")



def query_to_gcare(query, query_idx, id_to_id_mapping, id_to_id_mapping_predicate, dataset, card, mode: Literal['default', 'label', 'rdf'] = 'default'):
    # Delete existing query:
    # remove_files_in_folder(f"{GCARE_HOME}/data/queryset/{dataset}")

    vertices = set()
    vertex_labels = {}
    # Get unique vertices
    for tp in query:
        vertices.add(tp[0])
        vertices.add(tp[2])
        if dataset == 'yago' or 'yago_inductive':
            rdf_type_uri = rdf_type_uri_yago
        elif dataset == 'wikidata':
            rdf_type_uri = rdf_type_uri_wikidata
        elif dataset == 'swdf' or dataset == 'swdf_inductive':
            rdf_type_uri = rdf_type_uri_swdf
        elif dataset == 'lubm':
            rdf_type_uri = rdf_type_uri_lubm
        elif dataset == 'wn18rr' or dataset == 'wn18rr_ind':
            rdf_type_uri = rdf_type_uri_wn18rr
        elif dataset == 'dbpedia100k':
            rdf_type_uri = rdf_type_uri_dbpedia100k
        else:
            raise AssertionError("rdf type uri missing !")

        if (tp[1] == rdf_type_uri) and ('?' not in tp[2]):

            if tp[0] in vertex_labels:
                vertex_labels[tp[0]] += [tp[2]]
            else:
                vertex_labels[tp[0]] = [tp[2]]

    # Creating Vertex Dict
    vertex_dict = {}
    vid = 0
    for vertex in vertices:
        # dvid = vertex.split("/")[-1].replace(">", "") if not "?" in vertex else -1
        try:
            dvid = id_to_id_mapping[vertex] if not "?" in vertex else -1
        except KeyError:
            dvid = 76711
        if vertex in vertex_labels:
            # print('In vertex')
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
        # edge_label = tp[1].split("/")[-1].replace(">", "") if not "?" in tp[1] else -1
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



