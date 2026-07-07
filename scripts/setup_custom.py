import csv
import json
import re
import os

def parse_sparql(query_str):
    # Extract lines between WHERE { and }
    match = re.search(r'WHERE\s*\{([^}]+)\}', query_str)
    if not match:
        return []
    
    body = match.group(1).strip()
    triples = []
    for line in body.split('.'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            triples.append([parts[0], parts[1], parts[2]])
    return triples

def process_csv(csv_path, out_json_path):
    out_dir = os.path.dirname(out_json_path)
    os.makedirs(out_dir, exist_ok=True)
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_str = row['query']
            card = int(row['cardinality'])
            triples = parse_sparql(q_str)
            data.append({
                "triples": triples,
                "y": card
            })
            
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} queries to {out_json_path}")

process_csv('../orthogonal_and_two_cluster_kgs/two_clusters_queries.csv', 'data/two_clusters_queries/raw/queries.json')
process_csv('../orthogonal_and_two_cluster_kgs/orthogonal_queries.csv', 'data/orthogonal_queries/raw/queries.json')
