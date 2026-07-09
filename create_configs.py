import os

configs = [
    ('fano_dataset', 'fano_queries', 'train_fano.yaml'),
    ('ising_dataset', 'ising_queries', 'train_ising.yaml'),
    ('orthogonal_dataset', 'orthogonal_queries', 'train_orthogonal.yaml'),
    ('two_clusters_dataset', 'two_clusters_queries', 'train_two_clusters.yaml')
]

with open('configs/train_local.yaml', 'r') as f:
    content = f.read()

for ds, q, name in configs:
    # simple string replacement to keep comments and formatting intact
    new_content = content.replace('"cycle_dataset"', f'"{ds}"').replace('"cycle_queries"', f'"{q}"').replace('"RelationalGNN-local"', f'"RelationalGNN-local-{ds.split("_")[0]}"')
    with open(f'configs/{name}', 'w') as f:
        f.write(new_content)

os.remove('create_configs.py')
print("Configs created successfully.")
