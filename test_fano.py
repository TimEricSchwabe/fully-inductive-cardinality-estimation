from scripts.utils.data_helper import load_dataset
try:
    ds = load_dataset("fano_dataset")
    print(f"Loaded fano_dataset successfully. Nodes: {ds.num_nodes}")
except Exception as e:
    print(f"Error loading fano_dataset: {e}")
