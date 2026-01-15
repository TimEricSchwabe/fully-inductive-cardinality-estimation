from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import json
from collections import Counter


DEFAULT_RDF_TYPE_URI = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


def prep_dirs(path: Path) -> Dict[str, Path]:
    """
    Create the necessary directory structure for NeuSO dataset.
    
    Args:
        path: Base path for the dataset
        
    Returns:
        Dictionary containing paths to all created directories
    """
    path.mkdir(exist_ok=True, parents=True)
    data_graph_path = path / "data_graph"
    query_graph_path = path / "query_graph"
    filter_path = path / "filter"
    ccg_path = path / "ccg"

    for p in (data_graph_path, query_graph_path, filter_path, ccg_path):
        p.mkdir(exist_ok=True, parents=True)

    return {
        "data_graph_path": data_graph_path,
        "query_graph_path": query_graph_path,
        "filter_path": filter_path,
        "ccg_path": ccg_path,
    }


def save_results(
    output_dir: Path,
    ground_truths: np.ndarray,
    predictions: np.ndarray,
    prediction_times: np.ndarray,
    query_sizes: np.ndarray,
    method_name: str = "neuso"
):
    """
    Save estimation results in the same format as gcare benchmarks.
    
    Args:
        output_dir: Directory to save results
        ground_truths: Array of ground truth cardinalities
        predictions: Array of predicted cardinalities
        prediction_times: Array of prediction times in seconds
        query_sizes: Array of query sizes (number of triples)
        method_name: Name of the method (default: "neuso")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(output_dir / "gts.npy", ground_truths)
    np.save(output_dir / "preds.npy", predictions)
    np.save(output_dir / "pred_times.npy", prediction_times)
    np.save(output_dir / "sizes.npy", query_sizes)
    
    # Compute metrics
    # Q-Error: max(pred/gt, gt/pred)
    q_errors = np.maximum(predictions / (ground_truths + 1e-10), 
                          ground_truths / (predictions + 1e-10))
    
    # MSE and MAE
    mse = np.mean((predictions - ground_truths) ** 2)
    mae = np.mean(np.abs(predictions - ground_truths))
    
    # Median Q-Error
    median_q_error = np.median(q_errors)
    mean_q_error = np.mean(q_errors)
    
    # Mean prediction time
    mean_time = np.mean(prediction_times)
    
    results = {
        "method": method_name,
        "num_queries": len(ground_truths),
        "median_q_error": float(median_q_error),
        "mean_q_error": float(mean_q_error),
        "mse": float(mse),
        "mae": float(mae),
        "mean_prediction_time_seconds": float(mean_time),
        "total_prediction_time_seconds": float(np.sum(prediction_times))
    }
    
    # Save results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_dir}")
    print(f"Median Q-Error: {median_q_error:.4f}")
    print(f"Mean Q-Error: {mean_q_error:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean prediction time: {mean_time:.6f}s")


def load_queries(query_path: Path) -> List[Dict]:
    """
    Load queries from JSON file.
    
    Args:
        query_path: Path to query JSON file
        
    Returns:
        List of query dictionaries
    """
    with open(query_path, "r") as f:
        queries = json.load(f)
    return queries


def most_common_predicates(path: Union[str, Path], n: int = 10) -> List[tuple]:
    """
    Identify the most common RDF (type) predicates in a given RDF dataset file.
    The most common predicates are returned as a list of tuples containing the predicate and its count.
    We assume that the most common predicates is rdf:type.

    Parameters:
    - path (Union[str, Path]): The file path to the RDF dataset.
    - n (int): The number of most common predicates to return.

    Returns:
    - List[tuple]: A list of tuples where each tuple contains a predicate and its count.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"The file at {path} does not exist.")

    with path.open("r") as f:
        data = f.readlines()

    # process
    predicates = [d.split(" ")[1] for d in data]
    predicate_counts = Counter(predicates)
    for p in predicate_counts:
        if DEFAULT_RDF_TYPE_URI in p:
            print(f"Found rdf:type predicate: {p}")
            return p

    return predicate_counts.most_common(n)
