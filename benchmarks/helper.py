from pathlib import Path
from typing import Union, List
import numpy as np
from collections import Counter


DEFAULT_RDF_TYPE_URI = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

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
