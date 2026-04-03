rdf_type_uri_yago = "<http://example.com/13000179>"  # Yago
rdf_type_uri_lubm = "<http://example.org/1>"  # Lubm
rdf_type_uri_wn18rr = '<http://example.org_derivationally_related_form>'  # wn18rr / wn18rr_ind
rdf_type_uri_dbpedia100k = '<http://example.orggenre>'  # dbpedia100k
rdf_type_uri_wikidata = '<http://www.wikidata.org/prop/direct/P31>'  # wikidata
rdf_type_uri_swdf = '<http://ex.org/03>'  # SWDF
DEFAULT_RDF_TYPE_URI = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


def get_rdf_type_uri_for_dataset(dataset: str) -> str:
    """
    Get the RDF type URI for a given dataset name.
    
    Args:
        dataset: Name of the dataset (e.g., 'yago', 'wikidata', 'lubm', etc.)
        
    Returns:
        The RDF type URI string for the dataset
        
    Raises:
        ValueError: If the dataset name is not recognized
        
    Examples:
        >>> get_rdf_type_uri_for_dataset('yago')
        '<http://example.com/13000179>'
        >>> get_rdf_type_uri_for_dataset('wikidata')
        '<http://www.wikidata.org/prop/direct/P31>'
    """
    dataset_map = {
        'yago': rdf_type_uri_yago,
        'yago_inductive': rdf_type_uri_yago,
        'wikidata': rdf_type_uri_wikidata,
        'swdf': rdf_type_uri_swdf,
        'swdf_inductive': rdf_type_uri_swdf,
        'lubm': rdf_type_uri_lubm,
        'wn18rr': rdf_type_uri_wn18rr,
        'wn18rr_ind': rdf_type_uri_wn18rr,
        'dbpedia100k': rdf_type_uri_dbpedia100k,
    }
    
    if dataset not in dataset_map:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. "
            f"Supported datasets: {', '.join(sorted(dataset_map.keys()))}"
        )
    
    return dataset_map[dataset]