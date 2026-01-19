from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
import json
from pathlib import Path
from collections import Counter


def generate_embeddings_for_dataset(graph_file, query_files, output_statistics_dir, 
                                    max_depth=4, max_walks=10, n_jobs=24, 
                                    epochs=10, vector_size=100):
    # Extract entities from queries
    entities = []
    entity_counts = Counter()
    
    for query_file in query_files:
        with open(query_file, 'r') as f:
            queries = json.load(f)
        
        for query in queries:
            query_entities = query.get('x', [])
            entities.extend(query_entities)
            entity_counts.update(query_entities)
            
            if 'instantiated_objects' in query:
                inst_objs = [obj.strip('<>') for obj in query['instantiated_objects']]
                entities.extend(inst_objs)
                entity_counts.update(inst_objs)
            
            for triple in query.get('triples', []):
                if len(triple) >= 3:
                    entities.append(triple[1])
                    entity_counts[triple[1]] += 1
    
    entities = list(set(entities))
    print(f'{len(entities)} unique entities')
    entities = entities[:20]
    # Generate embeddings
    kg = KG(graph_file)
    
    # Check which entities exist in the graph
    kg_entities = [e.name for e in kg._vertices]
    kg_entities = set(kg_entities)
    missing_entities = [e for e in entities if e not in kg_entities]
    
    if missing_entities:
        print(f"\nERROR: {len(missing_entities)} entities not found in graph:")
        for entity in missing_entities[:10]:  # Print first 10
            print(f"  - {entity}")
        if len(missing_entities) > 10:
            print(f"  ... and {len(missing_entities) - 10} more")
        raise ValueError(f"At least one provided entity does not exist in the Knowledge Graph.")
    
    walker = RandomWalker(max_depth=max_depth, max_walks=max_walks, 
                         with_reverse=False, n_jobs=n_jobs)
    embedder = Word2Vec(epochs=epochs, vector_size=vector_size)
    model = RDF2VecTransformer(walkers=[walker], embedder=embedder, verbose=1)
    
    embeddings = model.fit_transform(kg, entities)
    
    # Save to statistics format
    Path(output_statistics_dir).mkdir(parents=True, exist_ok=True)
    
    for entity, embedding in zip(entities, embeddings):
        safe_filename = entity.replace("/", "|") + ".json"
        entity_file = Path(output_statistics_dir) / safe_filename
        
        with open(entity_file, 'w') as f:
            json.dump({
                "embedding": embedding.tolist(),
                "occurence": entity_counts.get(entity, 1)
            }, f)
    
    return {entity: embedding.tolist() for entity, embedding in zip(entities, embeddings)} 
