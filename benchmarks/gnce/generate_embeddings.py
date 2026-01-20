from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
import json
from pathlib import Path
from collections import Counter

def generate_embeddings_for_dataset(graph_file, query_files, output_statistics_dir):
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
    
    entities = set(entities)
    print(f'Using {len(entities)} entities for RDF2Vec')
    
    # Generate embeddings
    kg = KG(graph_file)
    
    walker = RandomWalker(max_depth=4, max_walks=10, with_reverse=False, n_jobs=24)
    embedder = Word2Vec(epochs=10, vector_size=100)
    
    model = RDF2VecTransformer(
        walkers=[walker],
        embedder=embedder,
        verbose=1
    )
    
    embeddings = model.fit_transform(kg, entities)
    
    # Save to statistics format
    Path(output_statistics_dir).mkdir(parents=True, exist_ok=True)
    
    for entity, embedding in zip(entities, embeddings):
        safe_filename = entity.replace("/", "|") + ".json"
        entity_file = Path(output_statistics_dir) / safe_filename
        
        # embedding is already a numpy array, convert to list
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        with open(entity_file, 'w') as f:
            json.dump({
                "embedding": embedding_list,
                "occurence": entity_counts.get(entity, 1)
            }, f)
    
    return {entity: (embedding.tolist() if hasattr(embedding, 'tolist') else embedding) 
            for entity, embedding in zip(entities, embeddings)} 
