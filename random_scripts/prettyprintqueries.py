import json
import re
import requests
import tqdm 


with open('data/lubm_queries/raw/59410580', 'r') as f:
    data = json.load(f)

for query in tqdm.tqdm(data):

    # Execute Query against the endpoint
    sparql_query = query['query'][0]
    
    # Replace SELECT clause with COUNT
    sparql_query = re.sub(r'SELECT\s+.*?WHERE', 'SELECT (COUNT(*) as ?count) WHERE', sparql_query, flags=re.IGNORECASE | re.DOTALL)
    
    # Execute query against HTTP endpoint
    endpoint_url = "http://tim-ThinkPad-X1-Carbon-Gen-11:7001"  # Replace with actual endpoint
    response = requests.post(endpoint_url, 
                            data={'query': sparql_query},
                            headers={'Accept': 'application/sparql-results+json'})
    
    # Extract count from response
    result = response.json()
    count = int(result['results']['bindings'][0]['count']['value'])

    if count != query['y']:
        print(sparql_query)
        print(count)
        print(query['y'])
        print("-"*100)
