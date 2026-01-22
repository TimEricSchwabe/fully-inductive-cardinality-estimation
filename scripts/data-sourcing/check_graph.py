#!/usr/bin/env python3
"""
Load an N-Triples file using rdflib and execute a SPARQL query.
"""

from rdflib import Graph

# ==================== CONFIGURATION ====================
NT_FILE = "data/fb-237_v4/raw/fb-237_v4.nt"

# SPARQL query to execute
QUERY = """
SELECT * FROM <fb_237_v4> WHERE { ?s <http://example.org/organization/organization/headquarters./location/mailing_address/citytown> ?o0 . ?s <http://example.org/education/educational_institution/students_graduates./education/education/major_field_of_study> ?o1 . ?s <http://example.org/education/educational_institution/students_graduates./education/education/student> ?o2 }
"""
# =======================================================

# Load the N-Triples file
print(f"Loading graph from: {NT_FILE}")
g = Graph()
g.parse(NT_FILE, format="nt")
print(f"Loaded {len(g)} triples")

# Execute the query
print("\nExecuting query...")
results = g.query(QUERY)

# Print results
print("\nQuery results:")
#for row in results:
#    print(row)

print(f"\nTotal results: {len(results)}")

