import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from rdflib import Graph, URIRef, Namespace

g = Graph()
EX = Namespace("https://project.org/data/")
friendsWith = EX.friendsWith

node_id = 0

# Create cycles of varying lengths: from 3 to 7 nodes
for length in range(3, 8):
    # Changed the multiplier to 77 to hit the ~5000 nodes target perfectly
    num_cycles = 77 * (8 - length)

    for _ in range(num_cycles):
        # Save the very first node of the cycle to close the loop later
        first_node = URIRef(f"{EX}person_{node_id}")
        current_node = first_node

        # Build the chain
        for step in range(1, length):
            next_node = URIRef(f"{EX}person_{node_id + step}")
            g.add((current_node, friendsWith, next_node))
            current_node = next_node

        # Close the cycle: connect the last node back to the first node
        g.add((current_node, friendsWith, first_node))

        # Shift the node ID counter so separate cycles do not intersect
        node_id += length

# Save to the .nt file
g.serialize(destination="raw_dataset.nt", format="nt")
