import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance

def graph_similarity(graph1, graph2):
    # Check if both graphs are empty
    if len(graph1) == 0 and len(graph2) == 0:
        return 1.0  # Both graphs are identical (empty)
    
    # Calculate the graph edit distance
    edit_distance = graph_edit_distance(graph1, graph2)
    
    # Calculate the maximum possible edit distance
    max_distance = max(len(graph1), len(graph2))
    
    # Normalize the similarity score to be between 0 and 1
    similarity_score = 1 - (edit_distance / max_distance)
    
    return similarity_score

# Example usage
if __name__ == "__main__":
    # Create two simple graphs
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 4)])

    G2 = nx.Graph()
    G2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

    G3 = nx.Graph()
    G3.add_edges_from([(1, 2), (2, 3), (3, 5)])

    # Calculate similarity scores
    score1 = graph_similarity(G1, G2)
    score2 = graph_similarity(G1, G3)
    score3 = graph_similarity(G2, G3)

    print(f"Similarity between G1 and G2: {score1:.2f}")
    print(f"Similarity between G1 and G3: {score2:.2f}")
    print(f"Similarity between G2 and G3: {score3:.2f}")
