import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

with open('D:/Study&Work/Study/硕士课程/CN/data/adj_mx_bay.pkl', 'rb') as f:
    _, _, adj = pickle.load(f, encoding='latin1')


#convert to networkx
G = nx.from_numpy_array(adj)

# Extract most connectaed
if not nx.is_connected(G):
    print("Graph is not connected. Using the largest connected component.")
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()


#calculate centrality
degree_centrality= nx.degree_centrality(G)
centrality_values = np.array([degree_centrality.get(n, 0) for n in G.nodes()])
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
closeness_centrality = nx.closeness_centrality(G)

def print_top(centrality_dict, name, top_n=10):
    print(f"\nTop {top_n} nodes by {name}:")
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_nodes[:top_n]:
        print(f"Node {node}: {score:.4f}")

print_top(degree_centrality, "Degree Centrality")
print_top(betweenness_centrality, "Betweenness Centrality")
print_top(eigenvector_centrality, "Eigenvector Centrality")
print_top(closeness_centrality, "Closeness Centrality")

#visualization
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)

nodes = nx.draw_networkx_nodes(G, pos,
                                node_color=centrality_values,
                                cmap=plt.cm.viridis,
                                node_size=100)

nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.colorbar(nodes, label="Centrality Score")
plt.title("Eigenvector Centrality")
plt.axis('off')
plt.show()
