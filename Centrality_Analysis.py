import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

#Paths
adj_path = "D:/Study&Work/Study/硕士课程/CN/data/adj_mx_bay.pkl"
output_dir = "D:/Study&Work/Study/硕士课程/CN/Results"
os.makedirs(output_dir, exist_ok=True)

#Read
with open(adj_path, 'rb') as f:
    _, _, adj = pickle.load(f, encoding='latin1')
adj = np.array(adj)

# construct NetworkX
G = nx.from_numpy_array(adj)

# maximum connection
if not nx.is_connected(G):
    print("Graph is not connected. Using the largest connected component.")
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

#Calculate centrality
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
closeness_centrality = nx.closeness_centrality(G)

#print top
def print_top(centrality_dict, name, top_n=10):
    print(f"\nTop {top_n} nodes by {name}:")
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_nodes[:top_n]:
        print(f"Node {node}: {score:.4f}")

print_top(degree_centrality, "Degree Centrality")
print_top(betweenness_centrality, "Betweenness Centrality")
print_top(eigenvector_centrality, "Eigenvector Centrality")
print_top(closeness_centrality, "Closeness Centrality")

#save
centrality_df = pd.DataFrame({
    'Node': list(G.nodes()),
    'Degree': [degree_centrality[n] for n in G.nodes()],
    'Betweenness': [betweenness_centrality[n] for n in G.nodes()],
    'Eigenvector': [eigenvector_centrality[n] for n in G.nodes()],
    'Closeness': [closeness_centrality[n] for n in G.nodes()],
})
centrality_df.to_csv(os.path.join(output_dir, "centrality_scores.csv"), index=False)
print(f"Centrality scores saved to {output_dir}/centrality_scores.csv")

# Visualization
centrality_values = np.array([eigenvector_centrality.get(n, 0) for n in G.nodes()])
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 8))
nodes = nx.draw_networkx_nodes(G, pos, node_color=centrality_values,
                                cmap=plt.cm.viridis, node_size=100)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.colorbar(nodes, label="Eigenvector Centrality")
plt.title("Eigenvector Centrality Visualization")
plt.axis('off')
save_path = os.path.join(output_dir, "eigenvector_centrality.png")
plt.savefig(save_path, dpi=300)
print(f"Centrality visualization saved to {save_path}")
plt.show()
