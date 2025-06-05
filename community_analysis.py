import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.algorithms.community import greedy_modularity_communities

output_dir = "community_results"
os.makedirs(output_dir, exist_ok=True)

with open("D:/Study&Work/Study/硕士课程/CN/data/adj_mx_bay.pkl", "rb") as f:
    _, _, adj = pickle.load(f, encoding="latin1")

G = nx.from_numpy_array(adj)

if not nx.is_connected(G):
    print("Graph is not connected. Using largest connected component.")
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

#Greedy modularity
communities = list(greedy_modularity_communities(G))
print(f"Detected {len(communities)} communities.")

with open(os.path.join(output_dir, "communities.txt"), "w") as f:
    for i, comm in enumerate(communities):
        comm_list = sorted(list(comm))
        f.write(f"Community {i+1} ({len(comm_list)} nodes): {comm_list}\n")

node_color_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_color_map[node] = i
node_colors = [node_color_map[n] for n in G.nodes()]

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 8))

nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20, node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.3)

plt.title("Community Detection (Greedy Modularity)")
plt.axis("off")

plt.colorbar(nodes, label="Community")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "community_visualization.png"), dpi=300)
plt.close()
print(f"Community visualization saved to {output_dir}/community_visualization.png")

