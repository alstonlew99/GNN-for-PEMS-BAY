import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

output_dir = "community_results"
os.makedirs(output_dir, exist_ok=True)

with open("D:/Study&Work/Study/硕士课程/CN/data/adj_mx_bay.pkl", "rb") as f:
    _, _, adj = pickle.load(f, encoding="latin1")

G = nx.from_numpy_array(adj)

if not nx.is_connected(G):
    print("Graph is not connected. Using largest connected component.")
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

# Greedy modularity
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

mod_score = modularity(G, communities)
community_sizes = [len(c) for c in communities]
avg_size = np.mean(community_sizes)
max_size = np.max(community_sizes)
min_size = np.min(community_sizes)

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"Modularity score: {mod_score:.4f}\n")
    f.write(f"Number of communities: {len(communities)}\n")
    f.write(f"Average community size: {avg_size:.2f}\n")
    f.write(f"Max community size: {max_size}\n")
    f.write(f"Min community size: {min_size}\n")

print(f"Metrics saved to {output_dir}/metrics.txt")
