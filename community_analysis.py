import os, pickle, numpy as np, networkx as nx, matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
import csv

def load_adj():
    if os.path.exists("adj_used.npy"):
        return np.load("adj_used.npy")
    with open("adj_mx_bay.pkl", "rb") as f:
        return pickle.load(f)

def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def main():
    A = load_adj()
    G = nx.from_numpy_array(A)

    comms = list(greedy_modularity_communities(G, weight="weight"))
    part = {}
    for cid, c in enumerate(comms):
        for n in c:
            part[n] = cid

    q = modularity(G, comms, weight="weight")
    with open("community_modularity.txt","w",encoding="utf-8") as f:
        f.write(f"num_communities={len(comms)}\nmodularity={q:.6f}\n")

    labels = [[n, part[n]] for n in sorted(G.nodes())]
    save_csv("community_labels.csv", ["node","community"], labels)

    sizes = [[i, len(c)] for i, c in enumerate(comms)]
    save_csv("community_sizes.csv", ["community","size"], sizes)

    pos = nx.spring_layout(G, seed=42, weight="weight")
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(8,8))
    for i, c in enumerate(comms):
        nx.draw_networkx_nodes(G, pos, nodelist=list(c), node_size=20, node_color=[cmap(i % 20)])
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3)
    plt.axis("off"); plt.tight_layout(); plt.savefig("communities.png", dpi=300)

if __name__ == "__main__":
    main()
