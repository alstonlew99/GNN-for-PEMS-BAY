import os, pickle, numpy as np, networkx as nx, matplotlib.pyplot as plt
import csv

def load_adj():
    if os.path.exists("adj_used.npy"):
        return np.load("adj_used.npy")
    with open("adj_mx_bay.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def eigenvector_centrality_safe(G):
    try:
        return nx.eigenvector_centrality_numpy(G, weight="weight")
    except nx.AmbiguousSolution:
        vals = {}
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            c = nx.eigenvector_centrality(sub, weight="weight", max_iter=2000, tol=1e-06)
            vals.update(c)
        # global normalize to [0,1]
        mx = max(vals.values()) if len(vals) else 1.0
        if mx > 0:
            for k in vals: vals[k] /= mx
        return vals

def main():
    A = load_adj()
    G = nx.from_numpy_array(A)

    deg = dict(nx.degree(G))
    close = nx.closeness_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    eig = eigenvector_centrality_safe(G)

    nodes = sorted(G.nodes())
    rows = []
    for n in nodes:
        rows.append([n, deg.get(n, 0), close.get(n, 0.0), btw.get(n, 0.0), eig.get(n, 0.0)])
    save_csv("centrality_scores.csv", ["node","degree","closeness","betweenness","eigenvector"], rows)

    vals = np.array([eig.get(n, 0.0) for n in nodes])
    plt.figure(figsize=(8,4))
    plt.bar(nodes, vals, width=1.0)
    plt.xlabel("node"); plt.ylabel("eigenvector centrality")
    plt.tight_layout(); plt.savefig("eigenvector_centrality_bar.png", dpi=200)

if __name__ == "__main__":
    main()
