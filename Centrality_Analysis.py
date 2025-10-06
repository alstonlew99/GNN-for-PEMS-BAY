import os, pickle, numpy as np, networkx as nx, matplotlib.pyplot as plt
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
    G = nx.from_numpy_array(A)  # 'weight' attribute is populated

    deg = dict(nx.degree(G))  # unweighted degree
    close = nx.closeness_centrality(G)  # unweighted
    btw = nx.betweenness_centrality(G, normalized=True)  # unweighted
    eig = nx.eigenvector_centrality_numpy(G, weight="weight")  # weighted

    nodes = sorted(G.nodes())
    rows = []
    for n in nodes:
        rows.append([n, deg[n], close[n], btw[n], eig[n]])
    save_csv("centrality_scores.csv", ["node","degree","closeness","betweenness","eigenvector"], rows)

    vals = np.array([eig[n] for n in nodes])
    plt.figure(figsize=(8,4))
    plt.bar(nodes, vals, width=1.0)
    plt.xlabel("node"); plt.ylabel("eigenvector centrality")
    plt.tight_layout(); plt.savefig("eigenvector_centrality_bar.png", dpi=200)

if __name__ == "__main__":
    main()
