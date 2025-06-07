import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

adj_path = "D:/Study&Work/Study/硕士课程/CN/data/adj_mx_bay.pkl"
output_dir = "D:/Study&Work/Study/硕士课程/CN/Results"
os.makedirs(output_dir, exist_ok=True)

# load adj
with open(adj_path, 'rb') as f:
    _, _, adj = pickle.load(f, encoding='latin1')

G = nx.from_numpy_array(adj)

if not nx.is_connected(G):
    print("Graph not connected, using largest connected component.")
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

# Graph Laplace Matrix
L = nx.normalized_laplacian_matrix(G).toarray()
eigvals, eigvecs = np.linalg.eigh(L)  # eigh 用于对称矩阵

#Spectral distribution
plt.figure(figsize=(8, 4))
plt.plot(np.arange(len(eigvals)), eigvals, 'bo-', markersize=3)
plt.title("Laplacian Spectrum")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "laplacian_spectrum.png"), dpi=300)
plt.close()
print("Laplacian spectrum plot saved.")

# algebraic connectivity
algebraic_connectivity = eigvals[1]
print(f"Algebraic Connectivity (2nd smallest eigenvalue): {algebraic_connectivity:.4f}")

# spectral embedding
embedding = eigvecs[:, 1:3]
plt.figure(figsize=(6, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c='b', alpha=0.6)
plt.title("Spectral Embedding using Laplacian Eigenvectors")
plt.xlabel("2nd eigenvector")
plt.ylabel("3rd eigenvector")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectral_embedding.png"), dpi=300)
plt.close()
print("Spectral embedding plot saved.")
