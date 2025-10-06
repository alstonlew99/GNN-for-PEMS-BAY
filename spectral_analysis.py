import os, pickle, numpy as np, matplotlib.pyplot as plt

def load_adj():
    if os.path.exists("adj_used.npy"):
        return np.load("adj_used.npy")
    with open("adj_mx_bay.pkl", "rb") as f:
        return pickle.load(f)

def normalized_laplacian(A):
    A = np.array(A, dtype=float)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d + 1e-12, -0.5)
    D_inv_s = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_s @ A @ D_inv_s
    return L

def main():
    A = load_adj()
    L = normalized_laplacian(A)
    w, V = np.linalg.eigh(L)
    np.save("laplacian_eigvals.npy", w)
    np.save("laplacian_eigvecs.npy", V)

    plt.figure(figsize=(8,4))
    plt.plot(np.sort(w))
    plt.xlabel("index"); plt.ylabel("eigenvalue")
    plt.tight_layout(); plt.savefig("laplacian_spectrum.png", dpi=200)

    if V.shape[1] >= 3:
        e2 = V[:, 1]; e3 = V[:, 2]
        plt.figure(figsize=(6,6))
        plt.scatter(e2, e3, s=10, alpha=0.8)
        plt.xlabel("eigenvector 2"); plt.ylabel("eigenvector 3")
        plt.tight_layout(); plt.savefig("spectral_embedding_e2_e3.png", dpi=200)

    if w.size >= 2:
        with open("algebraic_connectivity.txt","w",encoding="utf-8") as f:
            f.write(f"lambda_2={np.sort(w)[1]:.8f}\n")

if __name__ == "__main__":
    main()
