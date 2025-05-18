import h5py
import numpy as np


def build_adj_matrix_from_geo(meta_path, sigma=0.1, distance_threshold=0.3):
    """
    Build adjacency matrix using node geo-coordinates and a Gaussian kernel.

    Parameters:
        meta_path (str): path to pems-bay-meta.h5
        sigma (float): bandwidth for Gaussian kernel
        distance_threshold (float): only connect nodes within this distance

    """
    with h5py.File(meta_path, 'r') as f:
        coords = f["meta/block0_values"][:, 2:4]  # latitude, longitude

    num_nodes = coords.shape[0]
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])  # Euclidean distance
            if dist <= distance_threshold:
                adj[i, j] = np.exp(-dist ** 2 / (2 * sigma ** 2))

    print(f"Constructed adjacency matrix with shape: {adj.shape}")
    return adj

def build_knn_adj_matrix(meta_path, k=5):
    """
    Build adjacency matrix using k-nearest neighbors based on geo-distance.

    Parameters:
        meta_path (str): path to meta file (.h5)
        k (int): number of nearest neighbors

    Returns:
        adj (ndarray): [num_nodes, num_nodes]
    """
    import h5py
    import numpy as np

    with h5py.File(meta_path, 'r') as f:
        coords = f["meta/block0_values"][:, 2:4]  # latitude, longitude

    num_nodes = coords.shape[0]
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # calculate all pairwise distances
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    for i in range(num_nodes):
        neighbor_indices = np.argsort(dists[i])[1:k+1]
        for j in neighbor_indices:
            adj[i, j] = np.exp(-dists[i, j])


    adj = np.maximum(adj, adj.T)

    print(f"kNN adjacency matrix created: shape = {adj.shape}")
    return adj

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix:  D^{-1/2} A D^{-1/2}
    Assumes adj is numpy array, shape [N, N]
    """
    adj = adj + np.eye(adj.shape[0])  # Add self-loops
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return normalized_adj
