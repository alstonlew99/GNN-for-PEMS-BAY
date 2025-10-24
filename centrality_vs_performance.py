import csv, numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.stats import pearsonr

def safe_int(x):
    try:
        return int(float(x))
    except:
        return int(x)

def load_nodewise(path):
    nodes, r, mae, mse = [], [], [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(safe_int(row["node"]))
            r.append(float(row["r"]))
            mae.append(float(row["mae"]))
            mse.append(float(row["mse"]))
    return np.array(nodes), np.array(r), np.array(mae), np.array(mse)

def load_centrality(path):
    df = pd.read_csv(path)
    nodes = df["node"].astype(int).values
    degree = df["degree"].values
    close = df["closeness"].values
    btw = df["betweenness"].values
    eig = df["eigenvector"].values
    return nodes, degree, close, btw, eig

def scatter_and_corr(x, y, xlabel, ylabel, fname):
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask): return np.nan
    r, p = pearsonr(x[mask], y[mask])
    plt.figure(figsize=(6,5))
    plt.scatter(x[mask], y[mask], s=12, alpha=0.6)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel} (r={r:.3f}, p={p:.2e})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    return r, p

def main():
    tg_nodes, tg_r, tg_mae, tg_mse = load_nodewise("nodewise_tgcn.csv")
    ml_nodes, ml_r, ml_mae, ml_mse = load_nodewise("nodewise_mlp.csv")
    c_nodes, degree, close, btw, eig = load_centrality("centrality_scores.csv")

    df = pd.DataFrame({
        "node": c_nodes,
        "degree": degree,
        "closeness": close,
        "betweenness": btw,
        "eigenvector": eig
    })

    df_tg = pd.DataFrame({"node": tg_nodes, "r_tgcn": tg_r, "mae_tgcn": tg_mae, "mse_tgcn": tg_mse})
    df_ml = pd.DataFrame({"node": ml_nodes, "r_mlp": ml_r, "mae_mlp": ml_mae, "mse_mlp": ml_mse})

    df_all = df.merge(df_tg, on="node").merge(df_ml, on="node")
    df_all.to_csv("centrality_performance_merge.csv", index=False)

    results = []
    for name, arr in zip(["degree","closeness","betweenness","eigenvector"],
                         [degree, close, btw, eig]):
        r_mae_tg, p1 = scatter_and_corr(arr, df_all["mae_tgcn"].values, name, "MAE (T-GCN)", f"scat_{name}_mae_tgcn.png")
        r_mae_ml, p2 = scatter_and_corr(arr, df_all["mae_mlp"].values, name, "MAE (MLP)", f"scat_{name}_mae_mlp.png")
        r_r_tg, p3 = scatter_and_corr(arr, df_all["r_tgcn"].values, name, "r (T-GCN)", f"scat_{name}_r_tgcn.png")
        r_r_ml, p4 = scatter_and_corr(arr, df_all["r_mlp"].values, name, "r (MLP)", f"scat_{name}_r_mlp.png")
        results.append([name, r_mae_tg, p1, r_mae_ml, p2, r_r_tg, p3, r_r_ml, p4])

    cols = ["metric","r(mae_tgcn)","p1","r(mae_mlp)","p2","r(r_tgcn)","p3","r(r_mlp)","p4"]
    pd.DataFrame(results, columns=cols).to_csv("centrality_vs_perf_corr_table.csv", index=False)

    print("Saved: centrality_performance_merge.csv, centrality_vs_perf_corr_table.csv, scatter plots.")

if __name__ == "__main__":
    main()
