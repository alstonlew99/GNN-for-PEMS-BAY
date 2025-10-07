import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import pearsonr, spearmanr

def load_centrality(path="centrality_scores.csv"):
    nodes, deg, close, btw, eig = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nodes.append(int(row["node"]))
            deg.append(float(row["degree"]))
            close.append(float(row["closeness"]))
            btw.append(float(row["betweenness"]))
            eig.append(float(row["eigenvector"]))
    nodes = np.array(nodes)
    return {
        "nodes": nodes,
        "degree": np.array(deg, dtype=float),
        "closeness": np.array(close, dtype=float),
        "betweenness": np.array(btw, dtype=float),
        "eigenvector": np.array(eig, dtype=float),
    }

def load_nodewise(path):
    nodes, r, mae, l2, mse = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            nodes.append(int(row["node"]))
            r.append(float(row["r"]))
            mae.append(float(row["mae"]))
            l2.append(float(row["l2"]))
            mse.append(float(row["mse"]))
    return {
        "nodes": np.array(nodes, dtype=int),
        "r": np.array(r, dtype=float),
        "mae": np.array(mae, dtype=float),
        "l2": np.array(l2, dtype=float),
        "mse": np.array(mse, dtype=float),
    }

def align_by_node(cent, perf):
    idx = np.argsort(cent["nodes"])
    jdx = np.argsort(perf["nodes"])
    cn = cent["nodes"][idx]
    pn = perf["nodes"][jdx]
    assert np.array_equal(cn, pn), "node indices mismatch"
    out = {}
    for k,v in cent.items():
        if k=="nodes": out[k]=cn
        else: out[k]=v[idx]
    for k,v in perf.items():
        if k!="nodes": out[k]=v[jdx]
    return out

def corr_and_plot(x, y, xname, yname, prefix):
    mask = np.isfinite(x) & np.isfinite(y)
    x_ = x[mask]; y_ = y[mask]
    pr = pearsonr(x_, y_)
    sr = spearmanr(x_, y_)
    with open(f"{prefix}_corr.txt","w",encoding="utf-8") as f:
        f.write(f"Pearson r={pr.statistic:.6f}, p={pr.pvalue:.3e}\n")
        f.write(f"Spearman rho={sr.correlation:.6f}, p={sr.pvalue:.3e}\n")
    plt.figure(figsize=(6,5))
    plt.scatter(x_, y_, s=12, alpha=0.7)
    if x_.size >= 2:
        p = np.polyfit(x_, y_, 1)
        xv = np.linspace(x_.min(), x_.max(), 100)
        yv = p[0]*xv + p[1]
        plt.plot(xv, yv, linestyle="--", linewidth=1)
    plt.xlabel(xname); plt.ylabel(yname)
    plt.tight_layout(); plt.savefig(f"{prefix}_scatter.png", dpi=200)

def table_corrs(cent_perf, ykey, tag):
    rows = []
    for xkey in ["degree","closeness","betweenness","eigenvector"]:
        x = cent_perf[xkey]; y = cent_perf[ykey]
        mask = np.isfinite(x) & np.isfinite(y)
        pr = pearsonr(x[mask], y[mask])
        sr = spearmanr(x[mask], y[mask])
        rows.append([xkey, ykey, tag, pr.statistic, pr.pvalue, sr.correlation, sr.pvalue])
        corr_and_plot(x, y, xkey, ykey, prefix=f"{tag}_{xkey}_vs_{ykey}")
    with open(f"centrality_vs_{ykey}_corr_table.csv","a",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell()==0:
            w.writerow(["centrality","target","model","pearson_r","pearson_p","spearman_rho","spearman_p"])
        w.writerows(rows)

def main():
    cent = load_centrality("centrality_scores.csv")
    tg = load_nodewise("nodewise_tgcn.csv")
    ml = load_nodewise("nodewise_mlp.csv")

    ctg = align_by_node(cent, tg)
    cml = align_by_node(cent, ml)

    for y in ["mae","mse"]:
        table_corrs(ctg, y, tag="TGCN")
        table_corrs(cml, y, tag="MLP")

    # optional: correlation with (1 - r)
    for tag, cp in [("TGCN", ctg), ("MLP", cml)]:
        invr = 1.0 - cp["r"]
        corr_and_plot(cp["eigenvector"], invr, "eigenvector", "1 - r", f"{tag}_eig_vs_1mr")
        with open(f"{tag}_eig_vs_1mr_corr.txt","w",encoding="utf-8") as f:
            pr = pearsonr(cp["eigenvector"], invr)
            sr = spearmanr(cp["eigenvector"], invr)
            f.write(f"Pearson r={pr.statistic:.6f}, p={pr.pvalue:.3e}\n")
            f.write(f"Spearman rho={sr.correlation:.6f}, p={sr.pvalue:.3e}\n")

    print("Saved: scatter PNGs, *_corr.txt, centrality_vs_mae_corr_table.csv, centrality_vs_mse_corr_table.csv")

if __name__ == "__main__":
    main()
