import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

def nodewise_stats(y_true, y_pred):
    T, N = y_true.shape
    r = np.full(N, np.nan, dtype=float)
    mae = np.zeros(N, dtype=float)
    l2 = np.zeros(N, dtype=float)
    mse = np.zeros(N, dtype=float)
    for v in range(N):
        yt = y_true[:, v]
        yp = y_pred[:, v]
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            r[v] = pearsonr(yt, yp)[0]
        mae[v] = np.mean(np.abs(yp - yt))
        l2[v] = np.linalg.norm(yp - yt)
        mse[v] = np.mean((yp - yt) ** 2)
    return r, mae, l2, mse

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["y_true"], z["y_pred"]

def save_csv(path, header, arr2d):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in arr2d:
            f.write(",".join(str(x) for x in row) + "\n")

def hist_two(ax, a, b, bins, label_a, label_b):
    ax.hist(a[~np.isnan(a)], bins=bins, alpha=0.6, density=False, label=label_a)
    ax.hist(b[~np.isnan(b)], bins=bins, alpha=0.6, density=False, label=label_b)
    ax.legend(); ax.grid(True, alpha=0.3)

def violin_two(ax, a, b, labels):
    ax.violinplot([a[~np.isnan(a)], b[~np.isnan(b)]], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)

def scatter_compare(ax, x, y, xlabel, ylabel):
    ax.scatter(x, y, s=10, alpha=0.6)
    lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
    hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

def main():
    y_true_tg, y_pred_tg = load_npz("tgcn_test_outputs.npz")
    y_true_ml, y_pred_ml = load_npz("mlp_test_outputs.npz")

    # align time length
    T = min(y_true_tg.shape[0], y_true_ml.shape[0])
    y_true_tg, y_pred_tg = y_true_tg[:T], y_pred_tg[:T]
    y_true_ml, y_pred_ml = y_true_ml[:T], y_pred_ml[:T]

    assert y_true_tg.shape == y_pred_tg.shape == y_true_ml.shape == y_pred_ml.shape

    r_tg, mae_tg, l2_tg, mse_tg = nodewise_stats(y_true_tg, y_pred_tg)
    r_ml, mae_ml, l2_ml, mse_ml = nodewise_stats(y_true_ml, y_pred_ml)

    N = y_true_tg.shape[1]
    nodes = np.arange(N)

    comp = np.column_stack([
        nodes, r_tg, r_ml, mae_tg, mae_ml, l2_tg, l2_ml, mse_tg, mse_ml,
        mae_tg - mae_ml, r_tg - r_ml
    ])
    header = ["node","r_tgcn","r_mlp","mae_tgcn","mae_mlp","l2_tgcn","l2_mlp","mse_tgcn","mse_mlp","delta_mae(tgcn-mlp)","delta_r(tgcn-mlp)"]
    save_csv("nodewise_compare.csv", header, comp)

    save_csv("nodewise_tgcn.csv", ["node","r","mae","l2","mse"], np.column_stack([nodes, r_tg, mae_tg, l2_tg, mse_tg]))
    save_csv("nodewise_mlp.csv",   ["node","r","mae","l2","mse"], np.column_stack([nodes, r_ml, mae_ml, l2_ml, mse_ml]))

    bins_r = 20
    bins_e = 20

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    hist_two(ax, r_tg, r_ml, bins=bins_r, label_a="T-GCN r", label_b="MLP r")
    ax.set_title("Node-wise Pearson r distribution")
    plt.tight_layout(); plt.savefig("dist_r_hist.png", dpi=200)

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    hist_two(ax, mae_tg, mae_ml, bins=bins_e, label_a="T-GCN MAE", label_b="MLP MAE")
    ax.set_title("Node-wise MAE distribution")
    plt.tight_layout(); plt.savefig("dist_mae_hist.png", dpi=200)

    plt.figure(figsize=(7,6))
    ax = plt.gca()
    violin_two(ax, mae_tg, mae_ml, labels=["T-GCN","MLP"])
    ax.set_title("Node-wise MAE violin")
    plt.tight_layout(); plt.savefig("dist_mae_violin.png", dpi=200)

    plt.figure(figsize=(7,6))
    ax = plt.gca()
    violin_two(ax, r_tg, r_ml, labels=["T-GCN","MLP"])
    ax.set_title("Node-wise r violin")
    plt.tight_layout(); plt.savefig("dist_r_violin.png", dpi=200)

    plt.figure(figsize=(7,6))
    ax = plt.gca()
    scatter_compare(ax, r_ml, r_tg, xlabel="MLP r", ylabel="T-GCN r")
    ax.set_title("Node-wise r: T-GCN vs MLP")
    plt.tight_layout(); plt.savefig("scatter_r_tgcn_vs_mlp.png", dpi=200)

    plt.figure(figsize=(7,6))
    ax = plt.gca()
    scatter_compare(ax, mae_ml, mae_tg, xlabel="MLP MAE", ylabel="T-GCN MAE")
    ax.set_title("Node-wise MAE: T-GCN vs MLP")
    plt.tight_layout(); plt.savefig("scatter_mae_tgcn_vs_mlp.png", dpi=200)

    diff_mae = mae_tg - mae_ml
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.hist(diff_mae[~np.isnan(diff_mae)], bins=25, alpha=0.8)
    ax.set_title("Delta MAE (T-GCN - MLP)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("diff_mae_hist.png", dpi=200)

    mask_mae = np.isfinite(mae_tg) & np.isfinite(mae_ml)
    if np.any(mask_mae):
        w_mae = wilcoxon(mae_tg[mask_mae], mae_ml[mask_mae], zero_method="wilcox", alternative="two-sided", method="auto", nan_policy="omit")
        with open("paired_tests.txt", "w", encoding="utf-8") as f:
            f.write(f"Wilcoxon MAE T-GCN vs MLP: statistic={w_mae.statistic}, p={w_mae.pvalue}\n")
    mask_r = np.isfinite(r_tg) & np.isfinite(r_ml)
    if np.any(mask_r):
        w_r = wilcoxon(r_tg[mask_r], r_ml[mask_r], zero_method="wilcox", alternative="two-sided", method="auto", nan_policy="omit")
        with open("paired_tests.txt", "a", encoding="utf-8") as f:
            f.write(f"Wilcoxon r T-GCN vs MLP: statistic={w_r.statistic}, p={w_r.pvalue}\n")

    print("Saved: nodewise_compare.csv, nodewise_tgcn.csv, nodewise_mlp.csv")
    print("Saved: dist_r_hist.png, dist_mae_hist.png, dist_mae_violin.png, dist_r_violin.png")
    print("Saved: scatter_r_tgcn_vs_mlp.png, scatter_mae_tgcn_vs_mlp.png, diff_mae_hist.png")
    print("Saved: paired_tests.txt")

if __name__ == "__main__":
    main()
