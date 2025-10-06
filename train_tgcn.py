import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from preprocess import preprocess_by_time_and_windows, chronological_split_by_ratio, compute_node_stats_train_only
from graph_utils import build_knn_adj_matrix, normalize_adj, build_topo_flow_adj
from model_tgcn import TGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data: [T,N], meta: dict or array providing coordinates as meta["coords"] -> [N,2]
data = np.load("pems_speed.npy")
meta = np.load("pems_bay_meta.npy", allow_pickle=True).item()

L_in, H = 12, 1
prep = preprocess_by_time_and_windows(data, L_in=L_in, H=H, ratios=(0.7, 0.1, 0.2))

# choose adjacency
ADJ_TYPE = "knn"          # "knn" | "gaussian_threshold" | "topo_flow"
K = 5
SIGMA = 0.1
ALPHA = 0.5
BETA = 0.5
DIST_THRESH = 0.3  # only if you implement gaussian_threshold elsewhere

# X/Y tensors
Xtr, Ytr = prep["X_train"], prep["Y_train"]
Xva, Yva = prep["X_val"],   prep["Y_val"]
Xte, Yte = prep["X_test"],  prep["Y_test"]
Xtr = Xtr[..., None]; Xva = Xva[..., None]; Xte = Xte[..., None]
Xtr = torch.tensor(Xtr, dtype=torch.float32); Ytr = torch.tensor(Ytr[:,0,:], dtype=torch.float32)
Xva = torch.tensor(Xva, dtype=torch.float32); Yva = torch.tensor(Yva[:,0,:], dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32); Yte = torch.tensor(Yte[:,0,:], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=64, shuffle=False)

# adjacency
if ADJ_TYPE == "knn":
    A = build_knn_adj_matrix(meta, k=K)
elif ADJ_TYPE == "topo_flow":
    coords = meta["coords"]  # shape [N,2]
    knn_mask = (build_knn_adj_matrix(meta, k=K) > 0).astype(np.float32)
    (train_raw, _, _), _ = chronological_split_by_ratio(data, ratios=(0.7, 0.1, 0.2))
    node_mean, node_var, node_cv = compute_node_stats_train_only(train_raw)
    A = build_topo_flow_adj(coords_xy=coords,
                            topo_mask=knn_mask,
                            node_mean=node_mean,
                            node_var=node_var,
                            node_cv=node_cv,
                            sigma=SIGMA, alpha=ALPHA, beta=BETA)
else:
    raise ValueError("Unsupported ADJ_TYPE")

A_norm = normalize_adj(A)
np.save("adj_used.npy", A)          # raw weights for analysis
np.save("adj_used_norm.npy", A_norm)

A_t = torch.tensor(A_norm, dtype=torch.float32, device=device)

num_nodes = Ytr.shape[1]
model = TGCN(num_nodes=num_nodes, input_dim=1, hidden_dim=128, output_dim=1).to(device)
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = np.inf
patience, wait = 5, 0
epochs = 50

for epoch in range(1, epochs + 1):
    model.train(); train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb, A_t).squeeze(-1)
        loss = crit(pred, yb)
        loss.backward(); opt.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval(); val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, A_t).squeeze(-1)
            val_loss += crit(pred, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val:
        best_val = val_loss; wait = 0
        torch.save(model.state_dict(), "tgcn_best.pt")
    else:
        wait += 1
        if wait >= patience:
            break

# test
model.load_state_dict(torch.load("tgcn_best.pt", map_location=device))
model.eval(); test_loss = 0.0; pred_list, true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb, A_t).squeeze(-1)
        test_loss += crit(pred, yb).item() * xb.size(0)
        pred_list.append(pred.cpu().numpy()); true_list.append(yb.cpu().numpy())
test_loss /= len(test_loader.dataset)
y_pred_test = np.concatenate(pred_list, axis=0)
y_true_test = np.concatenate(true_list, axis=0)
np.savez("tgcn_test_outputs.npz", y_pred=y_pred_test, y_true=y_true_test, split_indices=prep["split_indices"])
print(f"Test MSE: {test_loss:.6f}")
