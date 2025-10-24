import numpy as np, torch, pickle, pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from preprocess import preprocess_by_time_and_windows, chronological_split_by_ratio, compute_node_stats_train_only
from graph_utils import build_knn_adj_matrix, normalize_adj, build_topo_flow_adj
from model_tgcn import TGCN

torch.manual_seed(2025)
np.random.seed(2025)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load PeMS-Bay data (pandas HDF) ---
data_df = pd.read_hdf("pems-bay.h5", key="speed")
data = data_df.values  # [T, N]
print("data shape:", data.shape)

# --- load meta info and align order to data columns ---
sensor_cols = data_df.columns.astype(int).tolist()
meta_df = pd.read_hdf("pems-bay-meta.h5", key="meta")
assert set(sensor_cols) == set(meta_df.index.astype(int).tolist())
meta_df = meta_df.loc[sensor_cols]
lat = meta_df["Latitude"].values
lon = meta_df["Longitude"].values
meta_coords = np.stack([lat, lon], axis=1)
meta_ids = np.array(sensor_cols, dtype=int)
meta = {"coords": meta_coords, "ids": meta_ids}
print("meta coords:", meta_coords.shape)


# --- load precomputed adjacency (optional) ---
with open("adj_mx_bay.pkl", "rb") as f:
    adj_precomputed = pickle.load(f, encoding="latin1")

# --- preprocessing ---
L_in, H = 24, 1
prep = preprocess_by_time_and_windows(data, L_in=L_in, H=H, ratios=(0.7, 0.1, 0.2))
Xtr, Ytr = prep["X_train"], prep["Y_train"]
Xva, Yva = prep["X_val"], prep["Y_val"]
Xte, Yte = prep["X_test"], prep["Y_test"]

# no extra channel dimension here
Xtr = torch.tensor(Xtr, dtype=torch.float32)
Ytr = torch.tensor(Ytr[:, 0, :], dtype=torch.float32)
Xva = torch.tensor(Xva, dtype=torch.float32)
Yva = torch.tensor(Yva[:, 0, :], dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
Yte = torch.tensor(Yte[:, 0, :], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=64, shuffle=False)

# --- adjacency: topo + flow (train-only stats), fallback to KNN topo ---
ADJ_TYPE = "topo_flow"  # enable topo+flow by default
K = 5
SIGMA, ALPHA, BETA = 0.1, 0.5, 0.5

# topo mask from precomputed adjacency if shape matches, else KNN mask
try:
    with open("adj_mx_bay.pkl", "rb") as f:
        adj_precomputed = pickle.load(f, encoding="latin1")
    if adj_precomputed.shape[0] == len(meta_ids):
        topo_mask = (adj_precomputed > 0).astype(np.float32)
    else:
        topo_mask = (build_knn_adj_matrix(meta, k=K) > 0).astype(np.float32)
except Exception:
    topo_mask = (build_knn_adj_matrix(meta, k=K) > 0).astype(np.float32)

(train_raw, _, _), _ = chronological_split_by_ratio(data, ratios=(0.7, 0.1, 0.2))
mean, var, cv = compute_node_stats_train_only(train_raw)
A = build_topo_flow_adj(coords_xy=meta_coords,
                        topo_mask=topo_mask,
                        node_mean=mean,
                        node_var=var,
                        node_cv=cv,
                        sigma=SIGMA, alpha=ALPHA, beta=BETA)


A_norm = normalize_adj(A)
np.save("adj_used.npy", A)
np.save("adj_used_norm.npy", A_norm)
A_t = torch.tensor(A_norm, dtype=torch.float32, device=device)

# --- model ---
num_nodes = Ytr.shape[1]
model = TGCN(num_nodes=num_nodes, in_dim=1, gcn_hidden_dim=256, gru_hidden_dim=256, dropout_rate=0.3).to(device)
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val, patience, wait = np.inf, 10, 0
epochs = 50

for epoch in range(1, epochs + 1):
    model.train(); tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb, A_t)        # shape [B, N]
        loss = crit(pred, yb)
        loss.backward(); opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    model.eval(); val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, A_t)
            val_loss += crit(pred, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:03d}: train={tr_loss:.6f}, val={val_loss:.6f}")

    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), "tgcn_best.pt")
    else:
        wait += 1
        if wait >= patience: break

# --- test ---
model.load_state_dict(torch.load("tgcn_best.pt", map_location=device))
model.eval(); test_loss, preds, trues = 0.0, [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb, A_t)
        test_loss += crit(pred, yb).item() * xb.size(0)
        preds.append(pred.cpu().numpy()); trues.append(yb.cpu().numpy())
test_loss /= len(test_loader.dataset)
y_pred_test = np.concatenate(preds, 0)
y_true_test = np.concatenate(trues, 0)
np.savez("tgcn_test_outputs.npz", y_pred=y_pred_test, y_true=y_true_test, split_indices=prep["split_indices"])
print(f"Test MSE: {test_loss:.6f}")
