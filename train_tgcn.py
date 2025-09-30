import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from preprocess import preprocess_by_time_and_windows
from graph_utils import build_knn_adj_matrix, normalize_adj
from model_tgcn import TGCN  # adjust if your class name differs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data: np.ndarray of shape [T, N]
# meta: metadata used to compute inter-sensor distances (as in your project)
# Replace the following two lines with your own loaders.
data = np.load("pems_speed.npy")        # placeholder
meta = np.load("pems_bay_meta.npy", allow_pickle=True).item()  # placeholder

L_in, H = 12, 1
prep = preprocess_by_time_and_windows(data, L_in=L_in, H=H, ratios=(0.7, 0.1, 0.2))

Xtr, Ytr = prep["X_train"], prep["Y_train"]  # [B, L_in, N], [B, H, N]
Xva, Yva = prep["X_val"],   prep["Y_val"]
Xte, Yte = prep["X_test"],  prep["Y_test"]


Xtr = Xtr[..., None]
Xva = Xva[..., None]
Xte = Xte[..., None]

# convert to torch
Xtr = torch.tensor(Xtr, dtype=torch.float32)
Ytr = torch.tensor(Ytr[:, 0, :], dtype=torch.float32)  # H=1 -> [B, N]
Xva = torch.tensor(Xva, dtype=torch.float32)
Yva = torch.tensor(Yva[:, 0, :], dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
Yte = torch.tensor(Yte[:, 0, :], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True, drop_last=False)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=64, shuffle=False, drop_last=False)
test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=64, shuffle=False, drop_last=False)

# adjacency for training graph
A = build_knn_adj_matrix(meta, k=5)
A = normalize_adj(A)
A = torch.tensor(A, dtype=torch.float32, device=device)

num_nodes = Ytr.shape[1]
model = TGCN(num_nodes=num_nodes, input_dim=1, hidden_dim=128, output_dim=1).to(device)  # adjust args if needed
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = np.inf
patience, wait = 5, 0
epochs = 50

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)        # [B, L_in, N, 1]
        yb = yb.to(device)        # [B, N]
        opt.zero_grad()
        pred = model(xb, A).squeeze(-1)  # expected [B, N]
        loss = crit(pred, yb)
        loss.backward()
        opt.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb, A).squeeze(-1)
            loss = crit(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), "tgcn_best.pt")
    else:
        wait += 1
        if wait >= patience:
            break

# test
model.load_state_dict(torch.load("tgcn_best.pt", map_location=device))
model.eval()
test_loss = 0.0
pred_list, true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb, A).squeeze(-1)
        loss = crit(pred, yb)
        test_loss += loss.item() * xb.size(0)
        pred_list.append(pred.cpu().numpy())
        true_list.append(yb.cpu().numpy())
test_loss /= len(test_loader.dataset)

y_pred_test = np.concatenate(pred_list, axis=0)  # [T_test, N]
y_true_test = np.concatenate(true_list, axis=0)  # [T_test, N]
np.savez("tgcn_test_outputs.npz", y_pred=y_pred_test, y_true=y_true_test, split_indices=prep["split_indices"])
print(f"Test MSE: {test_loss:.6f}")
