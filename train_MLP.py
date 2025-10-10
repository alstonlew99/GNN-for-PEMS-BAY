import numpy as np, torch, h5py
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from preprocess import preprocess_by_time_and_windows
from model_MLP import MLP

torch.manual_seed(2025)
np.random.seed(2025)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load PeMS-Bay data ---
with h5py.File("pems-bay.h5", "r") as f:
    data = np.array(f["speed"])  # [T, N]

L_in, H = 12, 1
prep = preprocess_by_time_and_windows(data, L_in=L_in, H=H, ratios=(0.7, 0.1, 0.2))

Xtr, Ytr = prep["X_train"], prep["Y_train"]
Xva, Yva = prep["X_val"], prep["Y_val"]
Xte, Yte = prep["X_test"], prep["Y_test"]

Btr, Lin, N = Xtr.shape
Xtr = Xtr.reshape(Btr, Lin * N)
Ytr = Ytr[:, 0, :]
Bva = Xva.shape[0]
Xva = Xva.reshape(Bva, Lin * N)
Yva = Yva[:, 0, :]
Bte = Xte.shape[0]
Xte = Xte.reshape(Bte, Lin * N)
Yte = Yte[:, 0, :]

Xtr = torch.tensor(Xtr, dtype=torch.float32)
Ytr = torch.tensor(Ytr, dtype=torch.float32)
Xva = torch.tensor(Xva, dtype=torch.float32)
Yva = torch.tensor(Yva, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
Yte = torch.tensor(Yte, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=128, shuffle=False)

in_dim, out_dim = Lin * N, N
model = MLP(input_dim=in_dim, hidden_dim=256, output_dim=out_dim, dropout=0.3).to(device)
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val, patience, wait = np.inf, 5, 0
epochs = 100

for epoch in range(1, epochs + 1):
    model.train(); tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = crit(pred, yb)
        loss.backward(); opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    model.eval(); val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += crit(pred, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), "mlp_best.pt")
    else:
        wait += 1
        if wait >= patience: break

model.load_state_dict(torch.load("mlp_best.pt", map_location=device))
model.eval(); test_loss, preds, trues = 0.0, [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        test_loss += crit(pred, yb).item() * xb.size(0)
        preds.append(pred.cpu().numpy()); trues.append(yb.cpu().numpy())
test_loss /= len(test_loader.dataset)
y_pred_test = np.concatenate(preds, 0)
y_true_test = np.concatenate(trues, 0)
np.savez("mlp_test_outputs.npz", y_pred=y_pred_test, y_true=y_true_test, split_indices=prep["split_indices"])
print(f"Test MSE: {test_loss:.6f}")
