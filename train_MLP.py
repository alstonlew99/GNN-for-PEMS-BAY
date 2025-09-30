import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from preprocess import preprocess_by_time_and_windows
from model_MLP import MLPForecast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data: np.ndarray of shape [T, N]
# Replace the following line with your own loader.
data = np.load("pems_speed.npy")  # placeholder

L_in, H = 12, 1
prep = preprocess_by_time_and_windows(data, L_in=L_in, H=H, ratios=(0.7, 0.1, 0.2))

Xtr, Ytr = prep["X_train"], prep["Y_train"]  # [B, L_in, N], [B, H, N]
Xva, Yva = prep["X_val"],   prep["Y_val"]
Xte, Yte = prep["X_test"],  prep["Y_test"]

Btr, Lin, N = Xtr.shape[0], Xtr.shape[1], Xtr.shape[2]
Xtr = Xtr.reshape(Btr, Lin * N)
Ytr = Ytr[:, 0, :]  # [B, N]

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

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=128, shuffle=True, drop_last=False)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=128, shuffle=False, drop_last=False)
test_loader  = DataLoader(TensorDataset(Xte, Yte), batch_size=128, shuffle=False, drop_last=False)

in_dim = Lin * N
out_dim = N
model = MLPForecast(input_dim=in_dim, hidden_dim=256, output_dim=out_dim, dropout=0.3).to(device)  # adjust args if needed
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = np.inf
patience, wait = 5, 0
epochs = 100

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
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
            pred = model(xb)
            loss = crit(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), "mlp_best.pt")
    else:
        wait += 1
        if wait >= patience:
            break

# test
model.load_state_dict(torch.load("mlp_best.pt", map_location=device))
model.eval()
test_loss = 0.0
pred_list, true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        test_loss += loss.item() * xb.size(0)
        pred_list.append(pred.cpu().numpy())
        true_list.append(yb.cpu().numpy())
test_loss /= len(test_loader.dataset)

y_pred_test = np.concatenate(pred_list, axis=0)  # [T_test, N]
y_true_test = np.concatenate(true_list, axis=0)  # [T_test, N]
np.savez("mlp_test_outputs.npz", y_pred=y_pred_test, y_true=y_true_test, split_indices=prep["split_indices"])
print(f"Test MSE: {test_loss:.6f}")
