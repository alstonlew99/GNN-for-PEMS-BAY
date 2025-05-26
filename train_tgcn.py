

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_tgcn import TGCN
from preprocess import load_pems_data, preprocess_data, split_data
from graph_utils import build_knn_adj_matrix,normalize_adj

print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

data = load_pems_data('D:\Study&Work\Study\硕士课程\CN\data\pems-bay.h5')
X, y, scaler = preprocess_data(data, window_size=12, pred_horizon=1)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.squeeze(1), dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.squeeze(1), dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.squeeze(1), dtype=torch.float32)

# Build Datasets
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

#Load adjacency matrix
adj_matrix = build_knn_adj_matrix(r'D:\Study&Work\Study\硕士课程\CN\data\pems-bay-meta.h5', k=5)
adj_matrix = normalize_adj(adj_matrix)
adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TGCN(num_nodes=325, gcn_hidden_dim=128, gru_hidden_dim=128, dropout_rate=0.3).to(device)
adj_tensor = adj_tensor.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch, adj_tensor)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch, adj_tensor)
            val_loss += criterion(y_pred, y_batch).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


    def evaluate_on_test(model, test_loader, criterion, adj_tensor):
        model.eval()
        total_loss = 0.0
        mae_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch, adj_tensor)
                loss = criterion(output, y_batch)
                mae = torch.mean(torch.abs(output - y_batch))
                total_loss += loss.item()
                mae_total += mae.item()
        print(f"Test MSE Loss: {total_loss / len(test_loader):.4f}")
        print(f"Test MAE: {mae_total / len(test_loader):.4f}")


    evaluate_on_test(model, test_loader, criterion, adj_tensor)


    def plot_prediction(model, test_loader, adj_tensor, node_index=0, num_batches=3, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(test_loader):
                if i >= num_batches:
                    break
                X_batch = X_batch.to(device)
                output = model(X_batch, adj_tensor).cpu().numpy()
                y_batch = y_batch.cpu().numpy()
                preds.append(output[:, node_index])
                trues.append(y_batch[:, node_index])

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        plt.figure(figsize=(10, 4))
        plt.plot(trues, label='Ground Truth')
        plt.plot(preds, label='Prediction')
        plt.title(f'T-GCN Prediction vs Ground Truth - Node {node_index}')
        plt.legend()
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        plt.show()


    plot_prediction(
        model=model,
        test_loader=test_loader,
        adj_tensor=adj_tensor,
        node_index=0,
        num_batches=3,
        save_path='D:\Study&Work\Study\硕士课程\CN\Results\\norm_TGCN_node0.png'
    )


