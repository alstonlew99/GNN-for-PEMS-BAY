from preprocess import load_pems_data, preprocess_data, split_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_MLP import MLPForecast
import matplotlib.pyplot as plt
import numpy as np

#Load and preprocess data
data = load_pems_data('D:\Study&Work\Study\硕士课程\CN\data\pems-bay.h5')
X, y, scaler = preprocess_data(data, window_size=12, pred_horizon=1)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

#Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.squeeze(1), dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.squeeze(1), dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.squeeze(1), dtype=torch.float32)

#Build DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = MLPForecast(input_len=12, num_nodes=325)

# Instantiate model
model = MLPForecast(input_len=12, num_nodes=325)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)  # shape: [batch, 325]
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_output = model(X_val_batch)
            loss = criterion(val_output, y_val_batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


#Evaluate on test set
def evaluate_on_test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    mae_total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            mae = torch.mean(torch.abs(output - y_batch))

            total_loss += loss.item()
            mae_total += mae.item()

    print(f"\nTest MSE Loss: {total_loss / len(test_loader):.4f}")
    print(f"Test MAE: {mae_total / len(test_loader):.4f}")

# Run test evaluation
evaluate_on_test(model, test_loader, criterion)

def plot_prediction(model, test_loader, node_index=0, num_batches=1, save_path=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            if i >= num_batches:
                break
            X_batch = X_batch.to(device)
            output = model(X_batch).cpu().numpy()
            y_batch = y_batch.squeeze(1).numpy()

            preds.append(output[:, node_index])
            trues.append(y_batch[:, node_index])
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    plt.figure(figsize=(10, 4))
    plt.plot(trues, label='Ground Truth')
    plt.plot(preds, label='Prediction')
    plt.title(f'Prediction vs Ground Truth - Node {node_index}')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    plt.show()

# Plot at Node 0 and save the plot
plot_prediction(
    model=model,
    test_loader=test_loader,
    node_index=0,
    num_batches=3,
    save_path='D:\Study&Work\Study\硕士课程\CN\Results\\node0_prediction.png'
)
