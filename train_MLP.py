from preprocess import load_pems_data, preprocess_data, split_data
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_MLP import MLPForecast

#Load and preprocess data
data = load_pems_data('data/pems-bay.h5')
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