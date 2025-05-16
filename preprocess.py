import h5py

def load_pems_data(filepath='D:\Study&Work\Study\硕士课程\CN\data/pems-bay.h5'):
    with h5py.File(filepath, 'r') as f:
        data = f['speed/block0_values'][:]
    print(f"data shape: {data.shape}")
    return data


from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(data, window_size=12, pred_horizon=1):
    """
    Standardize the data and generate sliding window samples.

    Parameters:
        data (ndarray): The raw traffic data with shape (T, N)
        window_size (int): Number of historical time steps used for prediction
        pred_horizon (int): Number of future steps to predict

    Returns:
        X (ndarray): Input features with shape (num_samples, window_size, N)
        y (ndarray): Targets with shape (num_samples, pred_horizon, N)
        scaler (StandardScaler): Fitted scaler object for inverse transform if needed
    """
    # Apply Z-score normalization to each sensor (column-wise)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Generate samples using a sliding window
    X, y = [], []
    for i in range(len(data_scaled) - window_size - pred_horizon + 1):
        X.append(data_scaled[i: i + window_size])
        y.append(data_scaled[i + window_size: i + window_size + pred_horizon])

    X = np.stack(X)  # Shape: (num_samples, window_size, num_nodes)
    y = np.stack(y)  # Shape: (num_samples, pred_horizon, num_nodes)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scaler

def split_data(X, y, train_ratio=0.7, val_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
        X (ndarray): Input features, shape (samples, window_size, nodes)
        y (ndarray): Targets, shape (samples, pred_horizon, nodes)
        train_ratio (float): Proportion of training data
        val_ratio (float): Proportion of validation data

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    total_samples = X.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


