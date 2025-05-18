import h5py

def load_pems_data(filepath='D:\Study&Work\Study\硕士课程\CN\data\pems-bay.h5'):
    with h5py.File(filepath, 'r') as f:
        data = f['speed/block0_values'][:]
    print(f"data shape: {data.shape}")
    return data


from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(data, window_size=12, pred_horizon=1):
    import numpy as np

    # [T, N]
    data = np.array(data)
    num_samples, num_nodes = data.shape

    # Δv = v_t - v_{t-1}
    diff = np.diff(data, axis=0, prepend=data[0:1])

    data_stack = np.stack([data, diff], axis=-1)

    X, y = [], []
    for t in range(num_samples - window_size - pred_horizon + 1):
        x_t = data_stack[t : t + window_size]
        y_t = data[t + window_size + pred_horizon - 1]
        X.append(x_t)
        y.append(y_t)

    y = np.array(y)

    # [B, T, N, 2]
    X = np.array(X)
    mean = X.mean(axis=(0, 1, 2), keepdims=True)
    std = X.std(axis=(0, 1, 2), keepdims=True)
    X = (X - mean) / std

    return X, y, None


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


