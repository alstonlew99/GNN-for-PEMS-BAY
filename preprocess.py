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

