import numpy as np
from sklearn.preprocessing import StandardScaler

def chronological_split_by_ratio(data: np.ndarray, ratios=(0.7, 0.1, 0.2)):
    T = data.shape[0]
    t1 = int(T * ratios[0])
    t2 = int(T * (ratios[0] + ratios[1]))
    train_raw = data[:t1]
    val_raw = data[t1:t2]
    test_raw = data[t2:]
    return (train_raw, val_raw, test_raw), ((0, t1), (t1, t2), (t2, T))

def fit_transform_splits(train_raw, val_raw, test_raw):
    scaler = StandardScaler(with_mean=True, with_std=True)
    train = scaler.fit_transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)
    return (train, val, test), scaler

def make_windows(series_2d: np.ndarray, L_in=12, H=1):
    T, N = series_2d.shape
    X, Y = [], []
    for t in range(L_in, T - H + 1):
        X.append(series_2d[t - L_in:t, :])
        Y.append(series_2d[t:t + H, :])
    if not X:
        return np.empty((0, L_in, series_2d.shape[1])), np.empty((0, H, series_2d.shape[1]))
    return np.stack(X), np.stack(Y)

def preprocess_by_time_and_windows(data: np.ndarray, L_in=12, H=1, ratios=(0.7, 0.1, 0.2)):
    (train_raw, val_raw, test_raw), idx = chronological_split_by_ratio(data, ratios)
    (train, val, test), scaler = fit_transform_splits(train_raw, val_raw, test_raw)
    X_tr, Y_tr = make_windows(train, L_in, H)
    X_va, Y_va = make_windows(val, L_in, H)
    X_te, Y_te = make_windows(test, L_in, H)
    return {
        "X_train": X_tr, "Y_train": Y_tr,
        "X_val": X_va, "Y_val": Y_va,
        "X_test": X_te, "Y_test": Y_te,
        "scaler": scaler,
        "split_indices": idx
    }

def compute_node_stats_train_only(data_train: np.ndarray, eps: float = 1e-8):
    """
    data_train: [T_train, N]
    returns node_mean, node_var, node_cv of shape [N]
    """
    mean = np.nanmean(data_train, axis=0)
    var = np.nanvar(data_train, axis=0)
    std = np.sqrt(var + eps)
    cv = std / (np.abs(mean) + eps)
    return mean, var, cv
