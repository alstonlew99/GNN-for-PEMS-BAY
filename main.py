from preprocess import load_pems_data, preprocess_data

data = load_pems_data()

X, y, scaler = preprocess_data(data, window_size=12, pred_horizon=1)