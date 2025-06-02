import h5py
import pandas as pd

with h5py.File('D:\Study&Work\Study\硕士课程\CN\data\pems-bay-meta.h5', 'r') as f:
    sensor_ids = [id.decode() for id in f['sensor_ids'][:]]  # sensor ID
    coordinates = f['coordinates'][:]

#DataFrame
meta_df = pd.DataFrame(coordinates, columns=['longitude', 'latitude'])
meta_df['sensor_id'] = sensor_ids

print(meta_df.head())
