import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_data/VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
df = df.iloc[::10].reset_index(drop=True)

vy = df["Vy_mps"].values
ay = df["ayCG_mps2"].values

actual_d_vy = np.diff(vy) / 0.05
print("Mean absolute error in d_vy (simple ay):", np.mean(np.abs(actual_d_vy - ay[:-1])))

