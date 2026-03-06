import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_data/VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
df = df.iloc[::10].reset_index(drop=True)

vx = df["Vx_mps"].values
vy = df["Vy_mps"].values
ax = df["axCG_mps2"].values
ay = df["ayCG_mps2"].values
r = df["yawRate_radps"].values

dt = 0.05
diff_y = ay - vx * r
print("ay - vx*r mean:", np.mean(diff_y))
print("ay - vx*r std:", np.std(diff_y))

actual_d_vy = np.diff(vy) / dt
pred_d_vy = (ay[:-1] - vx[:-1] * r[:-1])

print("actual d_vy mean:", np.mean(actual_d_vy))
print("pred d_vy mean:", np.mean(pred_d_vy))
print("Mean absolute error in d_vy:", np.mean(np.abs(actual_d_vy - pred_d_vy)))

diff_x = ax + vy * r
print("\nax + vy*r mean:", np.mean(diff_x))
actual_d_vx = np.diff(vx) / dt
pred_d_vx = (ax[:-1] + vy[:-1] * r[:-1])
print("Mean absolute error in d_vx:", np.mean(np.abs(actual_d_vx - pred_d_vx)))

