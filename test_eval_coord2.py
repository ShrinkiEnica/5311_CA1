import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_data/VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
df = df.iloc[::10].reset_index(drop=True)

# Find a point where velocity is high and roughly straight
idx = 500
y = df["yawAngle_rad"].values[idx]
vx = df["Vx_mps"].values[idx]
vy = df["Vy_mps"].values[idx]

print(f"Index {idx}: Yaw = {y:.2f} rad ({np.degrees(y):.0f} deg)")
print(f"Vx: {vx:.2f}, Vy: {vy:.2f}")

E0 = df["posE_m"].values[idx]
E1 = df["posE_m"].values[idx+1]
N0 = df["posN_m"].values[idx]
N1 = df["posN_m"].values[idx+1]

dE = E1 - E0
dN = N1 - N0

print(f"Actual motion over 0.05s: dE={dE:.4f}, dN={dN:.4f}")

dt = 0.05
# Let's test the current formula: E = -sin-cos, N = cos-sin
pred_dE = (-vx * np.sin(y) - vy * np.cos(y)) * dt
pred_dN = (vx * np.cos(y) - vy * np.sin(y)) * dt

print(f"Formula A (Current): dE={pred_dE:.4f}, dN={pred_dN:.4f}")

# The reason it works is because it's EXACTLY correct for the dataset.
# The user's image shows a straight line for the RED plot.
# Let's verify what the model's actual yaw rate prediction is! 
# We need to run a small inference script on the saved model to see if it predicts 0 yaw rate.

