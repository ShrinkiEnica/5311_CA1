import numpy as np

# In evaluation we compute:
# posE_next = state_raw[..., 0] + (-vx_next * torch.sin(yaw_next) - vy_next * torch.cos(yaw_next)) * DT
# Let's plot what E=-sin-cos, N=cos-sin actually looks like if we trace it.

# Suppose we start at (0,0), yaw=0.
# vx = 1 (forward), vy = 0.
# dE = -1*sin(0) - 0*cos(0) = 0
# dN = 1*cos(0) - 0*sin(0) = 1
# So yaw=0 moves NORTH.

# Now yaw = pi/2 (90 deg).
# dE = -1*sin(90) - 0*cos(90) = -1
# dN = 1*cos(90) - 0*sin(90) = 0
# So yaw=90 moves WEST (-East).

# Now yaw = pi (180 deg).
# dE = 0, dN = -1
# Moves SOUTH.

# Now yaw = -pi/2 (-90 deg).
# dE = -1*sin(-90) = 1
# dN = 0
# Moves EAST.

print("This means the coordinate system is X=North, Y=West!")
print("But standard map is X=East, Y=North.")

# So wait... What was "C (E=sin+cos, N=-cos+sin)"?
# dE = 1*sin(0) + 0*cos(0) = 0
# dN = -1*cos(0) + 0*sin(0) = -1
# yaw=0 moves SOUTH.

# Let's verify what the dataset actually does!
import pandas as pd
df = pd.read_csv("data/raw_data/VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
df = df.iloc[::10].reset_index(drop=True)

yaw0 = df["yawAngle_rad"].values[0]
vx0 = df["Vx_mps"].values[0]
vy0 = df["Vy_mps"].values[0]

E_diff = np.diff(df["posE_m"])[:5]
N_diff = np.diff(df["posN_m"])[:5]

print("Yaw[0]:", yaw0)
print("Vx[0]:", vx0, "Vy[0]:", vy0)
print("Actual E_diff[0]:", E_diff[0])
print("Actual N_diff[0]:", N_diff[0])
print("Actual E_diff[1]:", E_diff[1])
print("Actual N_diff[1]:", N_diff[1])
