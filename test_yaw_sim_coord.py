import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_data/VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
df = df.iloc[::10].reset_index(drop=True)
dt = 0.05

vx = df["Vx_mps"].values
vy = df["Vy_mps"].values
yaw = df["yawAngle_rad"].values
posE = df["posE_m"].values
posN = df["posN_m"].values

def wrap(angle):
    return np.remainder(angle + np.pi, 2 * np.pi) - np.pi

# Let's test the current mapping again over the whole trajectory
def simulate_pos(vx, vy, yaw, E0, N0, eq_E, eq_N):
    E_sim = [E0]
    N_sim = [N0]
    for i in range(len(vx) - 1):
        y = yaw[i]
        v_x = vx[i]
        v_y = vy[i]
        
        # Evaluate lambdas
        dE = eq_E(v_x, v_y, y) * dt
        dN = eq_N(v_x, v_y, y) * dt
        
        E_sim.append(E_sim[-1] + dE)
        N_sim.append(N_sim[-1] + dN)
    return np.array(E_sim), np.array(N_sim)

print("Testing mapping equations...")

eqs = {
    "A (Current Bugged)": (
        lambda vx, vy, y: -vx * np.sin(y) - vy * np.cos(y),
        lambda vx, vy, y: vx * np.cos(y) - vy * np.sin(y)
    ),
    "C (E=sin+cos, N=-cos+sin)": (
        lambda vx, vy, y: vx * np.sin(y) + vy * np.cos(y),
        lambda vx, vy, y: -vx * np.cos(y) + vy * np.sin(y)
    )
}

# The previous test output was:
# A (Current bugged E=-sin-cos, N=cos-sin): MAE E: 0.3156, MAE N: 0.4209
# SO E=-vx*sin - vy*cos AND N=vx*cos - vy*sin is ALREADY THE CORRECT ONE!

# WAIT! If A is correct, and I CHANGED IT TO A... what did I change it FROM?
# In previous version it was:
# posE_next = state_raw[..., 0] + (-vx_next * torch.cos(yaw_next) + vy_next * torch.sin(yaw_next)) * DT
# posN_next = state_raw[..., 1] + (-vx_next * torch.sin(yaw_next) - vy_next * torch.cos(yaw_next)) * DT

# But wait, why is the visualization still diverging in a straight line?
# Look at the user's uploaded image. The green diamond is the start. The blue line is Ground Truth (a loop).
# The red line (Dream Rollout) shoots off in a perfectly straight line!
# A straight line means it is NOT turning.
# This means YAW_RATE or YAW is constantly zero, or it's not being updated, or it's predicting zero turning!
# Let's check the trajectory of YAW in the dream output!
