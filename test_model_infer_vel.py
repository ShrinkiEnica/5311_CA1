import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "code"))
from train_wam_v2 import WAM, load_multi_lap_data, OBS_DIM, ACTION_DIM, DT

device = torch.device("cpu")
data_path = Path("data/raw_data")
segments = load_multi_lap_data(data_path, train_ratio=0.7, downsample=10)
val_segs = segments[1]

checkpoint = torch.load("models/wam_v2_best.pt", map_location=device)
config = checkpoint["config"]

model = WAM(
    obs_dim=config.get("obs_dim", OBS_DIM),
    action_dim=config.get("action_dim", ACTION_DIM),
    state_dim=config.get("state_dim", 18),
    d_model=config.get("d_model", 96),
    num_layers=config.get("num_layers", 3),
    ssm_state_dim=96,
    hidden_dim=192,
    dropout=0.0
).to(device)

state_dict = checkpoint["model_state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_state_dict[k[len("_orig_mod."):]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.eval()

norm_loaded = checkpoint["norm"]
s_mean_ld = norm_loaded["s_mean"]
s_std_ld = norm_loaded["s_std"]

model.set_normalization(
    torch.tensor(s_mean_ld, dtype=torch.float32, device=device),
    torch.tensor(s_std_ld, dtype=torch.float32, device=device)
)

s_val, a_val = val_segs[0]
s_val_n = (s_val - s_mean_ld) / s_std_ld

init = torch.tensor(s_val_n[0], dtype=torch.float32).unsqueeze(0).to(device)
acts = torch.tensor(a_val[:400], dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred_n = model.dream(init, acts).squeeze(0).cpu().numpy()

pred_raw = pred_n * s_std_ld + s_mean_ld
gt_raw = s_val_n[1:401] * s_std_ld + s_mean_ld

print("First 10 steps Yaw:")
print(pred_raw[:10, 6])
print("First 10 steps Vx:")
print(pred_raw[:10, 3])
print("First 10 steps Vy:")
print(pred_raw[:10, 4])
print("First 10 steps posE:")
print(pred_raw[:10, 0])
print("First 10 steps posN:")
print(pred_raw[:10, 1])

# Calculate dE and dN mathematically from formula
dt = 0.05
yaws = pred_raw[:10, 6]
vxs = pred_raw[:10, 3]
vys = pred_raw[:10, 4]

print("\nComputed dE / dN manually:")
for i in range(9):
    y = yaws[i]
    vx = vxs[i]
    vy = vys[i]
    
    # Bugged formula? Wait!
    # posE_next = state_raw[..., 0] + (-vx_next * torch.sin(yaw_next) - vy_next * torch.cos(yaw_next)) * DT
    
    # In my tests it was the OTHER ONE:
    # A (Current Bugged): MAE E: 0.3156 (matches ground truth perfectly in test_yaw_sim_coord)
    # The actual train_wam_v2.py line 345:
    # posE_next = state_raw[..., 0] + (-vx_next * torch.sin(yaw_next) - vy_next * torch.cos(yaw_next)) * DT
    
    dE = (-vx * np.sin(y) - vy * np.cos(y)) * dt
    dN = (vx * np.cos(y) - vy * np.sin(y)) * dt
    print(f"step {i}: dE={dE:.4f}, dN={dN:.4f}")
    
    actual_dE = pred_raw[i+1, 0] - pred_raw[i, 0]
    actual_dN = pred_raw[i+1, 1] - pred_raw[i, 1]
    print(f"       real_dE={actual_dE:.4f}, real_dN={actual_dN:.4f}")

