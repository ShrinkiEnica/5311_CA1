import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add code/ to path
sys.path.append(os.path.join(os.getcwd(), "code"))
from train_wam_v2 import WAM, load_multi_lap_data, OBS_DIM, ACTION_DIM, DT
import math

device = torch.device("cpu")
data_path = Path("data/raw_data")
segments = load_multi_lap_data(data_path, train_ratio=0.7, downsample=10)
train_segs, val_segs = segments[0], segments[1]

all_s = []
for s, _ in train_segs:
    all_s.append(s)
all_s = np.concatenate(all_s, axis=0)
s_mean = np.mean(all_s, axis=0)
s_std = np.std(all_s, axis=0) + 1e-8
norm = {"s_mean": s_mean, "s_std": s_std}

checkpoint = torch.load("models/wam_v2_best.pt", map_location=device)
config = checkpoint["config"]

model = WAM(
    obs_dim=config.get("obs_dim", OBS_DIM),
    action_dim=config.get("action_dim", ACTION_DIM),
    state_dim=config.get("state_dim", 18),
    d_model=config.get("d_model", 96),
    num_layers=config.get("num_layers", 3),
    ssm_state_dim=96,  # Look at the error size mismatch: gru state dim is 96 instead of 128
    hidden_dim=192,    # Decoder dim is 192 instead of 256
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
a_val_n = a_val

init = torch.tensor(s_val_n[0], dtype=torch.float32).unsqueeze(0).to(device)
acts = torch.tensor(a_val_n[:400], dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred_n = model.dream(init, acts).squeeze(0).cpu().numpy()

pred_raw = pred_n * s_std_ld + s_mean_ld
gt_raw = s_val_n[1:401] * s_std_ld + s_mean_ld

print("Predicted yaw_rate (rad/s) stats:", np.mean(pred_raw[:, 7]), np.std(pred_raw[:, 7]), np.min(pred_raw[:, 7]), np.max(pred_raw[:, 7]))
print("Ground truth yaw_rate (rad/s) stats:", np.mean(gt_raw[:, 7]), np.std(gt_raw[:, 7]), np.min(gt_raw[:, 7]), np.max(gt_raw[:, 7]))

print("\nPredicted yaw max diff from start:", np.max(np.abs(pred_raw[:, 6] - pred_raw[0, 6])))
print("Ground truth yaw max diff from start:", np.max(np.abs(gt_raw[:, 6] - gt_raw[0, 6])))

print("\nPredicted YAW first 10 steps:\n", pred_raw[:10, 6])
print("True YAW first 10 steps:\n", gt_raw[:10, 6])

