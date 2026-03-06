import torch
import numpy as np

# Let's inspect the saved best model to see what it predicts for yaw rate when dreaming
# Actually, the user's issue might be that the model just predicts 0 yaw rate!
# Why would it predict 0 yaw rate? 
# Maybe YAWRATE_IDX is not in the rollout loss, or it's overpowered?
# DYN_LOSS_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# 7 is YAWRATE_IDX.

# Wait, check how state_raw is mapped back into components.
# In `_apply_transition`:
# yaw_next = self._wrap_angle(state_raw[..., YAW_IDX] + yawrate_next * DT)
# ...
# elif idx == YAW_IDX:
#     components.append(yaw_next.unsqueeze(-1))
# ...
# So YAW_IDX (6) is appended.
# But what about the input to the next step?
# The network expects observations.
# OBS_STATE_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# Wait, OBS_STATE_INDICES does NOT include YAW_IDX!
# It includes 7 (YAWRATE).
# BUT in _build_obs:
# yaw = state_raw[..., YAW_IDX:YAW_IDX+1]
# dyn = state[..., OBS_STATE_INDICES]
# return torch.cat([dyn, torch.sin(yaw), torch.cos(yaw)], dim=-1)

# Is state passed to _build_obs normalized or unnormalized?
# `dyn = state[..., OBS_STATE_INDICES]` -> `state` is normalized.
# But for `yaw`, it uses `state_raw`!
# `state_raw = self._denorm_state(state)`
# `state_raw` is denormalized.
# However, during rollout, `state` comes from `pred`.
# `pred, hidden = model.step(state, actions[:, t, :], hidden)`
# `pred` is the NEXT NORMALIZED STATE returned by `_apply_transition`!
# So `state` is perfectly normalized.

# What if there's a bug in _build_obs where the sin/cos yaw are totally wrong?
# No, sin/cos of raw yaw in radians is completely fine.
print("Script parsed")
