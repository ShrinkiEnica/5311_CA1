# The buggy transition logic in train_wam_v2.py:
# vy_next = state_raw[..., 4] + (ay_next - state_raw[..., 3] * yawrate_next) * DT
#
# But earlier when we tested `actual_d_vy` vs `pred_d_vy = ay - vx * r`, the error was 3.3 m/s^2!
# WAIT! The standard physical formula is:
# ay = d(Vy)/dt + Vx * r
# So d(Vy)/dt = ay - Vx * r
# But we found earlier:
# "ay - vx*r mean: 0.0401, std: 4.02"
# "actual d_vy mean: -0.0015"
# "pred d_vy mean: 0.0412"
# The error between `actual_d_vy` and `pred_d_vy` is huge! 3.3!
# This means the dataset's `ay` does NOT strictly equal `d(Vy)/dt + Vx*r`! 
# Why? Because `ayCG_mps2` includes GRAVITY components if the vehicle rolls/banks!
# And it includes sensor noise.
# If we integrate `d(Vy) = (ay - Vx*r)*dt`, and `ay` is slightly off by just 0.1 m/s^2 due to road banking,
# it will accumulate 1 m/s error every 10 seconds.
# But even worse! What if `ay` is not the pure inertial acceleration?

# Let's check what happens if we just let the network predict `residual_map[4]` directly for Vy, instead of using kinematics!
# The network already predicts `ay`. If we force `vy_next` to be integrated from `ay`, we inject massive noise!
# Actually, the residual for Vy is currently not even predicted, or is it?
