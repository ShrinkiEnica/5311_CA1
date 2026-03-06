import numpy as np

# We wrap yaw during inference, but does it mess up the denormalization?
# state_raw = state_norm * std + mean
# But if true yaw jumps from -pi to pi, the normalized yaw jumps too.
# However, if the network predicts the wrapped angle, what happens?
# network outputs residual_yaw_rate, we integrate: yaw_next_raw = wrap(yaw_raw + r*dt)
# Then we normalize it back: yaw_next_norm = (yaw_next_raw - mean) / std.
# The network itself receives normalized state!
# Wait, if we wrap the angle to [-pi, pi] but the dataset ranges from -2pi to 0 (as seen in the print),
# then wrap() will shift the angle by 2pi!
# Let's check wrap logic:
def wrap_angle(angle):
    return np.remainder(angle + np.pi, 2 * np.pi) - np.pi

yaw_true = -6.0
print("True yaw:", yaw_true)
print("Wrapped:", wrap_angle(yaw_true))
# This is +0.28!

# The problem is that the dataset yaw angles might be centered around -pi to -2pi.
# If our wrap function maps it to [-pi, pi], there will be a sudden jump of 2pi between the true dataset trajectory and our rolled-out trajectory!
