# The current model does not predict Vx, Vy directly.
# RESIDUAL_TARGET_INDICES = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# 3 is Vx, 4 is Vy. They are missing!
# This means the model Relies 100% on the kinematic formulas:
# vx_next = vx + (ax + vy*r)*dt
# vy_next = vy + (ay - vx*r)*dt
# AND IT CLIPS THEM!
# vx_next = self._clip_state(vx_next, 3)
# vy_next = self._clip_state(vy_next, 4)

# Wait! The earlier error: "ay - vx*r" has an error of 3.3 m/s^2 compared to actual d(vy)/dt.
# That means if we integrate vy using ay, it diverges instantly.
# And then it hits the clip bound!
# That explains why in the rollout Vy becomes exactly 3.44436624 for 5 steps in a row!
# The network cannot track velocity properly because the physics model is violated by sensor noise/gravity components in ay.

# To fix this, we MUST let the network predict Vx and Vy residuals directly, just like it does for Vz (idx 5)!
