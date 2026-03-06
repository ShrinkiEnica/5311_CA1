"""
WAM v2 — World Action Model with Temporal Dynamics

升级要点:
  1. 时序动力学后端: 纯净 PyTorch GRU 避免显存瓶颈与算子兼容问题
  2. BPTT Rollout 训练: 修复长时自回归 Rollout 的梯度截断，使得模型真正学习长期一致性
  3. 动力学物理先验注入: Attitude-aware Gravity Compensation & 显式积分(ax->vx, vz->posU)
  4. 多段验证与批量生图: 训练结束后自动产出多圈 Lap 验证段的 3D/Topdown/Velocity 对比图
  5. Curriculum Learning: 从 5 步渐进增长到 30 步的多步自回归惩罚
  6. Delta Loss 监督: 对动态状态计算预测变化率 (Δstate) 的损失
  7. 动作干预可视化: 验证不同方向盘/刹车输入下的未来轨迹响应
  8. 多圈数据训练: 加载多 lap 赛道数据，拆分验证段

Full State (18 dim):
  posE_m, posN_m, posU_m          — 位置 (3)
  Vx_mps, Vy_mps, Vz_mps          — 车体速度 (3)
  yawAngle_rad                    — 偏航角 (1)
  yawRate_radps                   — 偏航角速度 (1)
  rollRate_radps, pitchRate_radps — 横滚/俯仰角速度 (2)
  axCG_mps2, ayCG_mps2, azCG_mps2 — 纵/横/垂向加速度 (3)
  slipAngle_rad                   — 侧滑角 (1)
  wheelspeed_fl/fr/rl/rr          — 四轮轮速 (4)

Observation (13 dim) — Obs Encoder 输入:
  动态状态子集 (11) + sin(yaw) + cos(yaw)

Action (6 dim):
  roadWheelAngle_rad              — 转向角 (1)
  throttleCmd_percent             — 油门指令 (1)
  brakeCmd_fl/fr/rl/rr_bar        — 四轮制动指令 (4)

Prediction Output:
  下一时刻完整状态；预测基于 GRU 的隐状态 (h_t) 和当前控制 (a_t)
  重点评估三维位置漂移 (dist_3d) 与三维速度匹配 (Vx/Vy/Vz)

Transition:  s_{t+1} = Transition(s_t, obs_t, a_t, h_t)
Dream:       给定初始状态 + 动作序列, 自回归推演未来轨迹

用法:
  python code/train_wam_v2.py
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from mamba_ssm import Mamba as MambaLayer
    MAMBA_AVAILABLE = True
except ImportError:
    MambaLayer = None
    MAMBA_AVAILABLE = False

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_data"
MODEL_DIR = ROOT / "models"
VIS_DIR = ROOT / "visualization" / "wam_v2"

# Laps to exclude (partial data)
EXCLUDE_LAPS = {"_5", "_10"}

# ─── Feature Definitions ─────────────────────────────────────────────────────

STATE_COLS = [
    "posE_m", "posN_m", "posU_m",       # position (3)        [0,1,2]
    "Vx_mps", "Vy_mps", "Vz_mps",
    "yawAngle_rad",
    "yawRate_radps",
    "rollRate_radps",
    "pitchRate_radps",
    "axCG_mps2", "ayCG_mps2", "azCG_mps2",
    "slipAngle_rad",
    "wheelspeed_fl", "wheelspeed_fr",
    "wheelspeed_rl", "wheelspeed_rr",
]
STATE_DIM = len(STATE_COLS)  # 18

ACTION_COLS = [
    "roadWheelAngle_rad",               # steering (1)
    "throttleCmd_percent",              # throttle (1)
    "brakeCmd_fl_bar",                  # brake commands (4)
    "brakeCmd_fr_bar",
    "brakeCmd_rl_bar",
    "brakeCmd_rr_bar",
]
ACTION_DIM = len(ACTION_COLS)  # 6

POS_INDICES = [0, 1, 2]  # position indices in state vector
VEL_INDICES = [3, 4, 5]  # velocity indices in state vector
VZ_IDX = 5
YAW_IDX = 6
YAWRATE_IDX = 7
SLIP_IDX = 13
OBS_STATE_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
OBS_INDICES = OBS_STATE_INDICES
OBS_DIM = len(OBS_STATE_INDICES) + 2
ACC_INDICES = [10, 11, 12]
RESIDUAL_TARGET_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
DYN_LOSS_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
TRAJ_LOSS_INDICES = [0, 1, 2, YAW_IDX]
ROLLOUT_DYN_LOSS_INDICES = [3, 4, 5, YAWRATE_IDX, SLIP_IDX]
ROLLOUT_DYN_LOSS_WEIGHTS = {
    3: 0.5,
    4: 1.0,
    5: 2.5,
    YAWRATE_IDX: 2.0,
    SLIP_IDX: 2.0,
}

# ─── Config ──────────────────────────────────────────────────────────────────

DOWNSAMPLE = 10         # 200Hz -> 20Hz (dt = 0.05s)
DT = 0.005 * DOWNSAMPLE # 0.05s per step
TRAIN_RATIO = 0.7
SEQ_LEN = 64            # training sequence length (3.2s @ 20Hz)
BATCH_SIZE = 512
EPOCHS = 2000
LR = 3e-4
WEIGHT_DECAY = 1e-5
D_MODEL = 96            # ssm model dimension
HIDDEN_DIM = 192        # decoder hidden dimension
SSM_LAYERS = 3          # stacked SSM layers
SSM_STATE_DIM = 96      # internal recurrent state dimension
DROPOUT = 0.1
# LR warmup for temporal model
WARMUP_EPOCHS = 10      # linear warmup for first 10 epochs
POS_WEIGHT_VAL = 10.0   # position loss weight
VEL_WEIGHT_VAL = 5.0    # velocity loss weight
DREAM_STEPS = 400       # dream rollout length for evaluation (20s @ 20Hz)

# Curriculum Learning: rollout steps grow during training
ROLLOUT_MIN = 5         # start with 5-step rollout (0.25s)
ROLLOUT_MAX = 30        # grow to 30-step rollout (1.5s)
ROLLOUT_GROW_START = 20
ROLLOUT_GROW_END = 100  # reach max rollout by epoch 100
ROLLOUT_WEIGHT = 0.15
ROLLOUT_EVERY_N = 10
ROLLOUT_DYN_WEIGHT = 0.06
TRAJ_LOSS_W_START = 0.2
TRAJ_LOSS_W_END = 0.8
TRAJ_LOSS_RAMP_END = 100
STATE_CLIP_MARGIN = 0.1

# Dream validation (for monitoring, NOT early stopping)
DREAM_VAL_STEPS = 400   # 20s dream for validation metric (match eval)
DREAM_VAL_EVERY = 5     # evaluate dream every N epochs
EARLY_STOP_DREAM_ERR_THRESHOLD = 8.0 # Stop training if dream error falls below this in meters
SAVE_EVERY = 100        # save checkpoint every N epochs
DREAM_PLOT_SEGMENTS = 4 # number of validation segments to visualize after training

# Cosine annealing with warm restarts (after warmup)
COSINE_T0 = 100         # first cycle length
COSINE_TMULT = 2        # cycle length multiplier

GRAVITY_MPS2 = 9.81
GRAVITY_COMPENSATION_RATIO = 0.15
MAX_GRAVITY_TILT_RAD = 0.25
VERTICAL_VEL_DAMPING = 1.0

TWO_PI = 2.0 * math.pi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding window dataset for sequential WAM training."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, seq_len: int):
        # Keep data on CPU, let DataLoader pin_memory handle it asynchronously
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.seq_len = seq_len
        self.n_samples = len(states) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = self.states[idx: idx + self.seq_len]        # [T, state_dim]
        a = self.actions[idx: idx + self.seq_len]        # [T, action_dim]
        s_next = self.states[idx + 1: idx + self.seq_len + 1]  # [T, state_dim]
        return s, a, s_next


# ─── Model ───────────────────────────────────────────────────────────────────

class GRUBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=state_dim, batch_first=True)
        self.out_proj = nn.Linear(state_dim, d_model)
        self.residual = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h=None):
        x_norm = self.norm(x)
        out, h_new = self.gru(x_norm, h)
        y = self.out_proj(out)
        y = self.dropout(y)
        return self.residual(x) + y, h_new

    def step(self, x, h=None):
        x_in = x
        x_norm = self.norm(x).unsqueeze(1)
        out, h_new = self.gru(x_norm, h)
        y = self.out_proj(out.squeeze(1))
        y = self.dropout(y)
        return self.residual(x_in) + y, h_new


class WAM(nn.Module):
    """
    World Action Model v2 — State Space Model Architecture

    Architecture:
      Obs Encoder:    obs → d_model
      Action Encoder: action → d_model
      Temporal:       stacked SSM blocks
      State Decoder:  d_model → Δstate
      Transition:     s_{t+1} = s_t + Δstate
    """

    def __init__(self, state_dim: int, action_dim: int,
                 obs_dim: int = 12,
                 d_model: int = 128, num_layers: int = 4,
                 ssm_state_dim: int = 128,
                 hidden_dim: int = 256, dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.ssm_state_dim = ssm_state_dim
        self.residual_dim = len(RESIDUAL_TARGET_INDICES)
        self.temporal_backend = "gru"

        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))
        self.register_buffer("s_min", torch.full((state_dim,), -1e9))
        self.register_buffer("s_max", torch.full((state_dim,), 1e9))

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.temporal = nn.ModuleList([
            GRUBlock(d_model=d_model, state_dim=ssm_state_dim, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        self.decoder_fc1 = nn.Linear(d_model, hidden_dim)
        self.decoder_ln = nn.LayerNorm(hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, self.residual_dim)
        self.decoder_drop = nn.Dropout(dropout)

        nn.init.zeros_(self.decoder_out.bias)
        nn.init.uniform_(self.decoder_out.weight, -0.001, 0.001)

    def _decode(self, z):
        """Decode latent to Δstate with residual connection."""
        h = self.decoder_ln(torch.nn.functional.gelu(self.decoder_fc1(z)))
        h = h + torch.nn.functional.gelu(self.decoder_fc2(h))  # skip
        h = self.decoder_drop(h)
        return self.decoder_out(h)

    def set_normalization(self, s_mean, s_std, s_min=None, s_max=None):
        self.s_mean.copy_(torch.as_tensor(s_mean, dtype=torch.float32))
        self.s_std.copy_(torch.as_tensor(s_std, dtype=torch.float32))
        if s_min is not None and s_max is not None:
            self.s_min.copy_(torch.as_tensor(s_min, dtype=torch.float32))
            self.s_max.copy_(torch.as_tensor(s_max, dtype=torch.float32))

    def _clip_state(self, value, idx):
        low = self.s_min[idx]
        high = self.s_max[idx]
        span = torch.clamp(high - low, min=1e-6)
        margin = span * STATE_CLIP_MARGIN
        return torch.clamp(value, min=low - margin, max=high + margin)

    def _denorm_state(self, state):
        return state * self.s_std + self.s_mean

    def _norm_state(self, state):
        return (state - self.s_mean) / self.s_std

    def _build_obs(self, state):
        state_raw = self._denorm_state(state)
        yaw = state_raw[..., YAW_IDX:YAW_IDX+1]
        dyn = state[..., OBS_STATE_INDICES]
        return torch.cat([dyn, torch.sin(yaw), torch.cos(yaw)], dim=-1)

    def _wrap_angle(self, angle):
        return torch.remainder(angle + math.pi, TWO_PI) - math.pi

    def _apply_transition(self, state, residual):
      state_raw = self._denorm_state(state)
      residual_map = {
          idx: state_raw[..., idx] + residual[..., i]
          for i, idx in enumerate(RESIDUAL_TARGET_INDICES)
      }

      ax_next = residual_map[10]
      ay_next = residual_map[11]
      az_next = residual_map[12]
      yawrate_next = residual_map[YAWRATE_IDX]

      # Velocity states: we directly predict the residual \Delta V because kinematic
      # integration (vy_next = vy + (ay - vx*r)*dt) accumulates massive error due to sensor noise and gravity components in IMU.
      vx_next = residual_map[3]
      vy_next = residual_map[4]
      vz_next = residual_map[5]

      ax_next = self._clip_state(ax_next, 10)
      ay_next = self._clip_state(ay_next, 11)
      az_next = self._clip_state(az_next, 12)
      vx_next = self._clip_state(vx_next, 3)
      vy_next = self._clip_state(vy_next, 4)
      vz_next = self._clip_state(vz_next, VZ_IDX)
      yawrate_next = self._clip_state(yawrate_next, YAWRATE_IDX)
      yaw_next = self._wrap_angle(state_raw[..., YAW_IDX] + yawrate_next * DT)
      posE_next = state_raw[..., 0] + (-vx_next * torch.sin(yaw_next) - vy_next * torch.cos(yaw_next)) * DT
      posN_next = state_raw[..., 1] + (vx_next * torch.cos(yaw_next) - vy_next * torch.sin(yaw_next)) * DT
      posU_next = state_raw[..., 2] + 0.5 * (state_raw[..., VZ_IDX] + vz_next) * DT
      residual_map[13] = self._clip_state(residual_map[13], 13)
      for idx in [14, 15, 16, 17]:
          residual_map[idx] = self._clip_state(residual_map[idx], idx)

      components = []
      for idx in range(self.state_dim):
          if idx == 0:
              components.append(posE_next.unsqueeze(-1))
          elif idx == 1:
              components.append(posN_next.unsqueeze(-1))
          elif idx == 2:
              components.append(posU_next.unsqueeze(-1))
          elif idx == 3:
              components.append(vx_next.unsqueeze(-1))
          elif idx == 4:
              components.append(vy_next.unsqueeze(-1))
          elif idx == VZ_IDX:
              components.append(vz_next.unsqueeze(-1))
          elif idx == YAW_IDX:
              components.append(yaw_next.unsqueeze(-1))
          elif idx in residual_map:
              components.append(residual_map[idx].unsqueeze(-1))
          else:
              components.append(state_raw[..., idx:idx+1])

      next_raw = torch.cat(components, dim=-1)
      return self._norm_state(next_raw)

    def _init_hidden(self, batch_size, device):
        return [None for _ in range(self.num_layers)]

    def forward(self, states, actions, hidden=None):
        """
        Teacher-forcing forward pass.
        states:  [B, T, state_dim]  — ground truth full states
        actions: [B, T, action_dim]
        Returns: predicted s_{t+1} [B, T, state_dim], hidden
        """
        obs = self._build_obs(states)
        z = self.obs_encoder(obs) + self.action_encoder(actions)
        B = z.size(0)
        if hidden is None:
            hidden = self._init_hidden(B, z.device)

        new_hidden = []
        for layer, h in zip(self.temporal, hidden):
            z, h_new = layer(z, h)
            new_hidden.append(h_new)

        residual = self._decode(z)
        next_states = self._apply_transition(states, residual)
        return next_states, new_hidden

    def step(self, state, action, hidden=None):
        """
        Single autoregressive step.
        """
        obs = self._build_obs(state)
        z = self.obs_encoder(obs) + self.action_encoder(action)
        if hidden is None:
            hidden = self._init_hidden(z.size(0), z.device)

        new_hidden = []
        for layer, h in zip(self.temporal, hidden):
            z, h_new = layer.step(z, h)
            new_hidden.append(h_new)

        residual = self._decode(z)
        next_state = self._apply_transition(state, residual)
        return next_state, new_hidden

    @torch.no_grad()
    def dream(self, init_state, action_seq, hidden=None):
        """
        Autoregressive trajectory imagination.
        init_state: [B, state_dim]
        action_seq: [B, T, action_dim]
        Returns:    [B, T, state_dim] predicted trajectory
        """
        self.eval()
        B, T, _ = action_seq.shape
        predictions = []
        state = init_state

        for t in range(T):
            state, hidden = self.step(state, action_seq[:, t, :], hidden)
            predictions.append(state)

        return torch.stack(predictions, dim=1)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_multi_lap_data(raw_dir: Path, train_ratio: float, downsample: int = 1):
    """
    Load all complete laps, downsample, split each lap 7:3 by time.
    Returns training/validation segments as lists of (state, action) arrays,
    plus validation metadata and concatenated arrays for normalization.
    """
    files = sorted(raw_dir.glob("VehicleDynamicsDataset_Nov2023_2023-11*.csv"))

    train_segments = []  # list of (states, actions) arrays
    val_segments = []
    val_segment_names = []
    all_train_s, all_train_a = [], []

    for f in files:
        suffix = f.stem.replace("VehicleDynamicsDataset_Nov2023_2023-11", "")
        if suffix in EXCLUDE_LAPS:
            print(f"    skip {f.name} (partial lap)")
            continue
        name = f"Lap{suffix}" if suffix else "Lap_0"
        df = pd.read_csv(f, comment="#")
        # Downsample
        if downsample > 1:
            df = df.iloc[::downsample].reset_index(drop=True)
        source_state_cols = [
            "posE_m", "posN_m", "posU_m",
            "Vx_mps", "Vy_mps", "Vz_mps",
            "yawAngle_rad",
            "yawRate_radps",
            "rollRate_radps",
            "pitchRate_radps",
            "axCG_mps2", "ayCG_mps2", "azCG_mps2",
            "slipAngle_rad",
            "wheelspeed_fl", "wheelspeed_fr",
            "wheelspeed_rl", "wheelspeed_rr",
        ]
        s_src = df[source_state_cols].values.astype(np.float64)
        s = np.zeros((len(df), STATE_DIM), dtype=np.float64)
        s[:, 0:6] = s_src[:, 0:6]
        s[:, YAW_IDX:] = s_src[:, 6:]
        a = df[ACTION_COLS].values.astype(np.float64)
        s[:, YAW_IDX] = (s[:, YAW_IDX] + np.pi) % (2 * np.pi) - np.pi
        n = len(df)
        split = int(n * train_ratio)
        train_segments.append((s[:split], a[:split]))
        val_segments.append((s[split:], a[split:]))
        val_segment_names.append(name)
        all_train_s.append(s[:split])
        all_train_a.append(a[:split])
        print(f"    {name}: {n} pts (ds={downsample}) -> train={split}, val={n-split}")

    # Compute normalization from all training data
    all_s = np.concatenate(all_train_s, axis=0)
    all_a = np.concatenate(all_train_a, axis=0)
    s_mean, s_std = all_s.mean(axis=0), all_s.std(axis=0)
    s_min, s_max = all_s.min(axis=0), all_s.max(axis=0)
    a_mean, a_std = all_a.mean(axis=0), all_a.std(axis=0)
    s_mean[YAW_IDX] = 0.0
    s_std[YAW_IDX] = 1.0
    s_min[YAW_IDX] = -math.pi
    s_max[YAW_IDX] = math.pi
    s_std[s_std < 1e-8] = 1.0
    a_std[a_std < 1e-8] = 1.0

    norm = {"s_mean": s_mean, "s_std": s_std, "s_min": s_min, "s_max": s_max, "a_mean": a_mean, "a_std": a_std}

    # Normalize all segments
    train_norm = [((s - s_mean) / s_std, (a - a_mean) / a_std)
                  for s, a in train_segments]
    val_norm = [((s - s_mean) / s_std, (a - a_mean) / a_std)
                for s, a in val_segments]

    total_train = sum(len(s) for s, _ in train_norm)
    total_val = sum(len(s) for s, _ in val_norm)
    print(f"  Total: train={total_train}, val={total_val} "
          f"({len(train_norm)} laps)")

    return train_norm, val_norm, val_segments, val_segment_names, norm


class MultiLapSequenceDataset(Dataset):
    """Sliding window dataset across multiple laps (no cross-lap sequences)."""

    def __init__(self, segments: list, seq_len: int):
        self.seq_len = seq_len
        self.samples = []  # list of (segment_idx, start_idx)
        self.states = []
        self.actions = []
        for seg_idx, (s, a) in enumerate(segments):
            self.states.append(torch.tensor(s, dtype=torch.float32))
            self.actions.append(torch.tensor(a, dtype=torch.float32))
            n_windows = len(s) - seq_len
            for i in range(n_windows):
                self.samples.append((seg_idx, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg_idx, start = self.samples[idx]
        s = self.states[seg_idx][start: start + self.seq_len]
        a = self.actions[seg_idx][start: start + self.seq_len]
        s_next = self.states[seg_idx][start + 1: start + self.seq_len + 1]
        return s, a, s_next


# ─── Training ────────────────────────────────────────────────────────────────

def compute_losses(pred, target, dyn_weight, traj_weight, norm):
    # Dynamics MSE
    dyn_loss = ((pred[..., DYN_LOSS_INDICES] - target[..., DYN_LOSS_INDICES]) ** 2 * dyn_weight).mean()
    
    # Trajectory Loss (handle Yaw angle wrapping)
    # Positions (0, 1, 2) use standard MSE
    pos_diff = pred[..., 0:3] - target[..., 0:3]
    pos_loss = (pos_diff ** 2 * traj_weight[0:3]).mean()
    
    # Yaw uses wrapped angular difference. MUST DO THIS IN RAW RADIANS, NOT NORMALIZED!
    yaw_std = norm["s_std"][YAW_IDX].item()
    pred_yaw_raw = pred[..., YAW_IDX] * yaw_std
    target_yaw_raw = target[..., YAW_IDX] * yaw_std
    
    yaw_diff_raw = pred_yaw_raw - target_yaw_raw
    yaw_diff_wrapped_raw = torch.remainder(yaw_diff_raw + math.pi, 2 * math.pi) - math.pi
    
    # Convert wrapped error back to normalized scale so weights apply correctly
    yaw_diff_wrapped_norm = yaw_diff_wrapped_raw / yaw_std
    yaw_loss = (yaw_diff_wrapped_norm ** 2 * traj_weight[3]).mean()
    
    traj_loss = pos_loss + yaw_loss
    return dyn_loss, traj_loss


def trajectory_loss_scale(epoch):
    if epoch <= 1:
        return TRAJ_LOSS_W_START
    if epoch >= TRAJ_LOSS_RAMP_END:
        return TRAJ_LOSS_W_END
    frac = (epoch - 1) / max(1, TRAJ_LOSS_RAMP_END - 1)
    return TRAJ_LOSS_W_START + frac * (TRAJ_LOSS_W_END - TRAJ_LOSS_W_START)


def compute_rollout_loss(model, states, actions, s_next_gt, rollout_steps, traj_weight, dyn_weight, norm, device):
    """
    Compute multi-step rollout loss with recurrent hidden state.
    """
    B, T, sd = states.shape
    K = min(rollout_steps, T)

    # Pick a random start within each sequence
    max_start = T - K
    if max_start <= 0:
        return torch.tensor(0.0, device=device)
    start_idx = torch.randint(0, max_start, (1,)).item()

    state = states[:, start_idx, :]
    hidden = None
    rollout_loss = torch.tensor(0.0, device=device)

    for k in range(K):
        t = start_idx + k
        pred, hidden = model.step(state, actions[:, t, :], hidden)

        target = s_next_gt[:, t, :]
        
        # Handle Rollout Traj Loss with angle wrapping
        pos_diff = pred[..., 0:3] - target[..., 0:3]
        pos_loss = (pos_diff ** 2 * traj_weight[0:3]).mean()
        
        yaw_std = norm["s_std"][YAW_IDX].item()
        pred_yaw_raw = pred[..., YAW_IDX] * yaw_std
        target_yaw_raw = target[..., YAW_IDX] * yaw_std
        
        yaw_diff_raw = pred_yaw_raw - target_yaw_raw
        yaw_diff_wrapped_raw = torch.remainder(yaw_diff_raw + math.pi, 2 * math.pi) - math.pi
        
        yaw_diff_wrapped_norm = yaw_diff_wrapped_raw / yaw_std
        yaw_loss = (yaw_diff_wrapped_norm ** 2 * traj_weight[3]).mean()
        
        rollout_traj_loss = pos_loss + yaw_loss

        rollout_loss = rollout_loss + rollout_traj_loss
        if ROLLOUT_DYN_WEIGHT > 0:
            rollout_dyn_weight = torch.stack([
                dyn_weight[DYN_LOSS_INDICES.index(idx)] * ROLLOUT_DYN_LOSS_WEIGHTS.get(idx, 1.0)
                for idx in ROLLOUT_DYN_LOSS_INDICES
            ])
            
            # Supervise delta (derivative) instead of absolute value for dynamics
            pred_delta = pred[..., ROLLOUT_DYN_LOSS_INDICES] - state[..., ROLLOUT_DYN_LOSS_INDICES]
            target_delta = target[..., ROLLOUT_DYN_LOSS_INDICES] - state[..., ROLLOUT_DYN_LOSS_INDICES]
            
            dyn_diff = (pred_delta - target_delta) ** 2 * rollout_dyn_weight
            rollout_dyn_loss = dyn_diff.mean()
            rollout_loss = rollout_loss + ROLLOUT_DYN_WEIGHT * rollout_dyn_loss

        # CRITICAL FIX: Do NOT detach `state` and `hidden` here! 
        # Allow gradients to backpropagate through time (BPTT) across the rollout sequence
        # so the recurrent network actually learns to minimize accumulated errors over long horizons.
        state = pred

    return rollout_loss / K


def dream_val_error(model, val_seg_n, dream_steps, norm, device):
    """Quick dream rollout on one val segment, return mean 3D position error (m)."""
    model.eval()
    s_n, a_n = val_seg_n
    T = min(dream_steps, len(s_n) - 1)
    if T <= 0:
        return float("inf")

    init = torch.tensor(s_n[0], dtype=torch.float32).unsqueeze(0).to(device)
    acts = torch.tensor(a_n[:T], dtype=torch.float32).unsqueeze(0).to(device)

    pred_n = model.dream(init, acts).squeeze(0).cpu().numpy()  # [T, sd]

    s_mean, s_std = norm["s_mean"], norm["s_std"]
    pred_raw = pred_n * s_std + s_mean
    gt_raw = (s_n[1:T + 1] * s_std + s_mean)  # pred[t] corresponds to s[t+1]

    pos_err = pred_raw[:, POS_INDICES] - gt_raw[:, POS_INDICES]
    dist_3d = np.sqrt(np.sum(pos_err ** 2, axis=1))
    return float(np.mean(dist_3d))


def curriculum_rollout_steps(epoch):
    """Linearly grow rollout steps from ROLLOUT_MIN to ROLLOUT_MAX."""
    if epoch <= ROLLOUT_GROW_START:
        return ROLLOUT_MIN
    if epoch >= ROLLOUT_GROW_END:
        return ROLLOUT_MAX
    frac = (epoch - ROLLOUT_GROW_START) / (ROLLOUT_GROW_END - ROLLOUT_GROW_START)
    return int(ROLLOUT_MIN + frac * (ROLLOUT_MAX - ROLLOUT_MIN))


def train_model(model, train_loader, val_loader, val_seg_n, norm, config,
                model_dir=None):
    """Train with scheduled sampling + curriculum rollout loss.
    No early stopping — runs all epochs, saves best & periodic checkpoints."""
    device = config["device"]
    model = model.to(device)

    # Compile the model to fuse operations and reduce python/cuda launch overheads
    # This is highly effective for autoregressive step-by-step loops
    if hasattr(torch, "compile"):
        print("  Compiling model with torch.compile() for faster autoregressive training...", flush=True)
        # Use mode="default" because dynamic control flow in rollout/sampling causes graph breaks
        model = torch.compile(model, mode="default")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    # Warmup + Cosine Annealing
    total_epochs = config["epochs"]
    def lr_lambda(epoch_idx):
        if epoch_idx < WARMUP_EPOCHS:
            return (epoch_idx + 1) / WARMUP_EPOCHS
        epoch_after_warmup = epoch_idx - WARMUP_EPOCHS
        cosine_span = max(1, total_epochs - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * epoch_after_warmup / cosine_span))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dyn_weight = torch.ones(len(DYN_LOSS_INDICES), device=device)
    for i, idx in enumerate(DYN_LOSS_INDICES):
        if idx in VEL_INDICES:
            dyn_weight[i] = VEL_WEIGHT_VAL

    traj_weight = torch.ones(len(TRAJ_LOSS_INDICES), device=device)
    traj_weight[0:3] = POS_WEIGHT_VAL

    norm_tensors = {
        "s_mean": torch.tensor(norm["s_mean"], dtype=torch.float32, device=device),
        "s_std": torch.tensor(norm["s_std"], dtype=torch.float32, device=device),
    }

    best_dream_err = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "dream_err": [],
               "lr": [], "rollout_k": []}

    for epoch in range(1, config["epochs"] + 1):
        rollout_k = curriculum_rollout_steps(epoch)
        traj_scale = trajectory_loss_scale(epoch)

        # ---- Train ----
        model.train()
        train_losses = []
        for batch_idx, (s, a, s_next) in enumerate(train_loader):
            s, a, s_next = s.to(device), a.to(device), s_next.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred, _ = model(s, a)
                dyn_loss, traj_loss = compute_losses(pred, s_next, dyn_weight, traj_weight, norm_tensors)
                loss_tf = dyn_loss + traj_scale * traj_loss

                do_rollout = epoch >= ROLLOUT_GROW_START and (batch_idx % ROLLOUT_EVERY_N == 0)
                if do_rollout:
                    loss_rollout = compute_rollout_loss(
                        model, s, a, s_next, rollout_k, traj_weight, dyn_weight, norm_tensors, device)
                    loss = loss_tf + ROLLOUT_WEIGHT * loss_rollout
                else:
                    loss = loss_tf

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        scheduler.step()

        # ---- Validate (teacher forcing for logging) ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for s, a, s_next in val_loader:
                s, a, s_next = s.to(device), a.to(device), s_next.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    pred, _ = model(s, a)
                    dyn_loss, traj_loss = compute_losses(pred, s_next, dyn_weight, traj_weight, norm_tensors)
                    loss = dyn_loss + traj_scale * traj_loss
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        # ---- Dream validation (for monitoring) ----
        if epoch % DREAM_VAL_EVERY == 0 or epoch == 1:
            d_err = dream_val_error(
                model, val_seg_n, DREAM_VAL_STEPS, norm, device)
        else:
            d_err = history["dream_err"][-1] if history["dream_err"] else float("inf")

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["dream_err"].append(d_err)
        history["lr"].append(current_lr)
        history["rollout_k"].append(rollout_k)

        # Save best model by dream error
        marker = ""
        if d_err < best_dream_err:
            best_dream_err = d_err
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
            if model_dir is not None:
                torch.save(best_state, model_dir / "wam_v2_best_dream.pt")

        # Periodic checkpoint
        if model_dir is not None and epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, model_dir / f"wam_v2_ckpt_ep{epoch}.pt")

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:3d}/{config['epochs']} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f} | "
                  f"dream={d_err:.2f}m | rollout_k={rollout_k} | "
                  f"lr={current_lr:.1e} | traj_w={traj_scale:.2f}{marker}",
                  flush=True)

        # ---- Early stopping on absolute dream error threshold ----
        if d_err <= EARLY_STOP_DREAM_ERR_THRESHOLD:
            print(f"  [Early Stopping] Dream error ({d_err:.2f}m) reached threshold ({EARLY_STOP_DREAM_ERR_THRESHOLD}m) at epoch {epoch}!", flush=True)
            break

    # Load best model at the end
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_dream(model, s_val_n, a_val_n, s_val_raw, norm, device, dream_steps):
    """
    Evaluate with autoregressive dream rollout.
    Start from the first validation state, dream for dream_steps.
    The first prediction is aligned against the next ground-truth state.
    """
    model.eval()
    T = min(dream_steps, len(s_val_n) - 1)

    init_state = torch.tensor(s_val_n[0], dtype=torch.float32).unsqueeze(0).to(device)
    action_seq = torch.tensor(a_val_n[:T], dtype=torch.float32).unsqueeze(0).to(device)

    pred_n = model.dream(init_state, action_seq)  # [1, T, state_dim]
    pred_n = pred_n.squeeze(0).cpu().numpy()       # [T, state_dim]

    # Denormalize
    s_mean, s_std = norm["s_mean"], norm["s_std"]
    pred_raw = pred_n * s_std + s_mean
    gt_raw = s_val_raw[1:T + 1]

    # Position errors
    pos_pred = pred_raw[:, POS_INDICES]
    pos_gt = gt_raw[:, POS_INDICES]
    errors = pos_pred - pos_gt
    dist_3d = np.sqrt(np.sum(errors ** 2, axis=1))

    # Velocity errors
    vel_pred = pred_raw[:, VEL_INDICES]
    vel_gt = gt_raw[:, VEL_INDICES]
    vel_errors = vel_pred - vel_gt
    speed_err = np.sqrt(np.sum(vel_errors ** 2, axis=1))

    print("\n" + "=" * 55)
    print(f"Dream Rollout Evaluation ({T} steps = {T * DT:.1f}s)")
    print("=" * 55)
    print(f"  3D MAE:        {np.mean(dist_3d):.4f} m")
    print(f"  3D Max Error:  {np.max(dist_3d):.4f} m")
    for i, name in enumerate(["posE", "posN", "posU"]):
        mae = np.mean(np.abs(errors[:, i]))
        print(f"  {name}: MAE={mae:.4f} m")
    print(f"  --- Velocity ---")
    print(f"  Speed MAE:     {np.mean(speed_err):.4f} m/s")
    for i, name in enumerate(["Vx", "Vy", "Vz"]):
        mae = np.mean(np.abs(vel_errors[:, i]))
        print(f"  {name}: MAE={mae:.4f} m/s")

    # Also evaluate at specific horizons
    for horizon_s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        h_steps = int(horizon_s / DT)
        if h_steps <= T:
            d = dist_3d[h_steps - 1]
            v = speed_err[h_steps - 1]
            print(f"  @{horizon_s:.1f}s (step {h_steps}): 3D={d:.4f}m, Speed={v:.4f}m/s")

    return pred_raw, gt_raw, errors, dist_3d


def sanitize_stem_suffix(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")


def evaluate_teacher_forcing(model, s_val_n, a_val_n, s_val_raw, norm, device):
    """Evaluate one-step prediction accuracy with teacher forcing."""
    model.eval()
    T = len(s_val_n) - 1

    s_in = torch.tensor(s_val_n[:T], dtype=torch.float32).unsqueeze(0).to(device)
    a_in = torch.tensor(a_val_n[:T], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_n, _ = model(s_in, a_in)
    pred_n = pred_n.squeeze(0).cpu().numpy()

    s_mean, s_std = norm["s_mean"], norm["s_std"]
    pred_raw = pred_n * s_std + s_mean
    gt_raw = s_val_raw[1:T+1]

    pos_pred = pred_raw[:, POS_INDICES]
    pos_gt = gt_raw[:, POS_INDICES]
    errors = pos_pred - pos_gt
    dist_3d = np.sqrt(np.sum(errors ** 2, axis=1))

    print("\n" + "=" * 55)
    print(f"Teacher-Forcing One-Step Evaluation ({T} steps)")
    print("=" * 55)
    print(f"  3D MAE:        {np.mean(dist_3d):.6f} m")
    print(f"  3D Max Error:  {np.max(dist_3d):.6f} m")
    for i, name in enumerate(["posE", "posN", "posU"]):
        mae = np.mean(np.abs(errors[:, i]))
        print(f"  {name}: MAE={mae:.6f} m")

    return pred_raw, gt_raw


# ─── Visualization ───────────────────────────────────────────────────────────

def save_figure(fig, out_dir: Path, stem: str, keep_html: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if keep_html:
        out = out_dir / f"{stem}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        print(f"  saved {out}")
        return
    try:
        out = out_dir / f"{stem}.svg"
        fig.write_image(str(out))
    except Exception:
        out = out_dir / f"{stem}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_training_history(history, out_dir: Path):
    """Loss and LR curves."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Loss (log scale)", "Learning Rate"],
                        vertical_spacing=0.08)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             name="Train", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             name="Val", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["lr"],
                             name="LR", line=dict(color="green")), row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(title="WAM v2 Training History", height=600, width=900)
    save_figure(fig, out_dir, "wam_v2_training_loss")


def plot_dream_3d(pred_raw, gt_raw, out_dir: Path, stem_suffix: str = "", title_suffix: str = ""):
    """3D dream rollout vs ground truth."""
    fig = go.Figure()
    ds = 5
    fig.add_trace(go.Scatter3d(
        x=gt_raw[::ds, 0], y=gt_raw[::ds, 1], z=gt_raw[::ds, 2],
        mode="lines", name="Ground Truth",
        line=dict(color="blue", width=4),
    ))
    fig.add_trace(go.Scatter3d(
        x=pred_raw[::ds, 0], y=pred_raw[::ds, 1], z=pred_raw[::ds, 2],
        mode="lines", name="Dream Rollout",
        line=dict(color="red", width=4),
    ))
    # Mark start
    fig.add_trace(go.Scatter3d(
        x=[gt_raw[0, 0]], y=[gt_raw[0, 1]], z=[gt_raw[0, 2]],
        mode="markers", marker=dict(size=8, color="green", symbol="diamond"),
        name="Start",
    ))
    fig.update_layout(
        title=f"WAM v2 Dream Rollout{title_suffix} ({len(gt_raw)} steps = {len(gt_raw)*DT:.1f}s)",
        scene=dict(
            xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Up (m)",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.15),
        ),
        width=1100, height=800,
    )
    save_figure(fig, out_dir, f"wam_v2_dream_3d{stem_suffix}", keep_html=True)


def plot_dream_topdown(pred_raw, gt_raw, out_dir: Path, stem_suffix: str = "", title_suffix: str = ""):
    """Top-down dream comparison."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gt_raw[:, 0], y=gt_raw[:, 1],
        mode="lines", name="Ground Truth", line=dict(color="blue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pred_raw[:, 0], y=pred_raw[:, 1],
        mode="lines", name="Dream Rollout", line=dict(color="red", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[gt_raw[0, 0]], y=[gt_raw[0, 1]],
        mode="markers+text", marker=dict(size=12, color="green", symbol="diamond"),
        text=["START"], textposition="top center", name="Start",
    ))
    fig.update_layout(
        title=f"WAM v2 Dream Rollout (Top-Down){title_suffix}",
        xaxis_title="East (m)", yaxis_title="North (m)",
        width=1000, height=800,
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
    )
    save_figure(fig, out_dir, f"wam_v2_dream_topdown{stem_suffix}")


def plot_dream_error_over_time(dist_3d, out_dir: Path, stem_suffix: str = "", title_suffix: str = ""):
    """3D error accumulation over dream horizon."""
    t = np.arange(len(dist_3d)) * DT  # seconds
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=dist_3d, mode="lines",
                             name="3D Error", line=dict(color="purple", width=2)))
    fig.update_layout(
        title=f"Dream Rollout: 3D Position Error vs Time{title_suffix}",
        xaxis_title="Time Horizon (s)", yaxis_title="3D Error (m)",
        width=900, height=400,
    )
    save_figure(fig, out_dir, f"wam_v2_dream_error_vs_time{stem_suffix}")


def plot_state_comparison(pred_raw, gt_raw, out_dir: Path, stem_suffix: str = "", title_suffix: str = ""):
    """Compare all predicted state dimensions vs ground truth."""
    t = np.arange(len(gt_raw)) * DT
    state_names = STATE_COLS
    n = len(state_names)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{s}" for s in state_names],
                        vertical_spacing=0.02)
    for i, name in enumerate(state_names):
        row = i + 1
        fig.add_trace(go.Scatter(x=t, y=gt_raw[:, i], name=f"GT {name}",
                                 line=dict(color="blue", width=1),
                                 showlegend=(i == 0)), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=pred_raw[:, i], name=f"Pred {name}",
                                 line=dict(color="red", width=1, dash="dash"),
                                 showlegend=(i == 0)), row=row, col=1)
    fig.update_layout(
        title=f"WAM v2 Dream: All State Dimensions{title_suffix}",
        height=max(2200, n * 180), width=1400,
    )
    save_figure(fig, out_dir, f"wam_v2_state_comparison{stem_suffix}")


def plot_dream_velocity(pred_raw, gt_raw, out_dir: Path, stem_suffix: str = "", title_suffix: str = ""):
    """Compare predicted vs actual velocity (Vx, Vy, Vz) during dream rollout."""
    t = np.arange(len(gt_raw)) * DT
    vel_names = ["Vx_mps", "Vy_mps", "Vz_mps"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Vx (m/s)", "Vy (m/s)", "Vz (m/s)", "Speed Error (m/s)"],
                        vertical_spacing=0.08)

    for i, vi in enumerate(VEL_INDICES):
        row = i + 1
        fig.add_trace(go.Scatter(x=t, y=gt_raw[:, vi], name=f"GT {vel_names[i]}",
                                 line=dict(color="blue", width=1.5),
                                 showlegend=(i == 0)), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=pred_raw[:, vi], name=f"Pred {vel_names[i]}",
                                 line=dict(color="red", width=1.5, dash="dash"),
                                 showlegend=(i == 0)), row=row, col=1)

    # Speed error over time
    vel_err = pred_raw[:, VEL_INDICES] - gt_raw[:, VEL_INDICES]
    speed_err = np.sqrt(np.sum(vel_err ** 2, axis=1))
    fig.add_trace(go.Scatter(x=t, y=speed_err, name="Speed Error",
                             line=dict(color="purple", width=1.5),
                             showlegend=False), row=4, col=1)

    fig.update_layout(
        title=f"WAM v2 Dream: Velocity Prediction vs Ground Truth{title_suffix}",
        height=900, width=1100,
    )
    save_figure(fig, out_dir, f"wam_v2_dream_velocity{stem_suffix}")


def plot_action_intervention(model, s_val_n, a_val_n, norm, device, out_dir: Path):
    """
    动作干预实验: 从相同初始状态出发, 分别使用
    原始动作 / 左转加大 / 右转加大, 观察轨迹差异.
    """
    model.eval()
    T = 400  # 2 seconds
    T = min(T, len(a_val_n) - 1)

    init_state = torch.tensor(s_val_n[0], dtype=torch.float32).unsqueeze(0).to(device)
    actions_base = torch.tensor(a_val_n[:T], dtype=torch.float32).unsqueeze(0).to(device)

    s_mean, s_std = norm["s_mean"], norm["s_std"]

    scenarios = {
        "Original Actions": actions_base.clone(),
        "Steering +50%": actions_base.clone(),
        "Steering -50%": actions_base.clone(),
        "Full Brake": actions_base.clone(),
    }
    # Steering is the first action dimension (index 0)
    scenarios["Steering +50%"][:, :, 0] *= 1.5
    scenarios["Steering -50%"][:, :, 0] *= 0.5
    # Brake commands are indices 2-5, throttle is index 1
    scenarios["Full Brake"][:, :, 1] = (0 - norm["a_mean"][1]) / norm["a_std"][1]  # zero throttle
    for bi in range(2, 6):
        scenarios["Full Brake"][:, :, bi] = (40 - norm["a_mean"][bi]) / norm["a_std"][bi]  # max brake

    colors = {"Original Actions": "blue", "Steering +50%": "red",
              "Steering -50%": "green", "Full Brake": "orange"}

    fig = go.Figure()
    for name, act in scenarios.items():
        pred = model.dream(init_state, act)  # [1, T, state_dim]
        pred = pred.squeeze(0).cpu().numpy()
        pred_raw = pred * s_std + s_mean
        fig.add_trace(go.Scatter(
            x=pred_raw[:, 0], y=pred_raw[:, 1],
            mode="lines", name=name,
            line=dict(color=colors[name], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[s_val_n[0, 0] * s_std[0] + s_mean[0]],
        y=[s_val_n[0, 1] * s_std[1] + s_mean[1]],
        mode="markers+text",
        marker=dict(size=12, color="black", symbol="diamond"),
        text=["START"], textposition="top center", name="Start",
    ))

    fig.update_layout(
        title="Action Intervention: Same Start, Different Actions (2s Dream)",
        xaxis_title="East (m)", yaxis_title="North (m)",
        width=1000, height=800,
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
    )
    save_figure(fig, out_dir, "wam_v2_action_intervention")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Force unbuffered stdout for nohup
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("WAM v2 — World Action Model with Temporal Dynamics")
    print("=" * 60)

    # 1) Load multi-lap data
    print(f"\n[1/6] Loading multi-lap data (downsample={DOWNSAMPLE}, "
          f"dt={DT:.3f}s = {1/DT:.0f}Hz)...")
    train_segs, val_segs, val_raw_segs, val_seg_names, norm = load_multi_lap_data(
        RAW_DIR, TRAIN_RATIO, downsample=DOWNSAMPLE)

    train_ds = MultiLapSequenceDataset(train_segs, SEQ_LEN)
    val_ds = MultiLapSequenceDataset(val_segs, SEQ_LEN)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    print(f"  State dim: {STATE_DIM}, Obs dim: {OBS_DIM}, Action dim: {ACTION_DIM}")
    print(f"  Seq len: {SEQ_LEN} ({SEQ_LEN * DT:.1f}s)")
    print(f"  Train sequences: {len(train_ds)}")
    print(f"  Val sequences: {len(val_ds)}")
    print(f"  Curriculum rollout: {ROLLOUT_MIN}->{ROLLOUT_MAX} steps "
          f"({ROLLOUT_MIN*DT:.1f}s->{ROLLOUT_MAX*DT:.1f}s), weight={ROLLOUT_WEIGHT}")
    print(f"  Dream val: {DREAM_VAL_STEPS} steps ({DREAM_VAL_STEPS*DT:.1f}s) "
          f"every {DREAM_VAL_EVERY} epochs")
    print(f"  Dream plots: {min(DREAM_PLOT_SEGMENTS, len(val_segs))} validation segments")
    print(f"  No early stopping. Checkpoints every {SAVE_EVERY} epochs.")
    print(f"  Device: {DEVICE}")

    # 2) Build model
    print("\n[2/6] Building WAM v2...")
    model = WAM(STATE_DIM, ACTION_DIM, obs_dim=OBS_DIM,
                d_model=D_MODEL, num_layers=SSM_LAYERS,
                ssm_state_dim=SSM_STATE_DIM,
                hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    model.set_normalization(norm["s_mean"], norm["s_std"], norm["s_min"], norm["s_max"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Obs encoder input: {OBS_DIM}d (dynamic-only + sin/cos yaw)")
    print(f"  Temporal backend: {model.temporal_backend}")
    print(f"  SSM: d_model={D_MODEL}, layers={SSM_LAYERS}, state_dim={SSM_STATE_DIM}")
    print(f"  Decoder hidden: {HIDDEN_DIM}")
    print(f"  Parameters: {n_params:,}")

    # Prepare val segment for dream monitoring
    val_seg_n = val_segs[0]  # first lap's validation segment (normalized)

    # 3) Train
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[3/6] Training with curriculum rollout + scheduled sampling...")
    config = {
        "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "device": DEVICE,
    }
    model, history = train_model(
        model, train_loader, val_loader, val_seg_n, norm, config,
        model_dir=MODEL_DIR)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "wam_v2_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm": {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in norm.items()},
        "config": {
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "obs_dim": OBS_DIM,
            "d_model": D_MODEL, "num_layers": SSM_LAYERS,
            "ssm_state_dim": SSM_STATE_DIM,
            "temporal_backend": model.temporal_backend,
            "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
            "state_cols": STATE_COLS, "action_cols": ACTION_COLS,
            "obs_indices": OBS_INDICES,
            "obs_state_indices": OBS_STATE_INDICES,
            "yaw_idx": YAW_IDX,
            "yawrate_idx": YAWRATE_IDX,
            "residual_target_indices": RESIDUAL_TARGET_INDICES,
        },
    }, model_path)
    print(f"  Model saved to {model_path}")

    # Use first validation segment for single teacher-forcing evaluation
    s_val_n = val_segs[0][0]
    a_val_n = val_segs[0][1]
    s_val_raw = val_raw_segs[0][0]

    # 4) Evaluate (teacher forcing)
    print("\n[4/6] Evaluating (teacher forcing, one-step)...")
    evaluate_teacher_forcing(model, s_val_n, a_val_n, s_val_raw, norm, DEVICE)

    # 5) Evaluate (dream rollout)
    print("\n[5/6] Evaluating (dream rollout across validation segments)...")
    dream_results = []
    n_dream_segments = min(DREAM_PLOT_SEGMENTS, len(val_segs))
    for i in range(n_dream_segments):
        seg_name = val_seg_names[i]
        print(f"\n  Dream segment {i+1}/{n_dream_segments}: {seg_name}")
        pred_raw, gt_raw, errors, dist_3d = evaluate_dream(
            model, val_segs[i][0], val_segs[i][1], val_raw_segs[i][0], norm, DEVICE, DREAM_STEPS)
        dream_results.append({
            "name": seg_name,
            "pred_raw": pred_raw,
            "gt_raw": gt_raw,
            "errors": errors,
            "dist_3d": dist_3d,
        })

    # 6) Visualize
    print("\n[6/6] Generating visualizations...")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, VIS_DIR)
    for result in dream_results:
        # Save per-lap plots into their own subdirectories
        lap_name = sanitize_stem_suffix(result['name'])
        lap_dir = VIS_DIR / lap_name
        
        # 3D plot is interactive HTML, the rest are SVG
        plot_dream_3d(result["pred_raw"], result["gt_raw"], lap_dir, title_suffix=f" — {result['name']}")
        plot_dream_topdown(result["pred_raw"], result["gt_raw"], lap_dir, title_suffix=f" — {result['name']}")
        plot_dream_error_over_time(result["dist_3d"], lap_dir, title_suffix=f" — {result['name']}")
        plot_state_comparison(result["pred_raw"], result["gt_raw"], lap_dir, title_suffix=f" — {result['name']}")
        plot_dream_velocity(result["pred_raw"], result["gt_raw"], lap_dir, title_suffix=f" — {result['name']}")
    plot_action_intervention(model, s_val_n, a_val_n, norm, DEVICE, VIS_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Model:          {model_path}")
    print(f"  Visualizations: {VIS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
