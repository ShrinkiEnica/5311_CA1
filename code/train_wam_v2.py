"""
WAM v2 — World Action Model with Temporal Dynamics

升级要点:
  1. State/Action 明确分离
  2. 隐空间编码 (Latent Space Encoder/Decoder)
  3. GRU 时序动力学模型
  4. 残差预测 (预测 Δstate 而非绝对值)
  5. Scheduled Sampling 训练 (Teacher-Forcing → Autoregressive 渐进过渡)
  6. 多步 Rollout Loss
  7. 多圈数据训练 (10 圈, ~150K 样本)
  8. 动作干预可视化

State (10 dim):
  posE_m, posN_m, posU_m          — 位置 (3)
  Vx_mps, Vy_mps                  — 车体速度 (2)
  yawAngle_rad                    — 偏航角 (1)
  yawRate_radps                   — 偏航角速度 (1)
  axCG_mps2, ayCG_mps2            — 纵/横向加速度 (2)
  slipAngle_rad                   — 侧滑角 (1)

Action (6 dim):
  roadWheelAngle_rad              — 转向角 (1)
  throttleCmd_percent             — 油门指令 (1)
  brakeCmd_fl/fr/rl/rr_bar        — 四轮制动指令 (4)

Transition:  s_{t+1} = s_t + f(s_t, a_t, h_t)
Dream:       给定初始状态 + 动作序列, 自回归推演未来轨迹

用法:
  python code/train_wam_v2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_data"
MODEL_DIR = ROOT / "models"
VIS_DIR = ROOT / "visualization" / "wam_v2"

# Laps to exclude (partial data)
EXCLUDE_LAPS = {"_5", "_10"}

# ─── Feature Definitions ─────────────────────────────────────────────────────

STATE_COLS = [
    "posE_m", "posN_m", "posU_m",       # position (3)
    "Vx_mps", "Vy_mps",                 # body velocity (2)
    "yawAngle_rad",                      # heading (1)
    "yawRate_radps",                     # yaw rate (1)
    "axCG_mps2", "ayCG_mps2",           # acceleration (2)
    "slipAngle_rad",                     # side slip (1)
]
STATE_DIM = len(STATE_COLS)  # 10

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

# ─── Config ──────────────────────────────────────────────────────────────────

DOWNSAMPLE = 10         # 200Hz -> 20Hz (dt = 0.05s)
DT = 0.005 * DOWNSAMPLE # 0.05s per step

TRAIN_RATIO = 0.7
SEQ_LEN = 100           # training sequence length (5.0s @ 20Hz)
BATCH_SIZE = 256
EPOCHS = 300
LR = 3e-4
WEIGHT_DECAY = 1e-5
LATENT_DIM = 128
HIDDEN_DIM = 256
GRU_LAYERS = 3
DROPOUT = 0.1
LR_PATIENCE = 20
DREAM_STEPS = 400       # dream rollout length for evaluation (20s @ 20Hz)

# Scheduled Sampling
SS_START_EPOCH = 5
SS_END_EPOCH = 150
SS_MAX_RATIO = 0.5

# Curriculum Learning: rollout steps grow during training
ROLLOUT_MIN = 5         # start with 5-step rollout (0.25s)
ROLLOUT_MAX = 60        # grow to 60-step rollout (3.0s)
ROLLOUT_GROW_START = 1  # start growing at epoch 1
ROLLOUT_GROW_END = 200  # reach max rollout by epoch 200
ROLLOUT_WEIGHT = 0.3
ROLLOUT_EVERY_N = 4

# Dream validation (for monitoring, NOT early stopping)
DREAM_VAL_STEPS = 200   # 10s dream for validation metric
DREAM_VAL_EVERY = 10    # evaluate dream every N epochs
SAVE_EVERY = 50         # save checkpoint every N epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding window dataset for sequential WAM training."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, seq_len: int):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
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

class WAM(nn.Module):
    """
    Physics-Informed World Action Model v2

    Architecture:
      State Encoder:  state (10d) → latent
      Action Encoder: action (6d) → latent/2
      Dynamics:       GRU over concatenated latent
      State Decoder:  latent → Δstate (10d) correction
      Transition:     s_{t+1} = s_t + physics_delta + nn_correction

    Physics prior (computed in raw space, converted back to normalized):
      ΔposE = (Vx·cos(yaw) - Vy·sin(yaw)) · dt
      ΔposN = (Vx·sin(yaw) + Vy·cos(yaw)) · dt
      ΔposU = 0
      Δyaw  = yawRate · dt
    """

    # State column indices
    IDX_POSE, IDX_POSN, IDX_POSU = 0, 1, 2
    IDX_VX, IDX_VY = 3, 4
    IDX_YAW = 5
    IDX_YAWRATE = 6

    def __init__(self, state_dim: int, action_dim: int,
                 latent_dim: int = 64, hidden_dim: int = 128,
                 gru_layers: int = 2, dropout: float = 0.1,
                 dt: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.dt = dt

        # Normalization params (set via set_norm)
        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))

        # State encoder: compress state to latent space
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Action encoder
        action_latent = latent_dim // 2
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_latent),
        )

        # GRU dynamics in latent space
        self.gru = nn.GRU(
            input_size=latent_dim + action_latent,
            hidden_size=latent_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # State decoder: latent → Δstate correction (small residual)
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def set_norm(self, s_mean, s_std):
        """Set normalization parameters for physics computation."""
        self.s_mean.copy_(torch.tensor(s_mean, dtype=torch.float32))
        self.s_std.copy_(torch.tensor(s_std, dtype=torch.float32))

    def _physics_delta_normalized(self, states_n):
        """
        Compute physics-based state delta in normalized space.
        states_n: [..., state_dim] normalized states
        Returns:  [..., state_dim] physics delta in normalized space
        """
        # Denormalize the relevant states
        vx_raw = states_n[..., self.IDX_VX] * self.s_std[self.IDX_VX] + self.s_mean[self.IDX_VX]
        vy_raw = states_n[..., self.IDX_VY] * self.s_std[self.IDX_VY] + self.s_mean[self.IDX_VY]
        yaw_raw = states_n[..., self.IDX_YAW] * self.s_std[self.IDX_YAW] + self.s_mean[self.IDX_YAW]
        yaw_rate_raw = states_n[..., self.IDX_YAWRATE] * self.s_std[self.IDX_YAWRATE] + self.s_mean[self.IDX_YAWRATE]

        cos_yaw = torch.cos(yaw_raw)
        sin_yaw = torch.sin(yaw_raw)

        # Position deltas in raw space (kinematic bicycle model)
        dE_raw = (vx_raw * cos_yaw - vy_raw * sin_yaw) * self.dt
        dN_raw = (vx_raw * sin_yaw + vy_raw * cos_yaw) * self.dt
        dYaw_raw = yaw_rate_raw * self.dt

        # Convert back to normalized space
        physics_delta = torch.zeros_like(states_n)
        physics_delta[..., self.IDX_POSE] = dE_raw / self.s_std[self.IDX_POSE]
        physics_delta[..., self.IDX_POSN] = dN_raw / self.s_std[self.IDX_POSN]
        physics_delta[..., self.IDX_YAW] = dYaw_raw / self.s_std[self.IDX_YAW]

        return physics_delta

    def forward(self, states, actions, hidden=None):
        """
        Teacher-forcing forward pass.
        states:  [B, T, state_dim]  — ground truth normalized states
        actions: [B, T, action_dim]
        Returns: predicted s_{t+1} [B, T, state_dim], hidden
        """
        z_s = self.state_encoder(states)         # [B, T, latent]
        z_a = self.action_encoder(actions)       # [B, T, latent//2]
        z_in = torch.cat([z_s, z_a], dim=-1)

        z_out, hidden = self.gru(z_in, hidden)   # [B, T, latent]

        nn_correction = self.state_decoder(z_out) # [B, T, state_dim]
        physics_delta = self._physics_delta_normalized(states)

        next_states = states + physics_delta + nn_correction

        return next_states, hidden

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
            s = state.unsqueeze(1)                    # [B, 1, state_dim]
            a = action_seq[:, t:t+1, :]               # [B, 1, action_dim]

            z_s = self.state_encoder(s)
            z_a = self.action_encoder(a)
            z_in = torch.cat([z_s, z_a], dim=-1)

            z_out, hidden = self.gru(z_in, hidden)
            nn_correction = self.state_decoder(z_out)
            physics_delta = self._physics_delta_normalized(s)

            next_state = s + physics_delta + nn_correction
            state = next_state.squeeze(1)
            predictions.append(state)

        return torch.stack(predictions, dim=1)         # [B, T, state_dim]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_multi_lap_data(raw_dir: Path, train_ratio: float, downsample: int = 1):
    """
    Load all complete laps, downsample, split each lap 7:3 by time.
    Returns training/validation segments as lists of (state, action) arrays,
    plus concatenated arrays for normalization.
    """
    files = sorted(raw_dir.glob("VehicleDynamicsDataset_Nov2023_2023-11*.csv"))

    train_segments = []  # list of (states, actions) arrays
    val_segments = []
    all_train_s, all_train_a = [], []

    for f in files:
        suffix = f.stem.replace("VehicleDynamicsDataset_Nov2023_2023-11", "")
        if suffix in EXCLUDE_LAPS:
            print(f"    skip {f.name} (partial lap)")
            continue
        df = pd.read_csv(f, comment="#")
        # Downsample
        if downsample > 1:
            df = df.iloc[::downsample].reset_index(drop=True)
        s = df[STATE_COLS].values.astype(np.float64)
        a = df[ACTION_COLS].values.astype(np.float64)
        n = len(df)
        split = int(n * train_ratio)
        train_segments.append((s[:split], a[:split]))
        val_segments.append((s[split:], a[split:]))
        all_train_s.append(s[:split])
        all_train_a.append(a[:split])
        name = f"Lap{suffix}" if suffix else "Lap_0"
        print(f"    {name}: {n} pts (ds={downsample}) -> train={split}, val={n-split}")

    # Compute normalization from all training data
    all_s = np.concatenate(all_train_s, axis=0)
    all_a = np.concatenate(all_train_a, axis=0)
    s_mean, s_std = all_s.mean(axis=0), all_s.std(axis=0)
    a_mean, a_std = all_a.mean(axis=0), all_a.std(axis=0)
    s_std[s_std < 1e-8] = 1.0
    a_std[a_std < 1e-8] = 1.0

    norm = {"s_mean": s_mean, "s_std": s_std, "a_mean": a_mean, "a_std": a_std}

    # Normalize all segments
    train_norm = [((s - s_mean) / s_std, (a - a_mean) / a_std)
                  for s, a in train_segments]
    val_norm = [((s - s_mean) / s_std, (a - a_mean) / a_std)
                for s, a in val_segments]

    total_train = sum(len(s) for s, _ in train_norm)
    total_val = sum(len(s) for s, _ in val_norm)
    print(f"  Total: train={total_train}, val={total_val} "
          f"({len(train_norm)} laps)")

    return train_norm, val_norm, val_segments, norm


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

def scheduled_sampling_ratio(epoch, start_epoch, end_epoch, max_ratio):
    """Linearly increase autoregressive ratio from 0 to max_ratio."""
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return max_ratio
    return max_ratio * (epoch - start_epoch) / (end_epoch - start_epoch)


def forward_with_scheduled_sampling(model, states, actions, s_next_gt, ss_ratio, device):
    """
    Forward pass with scheduled sampling.
    With probability ss_ratio, use model's own prediction as next input
    instead of ground truth.
    """
    B, T, sd = states.shape
    if ss_ratio <= 0.0:
        # Pure teacher forcing
        return model(states, actions)

    # Step-by-step with scheduled sampling
    predictions = []
    hidden = None
    current_state = states[:, 0:1, :]  # [B, 1, sd]

    for t in range(T):
        a_t = actions[:, t:t+1, :]  # [B, 1, ad]
        pred_t, hidden = model(current_state, a_t, hidden)
        predictions.append(pred_t.squeeze(1))  # [B, sd]

        if t < T - 1:
            # Decide: use GT or own prediction for next step
            use_pred = (torch.rand(B, 1, 1, device=device) < ss_ratio).float()
            gt_next = states[:, t+1:t+2, :]  # [B, 1, sd]
            current_state = use_pred * pred_t.detach() + (1 - use_pred) * gt_next

    return torch.stack(predictions, dim=1), hidden  # [B, T, sd]


def compute_rollout_loss(model, states, actions, s_next_gt, rollout_steps, pos_weight, device):
    """
    Compute multi-step rollout loss: unroll for K steps autoregressively
    and accumulate position prediction error.
    """
    B, T, sd = states.shape
    K = min(rollout_steps, T)

    # Pick a random start within each sequence
    max_start = T - K
    if max_start <= 0:
        return torch.tensor(0.0, device=device)
    start_idx = torch.randint(0, max_start, (1,)).item()

    state = states[:, start_idx, :]  # [B, sd]
    hidden = None
    rollout_loss = torch.tensor(0.0, device=device)

    for k in range(K):
        t = start_idx + k
        s_in = state.unsqueeze(1)
        a_in = actions[:, t:t+1, :]
        pred, hidden = model(s_in, a_in, hidden)
        pred = pred.squeeze(1)  # [B, sd]

        target = s_next_gt[:, t, :]  # [B, sd]
        diff = (pred - target) ** 2 * pos_weight
        rollout_loss = rollout_loss + diff.mean()

        state = pred  # use own prediction (autoregressive)

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
    gt_raw = (s_n[:T] * s_std + s_mean)  # denorm GT

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config["lr_patience"], factor=0.5,
        min_lr=1e-6)

    # Position-weighted loss
    pos_weight = torch.ones(STATE_DIM, device=device)
    pos_weight[POS_INDICES] = 5.0

    best_dream_err = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "dream_err": [],
               "lr": [], "ss_ratio": [], "rollout_k": []}

    for epoch in range(1, config["epochs"] + 1):
        ss_ratio = scheduled_sampling_ratio(
            epoch, SS_START_EPOCH, SS_END_EPOCH, SS_MAX_RATIO)
        rollout_k = curriculum_rollout_steps(epoch)

        # ---- Train ----
        model.train()
        train_losses = []
        for batch_idx, (s, a, s_next) in enumerate(train_loader):
            s, a, s_next = s.to(device), a.to(device), s_next.to(device)
            optimizer.zero_grad()

            # Single-step loss with scheduled sampling
            pred, _ = forward_with_scheduled_sampling(
                model, s, a, s_next, ss_ratio, device)
            diff = (pred - s_next) ** 2 * pos_weight
            loss_tf = diff.mean()

            # Curriculum rollout loss (every N batches to save time)
            if batch_idx % ROLLOUT_EVERY_N == 0:
                loss_rollout = compute_rollout_loss(
                    model, s, a, s_next, rollout_k, pos_weight, device)
                loss = loss_tf + ROLLOUT_WEIGHT * loss_rollout
            else:
                loss = loss_tf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ---- Validate (teacher forcing for logging) ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for s, a, s_next in val_loader:
                s, a, s_next = s.to(device), a.to(device), s_next.to(device)
                pred, _ = model(s, a)
                diff = (pred - s_next) ** 2 * pos_weight
                loss = diff.mean()
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
        history["ss_ratio"].append(ss_ratio)
        history["rollout_k"].append(rollout_k)
        scheduler.step(val_loss)

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
                  f"lr={current_lr:.1e} | ss={ss_ratio:.2f}{marker}",
                  flush=True)

        # # ---- Early stopping (DISABLED) ----
        # if no_improve >= config["patience"]:
        #     print(f"  Early stopping at epoch {epoch}")
        #     break

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
    gt_raw = s_val_raw[:T]

    # Position errors
    pos_pred = pred_raw[:, POS_INDICES]
    pos_gt = gt_raw[:, POS_INDICES]
    errors = pos_pred - pos_gt
    dist_3d = np.sqrt(np.sum(errors ** 2, axis=1))

    print("\n" + "=" * 55)
    print(f"Dream Rollout Evaluation ({T} steps = {T * DT:.1f}s)")
    print("=" * 55)
    print(f"  3D MAE:        {np.mean(dist_3d):.4f} m")
    print(f"  3D Max Error:  {np.max(dist_3d):.4f} m")
    for i, name in enumerate(["posE", "posN", "posU"]):
        mae = np.mean(np.abs(errors[:, i]))
        print(f"  {name}: MAE={mae:.4f} m")

    # Also evaluate at specific horizons
    for horizon_s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        h_steps = int(horizon_s / DT)
        if h_steps <= T:
            d = dist_3d[h_steps - 1]
            print(f"  @{horizon_s:.1f}s (step {h_steps}): 3D error = {d:.4f} m")

    return pred_raw, gt_raw, errors, dist_3d


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
    out = out_dir / "wam_v2_training_loss.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_dream_3d(pred_raw, gt_raw, out_dir: Path):
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
        title=f"WAM v2 Dream Rollout ({len(gt_raw)} steps = {len(gt_raw)*DT:.1f}s)",
        scene=dict(
            xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Up (m)",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.15),
        ),
        width=1100, height=800,
    )
    out = out_dir / "wam_v2_dream_3d.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_dream_topdown(pred_raw, gt_raw, out_dir: Path):
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
        title="WAM v2 Dream Rollout (Top-Down)",
        xaxis_title="East (m)", yaxis_title="North (m)",
        width=1000, height=800,
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
    )
    out = out_dir / "wam_v2_dream_topdown.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_dream_error_over_time(dist_3d, out_dir: Path):
    """3D error accumulation over dream horizon."""
    t = np.arange(len(dist_3d)) * DT  # seconds
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=dist_3d, mode="lines",
                             name="3D Error", line=dict(color="purple", width=2)))
    fig.update_layout(
        title="Dream Rollout: 3D Position Error vs Time",
        xaxis_title="Time Horizon (s)", yaxis_title="3D Error (m)",
        width=900, height=400,
    )
    out = out_dir / "wam_v2_dream_error_vs_time.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_state_comparison(pred_raw, gt_raw, out_dir: Path):
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
        title="WAM v2 Dream: All State Dimensions",
        height=250 * n, width=1100,
    )
    out = out_dir / "wam_v2_dream_states.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


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
    out = out_dir / "wam_v2_action_intervention.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


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
    train_segs, val_segs, val_raw_segs, norm = load_multi_lap_data(
        RAW_DIR, TRAIN_RATIO, downsample=DOWNSAMPLE)

    train_ds = MultiLapSequenceDataset(train_segs, SEQ_LEN)
    val_ds = MultiLapSequenceDataset(val_segs, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"  State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"  Seq len: {SEQ_LEN} ({SEQ_LEN * DT:.1f}s)")
    print(f"  Train sequences: {len(train_ds)}")
    print(f"  Val sequences: {len(val_ds)}")
    print(f"  Scheduled Sampling: epoch {SS_START_EPOCH}-{SS_END_EPOCH}, max={SS_MAX_RATIO}")
    print(f"  Curriculum rollout: {ROLLOUT_MIN}->{ROLLOUT_MAX} steps "
          f"({ROLLOUT_MIN*DT:.1f}s->{ROLLOUT_MAX*DT:.1f}s), weight={ROLLOUT_WEIGHT}")
    print(f"  Dream val: {DREAM_VAL_STEPS} steps ({DREAM_VAL_STEPS*DT:.1f}s) "
          f"every {DREAM_VAL_EVERY} epochs")
    print(f"  No early stopping. Checkpoints every {SAVE_EVERY} epochs.")
    print(f"  Device: {DEVICE}")

    # 2) Build model
    print("\n[2/6] Building Physics-Informed WAM v2...")
    model = WAM(STATE_DIM, ACTION_DIM, LATENT_DIM, HIDDEN_DIM, GRU_LAYERS,
                DROPOUT, dt=DT)
    model.set_norm(norm["s_mean"], norm["s_std"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Latent={LATENT_DIM}, Hidden={HIDDEN_DIM}, GRU layers={GRU_LAYERS}")
    print(f"  Physics prior: kinematic bicycle model (dt={DT}s)")
    print(f"  Parameters: {n_params:,}")

    # Prepare val segment for dream monitoring
    val_seg_n = val_segs[0]  # first lap's validation segment (normalized)

    # 3) Train
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[3/6] Training with curriculum rollout + scheduled sampling...")
    config = {
        "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "lr_patience": LR_PATIENCE, "device": DEVICE,
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
            "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
            "gru_layers": GRU_LAYERS, "dropout": DROPOUT,
            "state_cols": STATE_COLS, "action_cols": ACTION_COLS,
        },
    }, model_path)
    print(f"  Model saved to {model_path}")

    # Use first validation segment (Lap_0) for evaluation
    s_val_n = val_segs[0][0]
    a_val_n = val_segs[0][1]
    s_val_raw = val_raw_segs[0][0]

    # 4) Evaluate (teacher forcing)
    print("\n[4/6] Evaluating (teacher forcing, one-step)...")
    evaluate_teacher_forcing(model, s_val_n, a_val_n, s_val_raw, norm, DEVICE)

    # 5) Evaluate (dream rollout)
    print("\n[5/6] Evaluating (dream rollout)...")
    pred_raw, gt_raw, errors, dist_3d = evaluate_dream(
        model, s_val_n, a_val_n, s_val_raw, norm, DEVICE, DREAM_STEPS)

    # 6) Visualize
    print("\n[6/6] Generating visualizations...")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, VIS_DIR)
    plot_dream_3d(pred_raw, gt_raw, VIS_DIR)
    plot_dream_topdown(pred_raw, gt_raw, VIS_DIR)
    plot_dream_error_over_time(dist_3d, VIS_DIR)
    plot_state_comparison(pred_raw, gt_raw, VIS_DIR)
    plot_action_intervention(model, s_val_n, a_val_n, norm, DEVICE, VIS_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Model:          {model_path}")
    print(f"  Visualizations: {VIS_DIR}/wam_v2_*.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
