"""
Train WAM (World Action Model) for Vehicle Dynamics

从车辆动力学传感器数据预测车辆位置:
  Input  (55 dim): 速度、姿态、加速度、转向、制动、动力、轮速、温度、悬架、赛道几何等
  Output (3 dim):  posE_m, posN_m, posU_m

数据: VehicleDynamicsDataset_Nov2023_2023-11.csv (Lap 0, 22304 帧)
切分: 按时间顺序 70% train / 30% val (不打乱)

用法:
  python code/train_wam.py
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
DATA_FILE = ROOT / "data" / "raw_data" / "VehicleDynamicsDataset_Nov2023_2023-11.csv"
MODEL_DIR = ROOT / "models"
VIS_DIR = ROOT / "visualization" / "wam_v1"

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_COLS = ["posE_m", "posN_m", "posU_m"]

TRAIN_RATIO = 0.7
BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIMS = [256, 256, 128]
DROPOUT = 0.1
PATIENCE = 30          # early stopping
LR_PATIENCE = 10       # ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class VehicleDynamicsDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ─── Model ───────────────────────────────────────────────────────────────────

class WAM_MLP(nn.Module):
    """Multi-layer perceptron for World Action Model."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Data Loading & Preprocessing ────────────────────────────────────────────

def load_and_split(data_file: Path, train_ratio: float):
    """Load CSV, separate input/output, normalize, split by time order."""
    print(f"  Loading {data_file.name}...")
    df = pd.read_csv(data_file, comment="#")
    print(f"  Shape: {df.shape}")

    # Separate input and output
    input_cols = [c for c in df.columns if c not in OUTPUT_COLS]
    X_raw = df[input_cols].values.astype(np.float64)
    Y_raw = df[OUTPUT_COLS].values.astype(np.float64)

    print(f"  Input features: {len(input_cols)}")
    print(f"  Output features: {len(OUTPUT_COLS)}")

    # Time-ordered split
    n = len(df)
    split_idx = int(n * train_ratio)
    print(f"  Split: train={split_idx}, val={n - split_idx} "
          f"(t_split={df['t_s'].iloc[split_idx]:.1f}s)")

    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    Y_train_raw, Y_val_raw = Y_raw[:split_idx], Y_raw[split_idx:]

    # Z-score normalize (fit on train only)
    X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    Y_mean, Y_std = Y_train_raw.mean(axis=0), Y_train_raw.std(axis=0)

    # Avoid division by zero for constant columns
    X_std[X_std < 1e-8] = 1.0
    Y_std[Y_std < 1e-8] = 1.0

    X_train = (X_train_raw - X_mean) / X_std
    X_val = (X_val_raw - X_mean) / X_std
    Y_train = (Y_train_raw - Y_mean) / Y_std
    Y_val = (Y_val_raw - Y_mean) / Y_std

    norm_params = {
        "X_mean": X_mean, "X_std": X_std,
        "Y_mean": Y_mean, "Y_std": Y_std,
        "input_cols": input_cols,
    }

    return (X_train, Y_train, X_val, Y_val,
            Y_train_raw, Y_val_raw, norm_params, df)


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay,
                lr_patience, patience, device):
    """Train with early stopping and LR scheduling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f} | "
                  f"lr={current_lr:.1e}{marker}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improve for {patience} epochs)")
            break

    # Load best model
    model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, X_val, Y_val_raw, norm_params, device):
    """Evaluate model on validation set, return predictions in original scale."""
    model.eval()
    X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()

    # Denormalize
    Y_mean, Y_std = norm_params["Y_mean"], norm_params["Y_std"]
    pred = pred_norm * Y_std + Y_mean

    # Metrics
    errors = pred - Y_val_raw
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))
    dist_3d = np.sqrt(np.sum(errors ** 2, axis=1))
    mae_3d = np.mean(dist_3d)
    max_3d = np.max(dist_3d)

    print("\n" + "=" * 50)
    print("Validation Metrics (original scale)")
    print("=" * 50)
    print(f"  MSE:           {mse:.4f} m²")
    print(f"  MAE:           {mae:.4f} m")
    print(f"  Max Error:     {max_err:.4f} m")
    print(f"  3D MAE:        {mae_3d:.4f} m")
    print(f"  3D Max Error:  {max_3d:.4f} m")

    per_dim = ["posE_m", "posN_m", "posU_m"]
    for i, name in enumerate(per_dim):
        dim_mae = np.mean(np.abs(errors[:, i]))
        dim_max = np.max(np.abs(errors[:, i]))
        print(f"  {name}: MAE={dim_mae:.4f} m, Max={dim_max:.4f} m")

    return pred, errors, dist_3d


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_training_history(history, out_dir: Path):
    """Loss curves."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Loss", "Learning Rate"],
                        vertical_spacing=0.08)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             name="Train Loss", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             name="Val Loss", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["lr"],
                             name="LR", line=dict(color="green")), row=2, col=1)

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(title="WAM Training History", height=600, width=900)

    out = out_dir / "wam_training_loss.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_predictions_3d(Y_val_raw, pred, out_dir: Path):
    """3D trajectory comparison: ground truth vs prediction."""
    fig = go.Figure()

    # Ground truth
    ds = 5
    fig.add_trace(go.Scatter3d(
        x=Y_val_raw[::ds, 0], y=Y_val_raw[::ds, 1], z=Y_val_raw[::ds, 2],
        mode="lines", name="Ground Truth",
        line=dict(color="blue", width=4),
    ))

    # Prediction
    fig.add_trace(go.Scatter3d(
        x=pred[::ds, 0], y=pred[::ds, 1], z=pred[::ds, 2],
        mode="lines", name="WAM Prediction",
        line=dict(color="red", width=4),
    ))

    fig.update_layout(
        title="WAM Prediction vs Ground Truth (Validation Set, 3D)",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.15),
        ),
        width=1100, height=800,
    )

    out = out_dir / "wam_prediction_3d.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_predictions_topdown(Y_val_raw, pred, out_dir: Path):
    """2D top-down trajectory comparison."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=Y_val_raw[:, 0], y=Y_val_raw[:, 1],
        mode="lines", name="Ground Truth",
        line=dict(color="blue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pred[:, 0], y=pred[:, 1],
        mode="lines", name="WAM Prediction",
        line=dict(color="red", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="WAM Prediction vs Ground Truth (Top-Down)",
        xaxis_title="East (m)", yaxis_title="North (m)",
        width=1000, height=800,
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
    )

    out = out_dir / "wam_prediction_topdown.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def plot_error_analysis(errors, dist_3d, df_val, out_dir: Path):
    """Error distribution and along-track error."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Per-Dimension Error Distribution",
            "3D Distance Error Distribution",
            "Error Along Time",
            "Per-Dimension Error Along Time",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.1,
    )

    dim_names = ["posE_m", "posN_m", "posU_m"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 1) Per-dim histogram
    for i, (name, color) in enumerate(zip(dim_names, colors)):
        fig.add_trace(go.Histogram(
            x=errors[:, i], name=name, marker_color=color,
            opacity=0.7, nbinsx=80,
        ), row=1, col=1)

    # 2) 3D distance histogram
    fig.add_trace(go.Histogram(
        x=dist_3d, name="3D Error", marker_color="purple",
        nbinsx=80,
    ), row=1, col=2)

    # 3) 3D error along time
    t_val = df_val["t_s"].values
    fig.add_trace(go.Scatter(
        x=t_val, y=dist_3d, mode="lines",
        name="3D Error", line=dict(color="purple", width=1),
    ), row=2, col=1)

    # 4) Per-dim error along time
    for i, (name, color) in enumerate(zip(dim_names, colors)):
        fig.add_trace(go.Scatter(
            x=t_val, y=np.abs(errors[:, i]), mode="lines",
            name=f"|{name} err|", line=dict(color=color, width=1),
        ), row=2, col=2)

    fig.update_layout(
        title="WAM Error Analysis (Validation Set)",
        height=800, width=1200,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Error (m)", row=1, col=1)
    fig.update_xaxes(title_text="3D Error (m)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="3D Error (m)", row=2, col=1)
    fig.update_yaxes(title_text="|Error| (m)", row=2, col=2)

    out = out_dir / "wam_error_analysis.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WAM Training — Vehicle Dynamics World Action Model")
    print("=" * 60)

    # 1) Load & preprocess
    print("\n[1/5] Loading and preprocessing data...")
    (X_train, Y_train, X_val, Y_val,
     Y_train_raw, Y_val_raw, norm_params, df) = load_and_split(DATA_FILE, TRAIN_RATIO)

    split_idx = int(len(df) * TRAIN_RATIO)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    train_ds = VehicleDynamicsDataset(X_train, Y_train)
    val_ds = VehicleDynamicsDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"  Device: {DEVICE}")

    # 2) Build model
    print("\n[2/5] Building WAM model...")
    model = WAM_MLP(input_dim, output_dim, HIDDEN_DIMS, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {input_dim} -> {HIDDEN_DIMS} -> {output_dim}")
    print(f"  Parameters: {n_params:,}")
    print(f"  {model}")

    # 3) Train
    print("\n[3/5] Training...")
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        lr_patience=LR_PATIENCE, patience=PATIENCE, device=DEVICE,
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "wam_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm_params": {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in norm_params.items()},
        "config": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
        },
    }, model_path)
    print(f"  Model saved to {model_path}")

    # 4) Evaluate
    print("\n[4/5] Evaluating on validation set...")
    pred, errors, dist_3d = evaluate(model, X_val, Y_val_raw, norm_params, DEVICE)

    # 5) Visualize
    print("\n[5/5] Generating visualizations...")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, VIS_DIR)
    plot_predictions_3d(Y_val_raw, pred, VIS_DIR)
    plot_predictions_topdown(Y_val_raw, pred, VIS_DIR)
    plot_error_analysis(errors, dist_3d, df_val, VIS_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Model:          {model_path}")
    print(f"  Visualizations: {VIS_DIR}/wam_*.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
