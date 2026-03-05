"""
Fit Baseline Track from Multiple Laps

从多圈 GPS 数据中提取基准赛道中心线:
1. 选取完整圈 (排除半圈数据)
2. 以最封闭的 Lap_9 为参考, 按弧长均匀重采样
3. 其余圈投影到参考圈, 按弧长对齐
4. 跨圈求均值得到基准中心线
5. 平滑处理
6. 保存为 CSV + 可视化

输出:
  - data/raw_data/baseline_track.csv
  - visualization/raw/baseline_track_3d.html
  - visualization/raw/baseline_track_topdown.html
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import plotly.graph_objects as go

RAW_DIR = Path("/home/shiyuqi/CascadeProjects/wam/autodrive/data/raw_data")
VIS_DIR = Path("/home/shiyuqi/CascadeProjects/wam/autodrive/visualization/raw")

# Laps to exclude (partial data)
EXCLUDE_LAPS = {"_5", "_10"}

RESAMPLE_SPACING_M = 0.5  # baseline resolution: 0.5m per point


def load_laps(raw_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Load all complete laps, return list of (name, positions[N,3])."""
    files = sorted(raw_dir.glob("VehicleDynamicsDataset_Nov2023_2023-11*.csv"))
    laps = []
    for f in files:
        suffix = f.stem.replace("VehicleDynamicsDataset_Nov2023_2023-11", "")
        if suffix in EXCLUDE_LAPS:
            print(f"  skip {f.name} (partial lap)")
            continue
        df = pd.read_csv(f, comment="#")
        pos = df[["posE_m", "posN_m", "posU_m"]].values
        name = f"Lap{suffix}" if suffix else "Lap_0"
        laps.append((name, pos))
    return laps


def compute_arc_length(pos: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length for position array [N,3]."""
    ds = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])
    return s


def resample_by_arc_length(pos: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a trajectory to uniform arc-length spacing."""
    s = compute_arc_length(pos)
    # Remove duplicate arc-length values (stationary points)
    unique_mask = np.diff(s, prepend=-1) > 0
    s = s[unique_mask]
    pos = pos[unique_mask]
    total_len = s[-1]
    n_pts = int(total_len / spacing)
    s_new = np.linspace(0, total_len, n_pts, endpoint=False)

    interp_E = interp1d(s, pos[:, 0], kind="cubic")
    interp_N = interp1d(s, pos[:, 1], kind="cubic")
    interp_U = interp1d(s, pos[:, 2], kind="cubic")

    pos_new = np.column_stack([interp_E(s_new), interp_N(s_new), interp_U(s_new)])
    return pos_new


def align_lap_to_reference(ref: np.ndarray, lap: np.ndarray) -> np.ndarray:
    """
    For each reference point, find the nearest point on the lap.
    Returns lap positions aligned to the reference indexing [N_ref, 3].
    Uses 2D (E,N) for matching, returns full 3D.
    """
    tree = cKDTree(lap[:, :2])  # match on 2D plan view
    _, idx = tree.query(ref[:, :2])
    return lap[idx]


def fit_baseline(laps: list[tuple[str, np.ndarray]], spacing: float) -> np.ndarray:
    """
    Fit baseline track by averaging aligned laps.
    1. Resample all laps by arc length
    2. Use Lap_9 (most closed) as reference
    3. Align all other laps to reference
    4. Average across laps
    5. Smooth
    """
    # Resample all laps
    resampled = {}
    for name, pos in laps:
        rs = resample_by_arc_length(pos, spacing)
        resampled[name] = rs
        print(f"  {name}: {len(pos)} pts -> {len(rs)} resampled pts")

    # Use Lap_9 as reference (smallest start-end gap = 10.1m)
    ref_name = "Lap_9"
    if ref_name not in resampled:
        # fallback: use first lap
        ref_name = list(resampled.keys())[0]
    ref = resampled[ref_name]
    print(f"  Reference: {ref_name} ({len(ref)} pts)")

    # Align all laps to reference
    aligned_positions = [ref]  # include reference itself
    for name, rs in resampled.items():
        if name == ref_name:
            continue
        aligned = align_lap_to_reference(ref, rs)
        aligned_positions.append(aligned)
        dist = np.mean(np.sqrt(np.sum((aligned[:, :2] - ref[:, :2]) ** 2, axis=1)))
        print(f"  {name}: mean 2D alignment error = {dist:.2f} m")

    # Average across all aligned laps
    stacked = np.stack(aligned_positions, axis=0)  # [n_laps, n_pts, 3]
    baseline = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    print(f"  Averaged {len(aligned_positions)} laps, "
          f"mean std: E={std[:,0].mean():.2f}m, N={std[:,1].mean():.2f}m, U={std[:,2].mean():.2f}m")

    # Smooth with Savitzky-Golay filter
    window = 51  # ~25m window at 0.5m spacing
    if len(baseline) > window:
        baseline[:, 0] = savgol_filter(baseline[:, 0], window, 3)
        baseline[:, 1] = savgol_filter(baseline[:, 1], window, 3)
        baseline[:, 2] = savgol_filter(baseline[:, 2], window, 3)

    return baseline, std


def save_baseline(baseline: np.ndarray, out_path: Path) -> None:
    """Save baseline track as CSV with arc length."""
    s = compute_arc_length(baseline)
    df = pd.DataFrame({
        "s_m": s,
        "posE_m": baseline[:, 0],
        "posN_m": baseline[:, 1],
        "posU_m": baseline[:, 2],
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved baseline track: {out_path} ({len(df)} pts, {s[-1]:.0f}m)")
    return df


def visualize_baseline_3d(baseline: np.ndarray, laps: list[tuple[str, np.ndarray]]) -> None:
    """Interactive 3D comparison: baseline vs original laps."""
    fig = go.Figure()

    # Plot original laps (thin, transparent)
    colors = [
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    for i, (name, pos) in enumerate(laps):
        ds = pos[::50]  # heavy downsample for background
        fig.add_trace(go.Scatter3d(
            x=ds[:, 0], y=ds[:, 1], z=ds[:, 2],
            mode="lines",
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.4,
        ))

    # Plot baseline (thick, bold)
    ds = baseline[::5]
    fig.add_trace(go.Scatter3d(
        x=ds[:, 0], y=ds[:, 1], z=ds[:, 2],
        mode="lines",
        name="Baseline Track",
        line=dict(color="red", width=6),
        hovertemplate=(
            "<b>Baseline</b><br>"
            "E: %{x:.1f} m<br>N: %{y:.1f} m<br>U: %{z:.1f} m<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Baseline Track vs Original Laps (3D)",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.15),
        ),
        width=1200, height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out = VIS_DIR / "baseline_track_3d.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def visualize_baseline_topdown(baseline: np.ndarray, laps: list[tuple[str, np.ndarray]]) -> None:
    """Interactive 2D top-down comparison."""
    fig = go.Figure()

    colors = [
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    for i, (name, pos) in enumerate(laps):
        ds = pos[::50]
        fig.add_trace(go.Scatter(
            x=ds[:, 0], y=ds[:, 1],
            mode="lines",
            name=name,
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.4,
        ))

    ds = baseline[::5]
    fig.add_trace(go.Scatter(
        x=ds[:, 0], y=ds[:, 1],
        mode="lines",
        name="Baseline Track",
        line=dict(color="red", width=3),
    ))

    # Mark start/direction
    fig.add_trace(go.Scatter(
        x=[baseline[0, 0]], y=[baseline[0, 1]],
        mode="markers+text",
        marker=dict(size=12, color="green", symbol="diamond"),
        text=["START"], textposition="top center",
        name="Start",
    ))

    fig.update_layout(
        title="Baseline Track vs Original Laps (Top-Down)",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        width=1100, height=900,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )

    out = VIS_DIR / "baseline_track_topdown.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def visualize_baseline_profile(baseline_df: pd.DataFrame) -> None:
    """Elevation and curvature profile along the track."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=baseline_df["s_m"],
        y=baseline_df["posU_m"],
        mode="lines",
        name="Elevation",
        line=dict(color="blue", width=2),
    ))

    fig.update_layout(
        title="Baseline Track - Elevation Profile",
        xaxis_title="Arc Length (m)",
        yaxis_title="Elevation / Up (m)",
        width=1100, height=400,
    )

    out = VIS_DIR / "baseline_elevation_profile.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")


def main():
    print("[1/5] Loading laps...")
    laps = load_laps(RAW_DIR)
    print(f"  Loaded {len(laps)} complete laps")

    print("[2/5] Fitting baseline track...")
    baseline, std = fit_baseline(laps, RESAMPLE_SPACING_M)

    print("[3/5] Saving baseline track...")
    baseline_df = save_baseline(baseline, RAW_DIR / "baseline_track.csv")

    print("[4/5] Visualizing baseline (3D)...")
    visualize_baseline_3d(baseline, laps)

    print("[5/5] Visualizing baseline (top-down + profile)...")
    visualize_baseline_topdown(baseline, laps)
    visualize_baseline_profile(baseline_df)

    print("\nDone!")
    print(f"  Baseline track: {len(baseline)} points, "
          f"total length: {compute_arc_length(baseline)[-1]:.0f} m")
    print(f"  CSV: {RAW_DIR / 'baseline_track.csv'}")
    print(f"  Visualizations: {VIS_DIR}/baseline_*.html")


if __name__ == "__main__":
    main()
