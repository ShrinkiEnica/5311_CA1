"""
Vehicle Dynamics Raw Data - Interactive 3D Position Visualization

从 raw_data CSV 中提取 GPS 位置 (posE_m, posN_m, posU_m)，
生成可交互的 3D 轨迹图 (Plotly HTML)。

输出到: /autodrive/visualization/raw/
"""

import glob
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RAW_DIR = Path("/home/shiyuqi/CascadeProjects/wam/autodrive/data/raw_data")
OUT_DIR = Path("/home/shiyuqi/CascadeProjects/wam/autodrive/visualization/raw")


def load_all_laps(raw_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    files = sorted(raw_dir.glob("*.csv"))
    laps = []
    for f in files:
        df = pd.read_csv(f, comment="#")
        name = f.stem.replace("VehicleDynamicsDataset_Nov2023_2023-11", "Lap")
        if name == "Lap":
            name = "Lap_0"
        laps.append((name, df))
    return laps


def make_all_laps_3d(laps: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    """All laps overlaid in one interactive 3D plot, colored by lap."""
    fig = go.Figure()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]

    for i, (name, df) in enumerate(laps):
        color = colors[i % len(colors)]
        # Downsample for performance (every 10th point = 20Hz)
        ds = df.iloc[::10]
        fig.add_trace(go.Scatter3d(
            x=ds["posE_m"], y=ds["posN_m"], z=ds["posU_m"],
            mode="lines",
            name=name,
            line=dict(color=color, width=3),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "E: %{x:.1f} m<br>N: %{y:.1f} m<br>U: %{z:.1f} m<br>"
                "<extra></extra>"
            ),
        ))
        # Start marker
        fig.add_trace(go.Scatter3d(
            x=[df["posE_m"].iloc[0]], y=[df["posN_m"].iloc[0]], z=[df["posU_m"].iloc[0]],
            mode="markers",
            marker=dict(size=5, color=color, symbol="diamond"),
            name=f"{name} start",
            showlegend=False,
        ))

    fig.update_layout(
        title="All Laps - 3D GPS Position (posE, posN, posU)",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.15),
        ),
        legend=dict(x=0.01, y=0.99),
        width=1200, height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  saved {out_path}")


def make_single_lap_3d(name: str, df: pd.DataFrame, out_path: Path) -> None:
    """Single lap with speed coloring."""
    ds = df.iloc[::5]  # 40Hz for single lap
    speed = (ds["Vx_mps"] ** 2 + ds["Vy_mps"] ** 2).pow(0.5)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=ds["posE_m"], y=ds["posN_m"], z=ds["posU_m"],
        mode="lines+markers",
        marker=dict(
            size=2,
            color=speed,
            colorscale="Turbo",
            colorbar=dict(title="Speed (m/s)", x=0.95),
            showscale=True,
        ),
        line=dict(color="gray", width=1),
        hovertemplate=(
            f"<b>{name}</b><br>"
            "t: %{customdata[0]:.2f} s<br>"
            "E: %{x:.1f} m<br>N: %{y:.1f} m<br>U: %{z:.1f} m<br>"
            "Speed: %{customdata[1]:.1f} m/s<br>"
            "<extra></extra>"
        ),
        customdata=list(zip(ds["t_s"], speed)),
    ))

    # Start / End
    fig.add_trace(go.Scatter3d(
        x=[df["posE_m"].iloc[0]], y=[df["posN_m"].iloc[0]], z=[df["posU_m"].iloc[0]],
        mode="markers", marker=dict(size=8, color="green", symbol="diamond"),
        name="Start", showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[df["posE_m"].iloc[-1]], y=[df["posN_m"].iloc[-1]], z=[df["posU_m"].iloc[-1]],
        mode="markers", marker=dict(size=8, color="red", symbol="diamond"),
        name="End", showlegend=True,
    ))

    fig.update_layout(
        title=f"{name} - 3D Position (colored by speed)",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.15),
        ),
        width=1100, height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def make_2d_top_view(laps: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    """Interactive 2D top-down view (East vs North)."""
    fig = go.Figure()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]

    for i, (name, df) in enumerate(laps):
        color = colors[i % len(colors)]
        ds = df.iloc[::10]
        fig.add_trace(go.Scatter(
            x=ds["posE_m"], y=ds["posN_m"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "E: %{x:.1f} m<br>N: %{y:.1f} m<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="All Laps - Top-Down View (East vs North)",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        width=1100, height=900,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  saved {out_path}")


def main():
    print("Loading raw data...")
    laps = load_all_laps(RAW_DIR)
    print(f"  Loaded {len(laps)} laps")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. All laps 3D
    print("[1/4] All laps 3D overview...")
    make_all_laps_3d(laps, OUT_DIR / "all_laps_3d.html")

    # 2. Individual lap 3D (speed-colored)
    print("[2/4] Individual lap 3D plots...")
    ind_dir = OUT_DIR / "individual"
    for name, df in laps:
        make_single_lap_3d(name, df, ind_dir / f"{name}_3d.html")
    print(f"  saved {len(laps)} plots in {ind_dir}")

    # 3. 2D top-down view
    print("[3/4] 2D top-down view...")
    make_2d_top_view(laps, OUT_DIR / "all_laps_topdown.html")

    # 4. Summary stats
    print("[4/4] Summary...")
    for name, df in laps:
        speed = (df["Vx_mps"] ** 2 + df["Vy_mps"] ** 2).pow(0.5)
        print(f"  {name}: {len(df)} pts, {df['t_s'].max():.1f}s, "
              f"max_speed={speed.max():.1f} m/s ({speed.max()*3.6:.0f} km/h), "
              f"max_ay={df['ayCG_mps2'].abs().max():.1f} m/s2")

    print(f"\nDone! Interactive visualizations in: {OUT_DIR}")


if __name__ == "__main__":
    main()
