import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw_data/VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
print("Vy min/max:", df["Vy_mps"].min(), df["Vy_mps"].max())
print("Vx min/max:", df["Vx_mps"].min(), df["Vx_mps"].max())

