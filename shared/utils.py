"""Shared data loading and ambient assumptions."""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_workload(path: str) -> pd.DataFrame:
    """Return workload matrix: index=timestamp, columns=rack_id, values=power_kw."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "rack_id"])
    return df.pivot(index="timestamp", columns="rack_id", values="power_kw").ffill().fillna(0.0)


def ambient_temp(minute: int, t_min: float = 18.0, t_max: float = 33.0) -> float:
    """Assumption: sinusoidal outside dry-bulb temperature over 24h, peak near 14:00."""
    phase = (minute / 1440.0) * 2 * np.pi - np.pi / 2
    return float(t_min + (t_max - t_min) * (0.5 + 0.5 * np.sin(phase - np.pi / 4)))


def ensure_output_dir(path="outputs"):
    import os
    os.makedirs(path, exist_ok=True)
