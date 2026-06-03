"""Task 3 — Validation against sensor_reference.csv.

Assumptions:
- Sensor data is synthetic reference truth.
- We validate per-minute facility mean/peak temperatures.
- Metrics: RMSE, MAE, max absolute error, bias.
"""
import numpy as np
import pandas as pd

def load_reference(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.groupby("timestamp").agg(
        ref_inlet_mean=("inlet_temp_c", "mean"),
        ref_outlet_mean=("outlet_temp_c", "mean"),
        ref_outlet_peak=("outlet_temp_c", "max"),
        ref_it_kw=("pdu_power_kw", "sum"),
    ).reset_index()

def error_metrics(pred, actual, label):
    err = np.asarray(pred) - np.asarray(actual)
    return {"label": label, "rmse": np.sqrt(np.mean(err**2)), "mae": np.mean(np.abs(err)),
            "max_abs": np.max(np.abs(err)), "bias": np.mean(err)}
