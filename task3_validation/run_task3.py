import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
from validation import load_reference, error_metrics

SIM = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\outputs\task1_thermal_results.csv"
SENSOR = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\data\sensor_reference.csv"
OUT = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\outputs"

sim = pd.read_csv(SIM, parse_dates=["timestamp"])
ref = load_reference(SENSOR)
n = min(len(sim), len(ref))
merged = pd.DataFrame({
    "minute": sim["minute"].values[:n],
    "sim_inlet_mean": sim["t_cold_mean"].values[:n],
    "ref_inlet_mean": ref["ref_inlet_mean"].values[:n],
    "sim_outlet_mean": sim["t_outlet_mean"].values[:n],
    "ref_outlet_mean": ref["ref_outlet_mean"].values[:n],
    "sim_outlet_peak": sim["t_outlet_peak"].values[:n],
    "ref_outlet_peak": ref["ref_outlet_peak"].values[:n],
})
metrics = [
    error_metrics(merged.sim_outlet_mean, merged.ref_outlet_mean, "Mean outlet temp"),
    error_metrics(merged.sim_outlet_peak, merged.ref_outlet_peak, "Peak outlet temp"),
    error_metrics(merged.sim_inlet_mean, merged.ref_inlet_mean, "Mean inlet temp"),
]
pd.DataFrame(metrics).to_csv(f"{OUT}/task3_validation_metrics.csv", index=False)
merged.to_csv(f"{OUT}/task3_validation_merged.csv", index=False)
print(pd.DataFrame(metrics))

h = merged["minute"] / 60
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("Part A · Task 3 — Validation vs Sensor Reference")
ax[0].plot(h, merged.sim_outlet_mean, label="Simulation"); ax[0].plot(h, merged.ref_outlet_mean, "--", label="Reference"); ax[0].fill_between(h, merged.sim_outlet_mean, merged.ref_outlet_mean, alpha=.15); ax[0].set_ylabel("Outlet °C"); ax[0].legend(); ax[0].grid(True, alpha=.3)
ax[1].plot(h, merged.sim_inlet_mean, label="Simulation"); ax[1].plot(h, merged.ref_inlet_mean, "--", label="Reference"); ax[1].set_ylabel("Inlet °C"); ax[1].set_xlabel("Hour"); ax[1].legend(); ax[1].grid(True, alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_task3_validation.png", dpi=150)
print(f"Saved {OUT}/fig3_task3_validation.png")
