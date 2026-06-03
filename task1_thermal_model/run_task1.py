import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
from shared.utils import load_workload, ambient_temp, ensure_output_dir
from shared.crah import CRAHUnit
from thermal_model import ThermalModel

#  Please change the paths below as needed for your local setup. 
WORKLOAD = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\data\workload_trace.csv"
OUT = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\outputs"
ensure_output_dir(OUT)

workload = load_workload(WORKLOAD)
model = ThermalModel([CRAHUnit(i, chw_supply_temp=9.0, air_flow_rate=80.0) for i in range(4)])
records = []
for minute, (ts, row) in enumerate(workload.iterrows()):
    thermal = model.step(row.values, ambient_temp(minute))
    records.append({"timestamp": ts, "minute": minute, "it_load_kw": row.sum(), **thermal})

df = pd.DataFrame(records)
df.to_csv(f"{OUT}/task1_thermal_results.csv", index=False)

h = df["minute"] / 60
fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Part A · Task 1 — Thermal Simulation")
ax[0].plot(h, df["t_cold_mean"], label="Average cold aisle"); ax[0].set_ylabel("°C"); ax[0].legend(); ax[0].grid(True, alpha=.3)
ax[1].plot(h, df["t_hot_mean"], label="Average hot aisle"); ax[1].set_ylabel("°C"); ax[1].legend(); ax[1].grid(True, alpha=.3)
ax[2].plot(h, df["t_outlet_peak"], label="Peak rack outlet"); ax[2].axhline(40, ls="--", label="40°C limit"); ax[2].set_ylabel("°C"); ax[2].legend(); ax[2].grid(True, alpha=.3)


ax[3].plot(h, df["it_load_kw"], label="Heat to remove ≈ IT load"); ax[3].set_ylabel("kW"); ax[3].set_xlabel("Hour"); ax[3].legend(); ax[3].grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_task1_thermal.png", dpi=150)
print(df.head())
print(f"Saved {OUT}/fig1_task1_thermal.png and task1_thermal_results.csv")

