import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
from shared.utils import load_workload, ensure_output_dir
from shared.crah import CRAHUnit
from power_model import compute_power_metrics

WORKLOAD = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\data\workload_trace.csv"
OUT = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\outputs"
ensure_output_dir(OUT)

workload = load_workload(WORKLOAD)
crahs = [CRAHUnit(i, chw_supply_temp=9.0, air_flow_rate=80.0, chw_flow_rate=200.0) for i in range(4)]
records = []
for minute, (ts, row) in enumerate(workload.iterrows()):
    records.append({"timestamp": ts, "minute": minute, **compute_power_metrics(row.values, crahs)})

df = pd.DataFrame(records)
df.to_csv(f"{OUT}/task2_power_pue_results.csv", index=False)

h = df["minute"] / 60
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Part A · Task 2 — Power & PUE")
ax[0].stackplot(h, df["it_load_kw"], df["ups_loss_kw"], df["cooling_infra_kw"], labels=["IT load", "UPS losses", "Cooling infra"]); ax[0].set_ylabel("kW"); ax[0].legend(); ax[0].grid(True, alpha=.3)
ax[1].plot(h, df["ups_efficiency"] * 100); ax[1].set_ylabel("UPS η (%)"); ax[1].grid(True, alpha=.3)
ax[2].plot(h, df["pue"], label=f"Mean PUE={df['pue'].mean():.3f}"); ax[2].set_ylabel("PUE"); ax[2].set_xlabel("Hour"); ax[2].legend(); ax[2].grid(True, alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_task2_pue.png", dpi=150)
print(df[["it_load_kw", "ups_loss_kw", "cooling_infra_kw", "pue"]].describe())
print(f"Saved {OUT}/fig2_task2_pue.png and task2_power_pue_results.csv")
