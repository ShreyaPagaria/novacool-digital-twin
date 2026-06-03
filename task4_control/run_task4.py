import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
from shared.utils import load_workload, ambient_temp, ensure_output_dir
from shared.crah import CRAHUnit
from task1_thermal_model.thermal_model import ThermalModel
from task2_power_pue.power_model import compute_power_metrics
from control import BaselinePolicy, ReactivePolicy, SafeOptimizerPolicy

WORKLOAD = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\data\workload_trace.csv"
OUT = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\outputs"
ensure_output_dir(OUT)

def run_policy(policy, name):
    workload = load_workload(WORKLOAD)
    crahs = [CRAHUnit(i, chw_supply_temp=9.0, air_flow_rate=80.0, chw_flow_rate=200.0) for i in range(4)]
    model = ThermalModel(crahs)
    records = []
    last_thermal = {"t_cold_mean":9, "t_hot_mean":26, "t_outlet_peak":26}
    for minute, (ts, row) in enumerate(workload.iterrows()):
        state = {"minute": minute, **last_thermal}
        model.crahs = policy(state, model.crahs)
        thermal = model.step(row.values, ambient_temp(minute))
        power = compute_power_metrics(row.values, model.crahs)
        last_thermal = thermal
        records.append({"policy": name, "timestamp": ts, "minute": minute, **thermal, **power,
                        "violations": int((thermal["rack_outlet"] > 40).sum()),
                        "supply_mean": sum(c.chw_supply_temp for c in model.crahs)/4,
                        "flow_mean": sum(c.air_flow_rate for c in model.crahs)/4})
    return pd.DataFrame(records)

results = {
    "Baseline": run_policy(BaselinePolicy(), "Baseline"),
    "Reactive": run_policy(ReactivePolicy(), "Reactive"),
    "Safe optimizer": run_policy(SafeOptimizerPolicy(), "Safe optimizer"),
}
summary = []
for name, df in results.items():
    df.to_csv(f"{OUT}/task4_{name.replace(' ','_').lower()}.csv", index=False)
    summary.append({"IT Load Served MWh": df.it_load_kw.sum() / 60000, "Policy": name, "Mean PUE": df.pue.mean(), "Thermal violations": df.violations.sum(),
                    "Peak outlet C": df.t_outlet_peak.max(), "Cooling energy MWh": df.cooling_infra_kw.sum()/60000})
summary = pd.DataFrame(summary); summary.to_csv(f"{OUT}/task4_control_summary.csv", index=False); print(summary)

fig, ax = plt.subplots(4,1,figsize=(13,11),sharex=True); fig.suptitle("Part B · Task 4 — Control Policy Comparison")
for name, df in results.items():
    h=df.minute/60; ax[0].plot(h, df.t_outlet_peak, label=name); ax[1].plot(h, df.pue, label=name); ax[2].plot(h, df.supply_mean, label=name); ax[3].plot(h, df.cooling_infra_kw, label=name)
ax[0].axhline(40, ls="--", label="40°C limit")
for a,y in zip(ax,["Peak outlet °C","PUE","Supply temp °C","Cooling infra kW"]): a.set_ylabel(y); a.grid(True,alpha=.3); a.legend()
ax[3].set_xlabel("Hour")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_task4_control.png", dpi=150)
print(f"Saved {OUT}/fig4_task4_control.png")
