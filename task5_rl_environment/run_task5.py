import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import DataCenterEnv

OUT = r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\outputs"; os.makedirs(OUT, exist_ok=True)
env = DataCenterEnv(r"C:\Users\pagar\Downloads\HammerheadAI\novacool_task_based\data\workload_trace.csv", seed=42)
obs = env.reset()
rows=[]
rng = np.random.default_rng(42)
# Sanity check with zero/stabilizing-ish random small actions rather than extreme random actions.
for _ in range(1440):
    # action = np.random.default_rng(42).normal(0, 0.05, env.ACTION_DIM)
    action = rng.normal(0, 0.05, env.ACTION_DIM)
    obs, reward, done, info = env.step(action)
    rows.append({"step": env.step_idx, "reward": reward, "pue": info["pue"], "peak_outlet": info["t_outlet_peak"], "violations": info["violations"]})
    if done: break

df=pd.DataFrame(rows); df.to_csv(f"{OUT}/task5_rl_rollout.csv", index=False)
h=df.step/60
fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True); fig.suptitle("Part B · Task 5 — RL Environment Sanity Rollout")
ax[0].plot(h,df.reward); ax[0].set_ylabel("Reward"); ax[0].grid(True,alpha=.3)
ax[1].plot(h,df.peak_outlet); ax[1].axhline(40,ls="--",label="40°C limit"); ax[1].set_ylabel("Peak outlet °C"); ax[1].legend(); ax[1].grid(True,alpha=.3)
ax[2].plot(h,df.pue); ax[2].set_ylabel("PUE"); ax[2].set_xlabel("Hour"); ax[2].grid(True,alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_task5_rl_env.png", dpi=150)
print(df.describe())
print(f"Saved {OUT}/fig5_task5_rl_env.png")
