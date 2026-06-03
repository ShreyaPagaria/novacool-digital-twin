"""Task 5 — Gym-style RL environment.

Observation dimension = 15
Action dimension = 8

"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from shared.utils import load_workload, ambient_temp
from shared.crah import CRAHUnit
from shared.constants import IT_CAPACITY_KW
from task1_thermal_model.thermal_model import ThermalModel
from task2_power_pue.power_model import compute_power_metrics

class DataCenterEnv:
    OBS_DIM = 15
    ACTION_DIM = 8
    def __init__(self, workload_path="data/workload_trace.csv", seed=0):
        self.workload = load_workload(workload_path).values.astype(float)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cum_violations = 0
        self.crahs = [CRAHUnit(i, chw_supply_temp=9.0, air_flow_rate=80.0, chw_flow_rate=200.0) for i in range(4)]
        self.model = ThermalModel(self.crahs)
        self.last = {"t_cold_mean":9.0, "t_hot_mean":26.0, "t_outlet_peak":26.0}
        return self._obs()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        for i,c in enumerate(self.crahs):
            c.chw_supply_temp = float(np.clip(c.chw_supply_temp + 1.0*action[i], 8.0, 22.0))
            c.air_flow_rate = float(np.clip(c.air_flow_rate + 5.0*action[i+4], 50.0, 120.0))
        rack_power = self.workload[self.step_idx]
        thermal = self.model.step(rack_power, ambient_temp(self.step_idx))
        power = compute_power_metrics(rack_power, self.crahs)
        violations = int((thermal["rack_outlet"] > 40.0).sum())
        self.cum_violations += violations
        it_norm = power["it_load_kw"] / IT_CAPACITY_KW
        reward = it_norm - 0.5*max(power["pue"]-1,0) - 5.0*violations - 0.3*max(thermal["t_outlet_peak"]-38,0)
        self.last = thermal
        self.step_idx += 1
        done = self.step_idx >= len(self.workload) or self.cum_violations > 500
        info = {**thermal, **power, "violations": violations, "cum_violations": self.cum_violations}
        return self._obs(), float(reward), bool(done), info

    def _obs(self):
        idx = min(self.step_idx, len(self.workload)-1)
        hour = (idx//60) % 24
        rack_power = self.workload[idx]
        return np.array([
            self.last["t_cold_mean"], self.last["t_hot_mean"], self.last["t_outlet_peak"],
            rack_power.sum()/IT_CAPACITY_KW, (ambient_temp(idx)-15)/25,
            np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
            *[c.chw_supply_temp for c in self.crahs], *[c.air_flow_rate for c in self.crahs]
        ], dtype=np.float32)
