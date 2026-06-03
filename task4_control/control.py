"""Task 4 — Control policies.

Objective & Constraints
• Maximize total IT load served across all racks over the 24-hour window.
• Hard constraint: No rack outlet temperature may exceed 40 °C at any time step.
• Soft constraint: Minimize total cooling energy consumed (PUE should improve, not degrade)

Assumptions:
- Workload is given, so all policies serve the same IT demand unless admission control is added.
- Control objective here is safe operation + lower cooling energy/PUE.
- Controls: CRAH supply temperature and airflow.
"""
from __future__ import annotations
import copy
import numpy as np
from shared.constants import OUTLET_TEMP_LIMIT_C

# Operating bounds around calibrated 9°C supply temperature.
# Allow limited overcooling and relaxation for control experiments.

SUPPLY_MIN, SUPPLY_MAX = 7.0, 12.0 
FLOW_MIN, FLOW_MAX = 50.0, 120.0 #the calibrated airflow is 80 so using bounds around the calibrated value

class BaselinePolicy:
    def __call__(self, state, crahs):
        return crahs

class ReactivePolicy:
    def __call__(self, state, crahs):
        new = copy.deepcopy(crahs)
        peak = state["t_outlet_peak"]
        for c in new:
            if peak > 38.0:
                c.chw_supply_temp = max(SUPPLY_MIN, c.chw_supply_temp - 0.5)
                c.air_flow_rate = min(FLOW_MAX, c.air_flow_rate + 5.0)
            elif peak < 34.0:
                c.chw_supply_temp = min(SUPPLY_MAX, c.chw_supply_temp + 0.2)
                c.air_flow_rate = max(FLOW_MIN, c.air_flow_rate - 2.0)
        return new

class SafeOptimizerPolicy:
    """Rule-based safety optimizer that relaxes cooling when safe and increases cooling near the thermal limit."""
    def __call__(self, state, crahs):
        current_peak = state["t_outlet_peak"]
        new = copy.deepcopy(crahs)
        for c in new:
            if current_peak > 39.0:
                c.chw_supply_temp = max(SUPPLY_MIN, c.chw_supply_temp - 1.0)
                c.air_flow_rate = min(FLOW_MAX, c.air_flow_rate + 10.0)
            elif current_peak < 36.0:
                c.chw_supply_temp = min(SUPPLY_MAX, c.chw_supply_temp + 0.3)
                c.air_flow_rate = max(FLOW_MIN, c.air_flow_rate - 3.0)
        return new
