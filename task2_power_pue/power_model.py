"""Task 2 — Power path and PUE model.

Assumptions:
- UPS efficiency is a quadratic curve peaking near 60% load.
- Cooling infrastructure power = CRAH fan power + pump power.
- Fan power follows cubic law with airflow.
- PUE = total facility power / IT load.
"""
import numpy as np
from shared.constants import IT_CAPACITY_KW

def ups_efficiency(load_fraction: float) -> float:
    x = float(np.clip(load_fraction, 0.0, 1.2))
    eta = -0.5 * (x - 0.6) ** 2 + 0.97
    return float(np.clip(eta, 0.80, 0.98))

def compute_power_metrics(rack_power_kw, crahs):
    it_load_kw = float(np.sum(rack_power_kw))
    eta = ups_efficiency(it_load_kw / IT_CAPACITY_KW)
    ups_input_kw = it_load_kw / eta if it_load_kw > 0 else 0.0
    ups_loss_kw = ups_input_kw - it_load_kw
    cooling_infra_kw = sum(c.fan_power_kw + c.pump_power_kw for c in crahs)
    total_facility_kw = it_load_kw + ups_loss_kw + cooling_infra_kw
    pue = total_facility_kw / it_load_kw if it_load_kw > 0 else np.nan
    return dict(it_load_kw=it_load_kw, ups_efficiency=eta, ups_loss_kw=ups_loss_kw,
                cooling_infra_kw=cooling_infra_kw, total_facility_kw=total_facility_kw, pue=pue)
