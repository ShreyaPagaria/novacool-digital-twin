"""Task 1 — Thermal model for NovaCool.

Assumptions:
- Lumped parameter model, not CFD.
- Facility has 4 rows x 50 racks; one CRAH per row.
- Nearly all IT electrical power becomes heat.
- Rack outlet temperature uses Q = m_dot * cp * ΔT.
- Cold aisle temperature is approximated by CRAH supply setpoint.

# Initial thermal state calibrated from sensor_reference.csv:
# inlet_temp_c mean ≈ 9°C  → cold aisle / rack inlet temperature
# outlet_temp_c mean ≈ 26°C → hot aisle / rack outlet temperature
"""
from __future__ import annotations
import numpy as np
from shared.constants import N_RACKS, N_ROWS, RACKS_PER_ROW, CP_AIR, KW_TO_W
from shared.crah import CRAHUnit

class ThermalModel:
    def __init__(self, crahs=None):
        self.crahs = crahs or [CRAHUnit(i) for i in range(N_ROWS)]
        self.t_cold_rows = np.full(N_ROWS, 9.0)
        self.t_hot_rows = np.full(N_ROWS, 26.0)

    def step(self, rack_power_kw: np.ndarray, ambient_c: float) -> dict:
        rack_outlet = np.zeros(N_RACKS)
        cold_rows, hot_rows = np.zeros(N_ROWS), np.zeros(N_ROWS)
        total_cooling_capacity_kw = 0.0

        for r in range(N_ROWS):
            s = slice(r * RACKS_PER_ROW, (r + 1) * RACKS_PER_ROW)
            row_power = rack_power_kw[s]
            q_row_kw = float(row_power.sum())
            crah = self.crahs[r]

            # Cold aisle temperature
            t_cold = crah.chw_supply_temp

            # Row-level hot aisle temperature rise.
            delta_t_row = (q_row_kw * KW_TO_W) / (crah.air_flow_rate * CP_AIR)
            t_hot = t_cold + delta_t_row

            # Small ambient leakage / imperfect containment.
            t_hot = 0.95 * t_hot + 0.05 * ambient_c

            # Per-rack outlet temp. Airflow is distributed evenly across racks in the row.
            airflow_per_rack = crah.air_flow_rate / RACKS_PER_ROW
            rack_delta_t = (row_power * KW_TO_W) / (airflow_per_rack * CP_AIR)
            rack_outlet[s] = t_cold + rack_delta_t

            cold_rows[r] = t_cold
            hot_rows[r] = t_hot
            total_cooling_capacity_kw += crah.cooling_capacity_kw(t_hot)

        self.t_cold_rows = cold_rows
        self.t_hot_rows = hot_rows
        return {
            "t_cold_mean": float(cold_rows.mean()),
            "t_hot_mean": float(hot_rows.mean()),
            "t_outlet_mean": float(rack_outlet.mean()),
            "t_outlet_peak": float(rack_outlet.max()),
            "rack_outlet": rack_outlet,
            "crah_cooling_capacity_kw": float(total_cooling_capacity_kw),
        }
