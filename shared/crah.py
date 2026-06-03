"""CRAH unit model shared across tasks.

"""
from dataclasses import dataclass
from .constants import CP_WATER, KW_TO_W

@dataclass
class CRAHUnit:
    unit_id: int
    chw_supply_temp: float = 9.0     # °C, air supply/control proxy
    chw_flow_rate: float = 200.0      # kg/s chilled water proxy
    air_flow_rate: float = 80.0      # kg/s air flow through row
    fan_power_nom: float = 80.0       # kW at 160 kg/s
    pump_power_per_kgs: float = 0.02  # kW per kg/s

    def cooling_capacity_kw(self, t_return_c: float) -> float:
        q_kw = self.chw_flow_rate * CP_WATER * max(t_return_c - self.chw_supply_temp, 0.0) / KW_TO_W
        return float(max(q_kw, 0.0))

    @property
    def fan_power_kw(self) -> float:
        return float(self.fan_power_nom * (self.air_flow_rate / 160.0) ** 3)

    @property
    def pump_power_kw(self) -> float:
        return float(self.pump_power_per_kgs * self.chw_flow_rate)
