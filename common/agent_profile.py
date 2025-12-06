from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class AgentProfile:
    type_name: str
    battery_capacity: float
    base_consumption: float
    idle_consumption: float
    charging_speed: float
    speed: int
    payload_factor: float

    def calculate_move_cost(self, has_payload: bool = False) -> float:
        cost = self.base_consumption
        if has_payload:
            cost *= (1.0 + self.payload_factor)
        return cost

AGENT_PROFILES: Dict[str, AgentProfile] = {
    "HeavyTruck": AgentProfile(
        type_name="HeavyTruck",
        battery_capacity=1000.0,
        base_consumption=5.0,
        idle_consumption=0.1,
        charging_speed=20.0,
        speed=1,
        payload_factor=0.5
    ),
    "FastDrone": AgentProfile(
        type_name="FastDrone",
        battery_capacity=350.0,
        base_consumption=1.5,
        idle_consumption=0.5,
        charging_speed=25.0,
        speed=1,
        payload_factor=0.1
    ),
    "StandardBot": AgentProfile(
        type_name="StandardBot",
        battery_capacity=500.0,
        base_consumption=2.5,
        idle_consumption=0.2,
        charging_speed=10.0,
        speed=1,
        payload_factor=0.25
    )
}