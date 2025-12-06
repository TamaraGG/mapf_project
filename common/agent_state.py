from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto

class AgentStatus(Enum):
    IDLE = auto()
    WORKING = auto()
    TO_CHARGER = auto()
    CHARGING = auto()
    DEAD = auto()

@dataclass
class AgentState:
    id: int
    profile: 'AgentProfile' # type: ignore
    pos: Tuple[int, int]
    battery: float = field(init=False)
    status: AgentStatus = AgentStatus.IDLE
    
    # Поля для Centralized
    path: List[Tuple[int, int, int]] = field(default_factory=list)
    
    # Поля для Decentralized (RL)
    target_pos: Optional[Tuple[int, int]] = None
    prev_target_pos: Optional[Tuple[int, int]] = None
    
    current_task: Optional['Task'] = None # type: ignore
    has_picked_up: bool = False 
    total_energy_consumed: float = 0.0

    def __post_init__(self):
        self.battery = self.profile.battery_capacity

    @property
    def is_dead(self) -> bool:
        return self.status == AgentStatus.DEAD

    def assign_path(self, new_path: List[Tuple[int, int, int]]):
        self.path = new_path

    def step(self, grid_map):
        if self.status == AgentStatus.DEAD: return

        energy_cost = 0.0
        
        # Логика движения по пути (Centralized)
        if self.path:
            next_node = self.path.pop(0)
            nx, ny, _ = next_node
            if (nx, ny) != self.pos:
                carrying = (self.status == AgentStatus.WORKING and self.has_picked_up)
                energy_cost = self.profile.calculate_move_cost(has_payload=carrying)
                self.pos = (nx, ny)
            else:
                energy_cost = self.profile.idle_consumption
        else:
            # Если пути нет, но мы не на зарядке - тратим idle
            if self.status != AgentStatus.CHARGING:
                energy_cost = self.profile.idle_consumption

        # Логика зарядки (Общая)
        if grid_map.is_charger(self.pos):
            if self.status == AgentStatus.TO_CHARGER:
                self.status = AgentStatus.CHARGING
            elif self.status == AgentStatus.IDLE and self.battery < self.profile.battery_capacity:
                 # Автозарядка если стоим на станции
                 self.status = AgentStatus.CHARGING
            
            if self.status == AgentStatus.CHARGING:
                energy_cost = 0.0 
                self.battery = min(self.battery + self.profile.charging_speed, self.profile.battery_capacity)
                if self.battery >= self.profile.battery_capacity:
                    self.status = AgentStatus.IDLE
                    # Если была задача "ехать на зарядку" (id=-1), сбрасываем её
                    if self.current_task and self.current_task.id == -1:
                        self.current_task = None

        self.battery -= energy_cost
        self.total_energy_consumed += energy_cost

        if self.battery <= 0:
            self.battery = 0
            self.status = AgentStatus.DEAD
            self.path = []