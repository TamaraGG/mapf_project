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
    path: List[Tuple[int, int, int]] = field(default_factory=list)
    current_task: Optional['Task'] = None # type: ignore
    total_energy_consumed: float = 0.0

    def __post_init__(self):
        self.battery = self.profile.battery_capacity

    @property
    def is_dead(self) -> bool:
        return self.status == AgentStatus.DEAD

    def assign_path(self, new_path: List[Tuple[int, int, int]]):
        self.path = new_path
        if self.path and self.status == AgentStatus.IDLE:
            self.status = AgentStatus.WORKING

    def step(self, grid_map):
        if self.status == AgentStatus.DEAD: return

        energy_cost = 0.0
        
        if self.path:
            # Извлекаем следующий шаг (он соответствует текущему тику симуляции)
            next_node = self.path.pop(0)
            nx, ny, _ = next_node
            
            # Проверяем: это движение или ожидание?
            if (nx, ny) != self.pos:
                # ДВИЖЕНИЕ
                has_payload = (self.current_task is not None and self.status == AgentStatus.WORKING)
                energy_cost = self.profile.calculate_move_cost(has_payload)
                self.pos = (nx, ny)
            else:
                # ОЖИДАНИЕ (даже если это часть пути)
                # Агент стоит на месте, удерживая позицию (согласно wait_extension)
                energy_cost = self.profile.idle_consumption
                
                # [ВАЖНО] Если мы достигли цели задачи, но путь продолжается (хвост),
                # мы не должны тратить энергию как "WORKING" (с грузом), если груз уже сдан.
                # Но логика сдачи груза обычно в main.py. 
                # Здесь мы просто тратим idle_consumption, что верно для любого статуса.
        else:
            # Путь кончился (или был пуст)
            if self.status != AgentStatus.CHARGING:
                energy_cost = self.profile.idle_consumption
                # Если мы не заряжаемся и пути нет -> мы IDLE
                if self.status != AgentStatus.DEAD:
                    self.status = AgentStatus.IDLE

        # --- Логика зарядки (исправленная по предыдущему замечанию) ---
        if grid_map.is_charger(self.pos):
            if self.status == AgentStatus.TO_CHARGER:
                self.status = AgentStatus.CHARGING
            elif self.status == AgentStatus.IDLE and self.battery < self.profile.battery_capacity:
                 self.status = AgentStatus.CHARGING
            
            if self.status == AgentStatus.CHARGING:
                energy_cost = 0.0 # На зарядке энергия не тратится (или тратится, но восполняется)
                self.battery = min(self.battery + self.profile.charging_speed, self.profile.battery_capacity)
                if self.battery >= self.profile.battery_capacity:
                    self.status = AgentStatus.IDLE

        self.battery -= energy_cost
        self.total_energy_consumed += energy_cost

        if self.battery <= 0:
            self.battery = 0
            self.status = AgentStatus.DEAD
            self.path = []