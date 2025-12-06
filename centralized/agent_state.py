from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto

class AgentStatus(Enum):
    IDLE = auto()
    WORKING = auto()      # Едет к задаче или выполняет её
    TO_CHARGER = auto()   # Едет заряжаться
    CHARGING = auto()     # Стоит на зарядке
    DEAD = auto()         # Батарея 0

@dataclass
class AgentState:
    id: int
    profile: 'AgentProfile' # type: ignore
    pos: Tuple[int, int]
    battery: float = field(init=False)
    status: AgentStatus = AgentStatus.IDLE
    path: List[Tuple[int, int, int]] = field(default_factory=list)
    current_task: Optional['Task'] = None # type: ignore
    
    # [НОВОЕ] Флаг: груз уже на борту?
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
        
        # 1. Движение по пути
        if self.path:
            # Берем следующую точку (t+1)
            next_node = self.path.pop(0)
            nx, ny, _ = next_node
            
            if (nx, ny) != self.pos:
                # Реальное перемещение
                # Груз учитываем только если мы WORKING и уже забрали его
                carrying = (self.status == AgentStatus.WORKING and self.has_picked_up)
                energy_cost = self.profile.calculate_move_cost(has_payload=carrying)
                self.pos = (nx, ny)
            else:
                # Ожидание (Wait)
                energy_cost = self.profile.idle_consumption
        else:
            # Пути нет
            if self.status != AgentStatus.CHARGING:
                energy_cost = self.profile.idle_consumption

        # 2. Логика зарядки
        if grid_map.is_charger(self.pos):
            # Если приехали заряжаться или просто стоим разряженные на зарядке
            if self.status == AgentStatus.TO_CHARGER:
                self.status = AgentStatus.CHARGING
            
            if self.status == AgentStatus.CHARGING:
                # На зарядке потребление 0 (или компенсируется)
                energy_cost = 0.0 
                self.battery = min(self.battery + self.profile.charging_speed, self.profile.battery_capacity)
                
                # Если зарядились - освобождаем слот (статус IDLE)
                if self.battery >= self.profile.battery_capacity:
                    self.status = AgentStatus.IDLE
                    self.current_task = None # Сбрасываем задачу "ехать на зарядку"

        # 3. Списание энергии
        self.battery -= energy_cost
        self.total_energy_consumed += energy_cost

        if self.battery <= 0:
            self.battery = 0
            self.status = AgentStatus.DEAD
            self.path = []