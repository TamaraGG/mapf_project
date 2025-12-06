import math
from typing import List, Tuple, Dict
from agent_state import AgentState, AgentStatus
from grid_map import GridMap
from reservation_table import ReservationTable
from a_star_solver import EnergyAwareAStar
from task_manager import Task, TaskManager

class CentralPlanner:
    def __init__(self, 
                 grid: GridMap, 
                 reservation_table: ReservationTable,
                 gamma: float = 0.8,
                 safety_buffer: float = 15.0,
                 charger_capacity: int = 1):
        self.grid = grid
        self.rt = reservation_table
        self.solver = EnergyAwareAStar(grid, reservation_table)
        self.gamma = gamma
        self.safety_buffer = safety_buffer
        self.charger_capacity = charger_capacity
        self.PLAN_HORIZON = 100  # Увеличили горизонт

    def _sigmoid(self, x: float) -> float:
        if x < -100: return 0.0
        if x > 100: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def update_priorities_and_goals(self, agents: List[AgentState], task_manager: TaskManager):
        priorities = []
        for agent in agents:
            if agent.is_dead: continue
            
            # 1. Реальное расстояние до зарядки
            dist_to_charger = self.grid.get_nearest_charger_dist(agent.pos)
            
            # Если агент замурован или карта сломана
            if dist_to_charger == float('inf'):
                dist_to_charger = 1000.0 

            has_payload = (agent.current_task is not None and agent.status == AgentStatus.WORKING)
            consumption = agent.profile.calculate_move_cost(has_payload)
            
            # Стоимость добраться до зарядки
            ret_cost = dist_to_charger * consumption
            
            # Оценка запаса хода (omega)
            # Если omega < 1.0, значит мы приближаемся к критической точке
            omega = (agent.battery - ret_cost) / (self.safety_buffer * 2.0)

            is_emergency = False
            need_proactive_charge = False # <--- НОВЫЙ ФЛАГ

            # 1. Если уже едем заряжаться - продолжаем
            if agent.status == AgentStatus.TO_CHARGER: 
                is_emergency = True
            
            # 2. Если заряд критический (выживание)
            elif omega < 1.0 and agent.status != AgentStatus.CHARGING:
                is_emergency = True
            
            # 3. Если не хватает на текущую задачу (выживание)
            elif agent.current_task and agent.current_task.id != -1:
                # ... (ваш код проверки total_needed) ...
                # (оставьте этот блок как был в исправлении)
                pass 

            # 4. [НОВОЕ] ПРОАКТИВНАЯ ЗАРЯДКА
            # Если агент свободен (IDLE), но заряд ниже порога работы (40%),
            # или просто низкий (например, < 60%) и нет задач - едем заряжаться.
            elif agent.status == AgentStatus.IDLE:
                # Порог должен совпадать с тем, что в task_manager (0.4)
                work_threshold = agent.profile.battery_capacity * 0.45 
                if agent.battery < work_threshold:
                    need_proactive_charge = True

            # ПРИМЕНЕНИЕ РЕШЕНИЯ
            # Если нужна зарядка (аварийная ИЛИ проактивная)
            if (is_emergency or need_proactive_charge) and agent.status != AgentStatus.CHARGING and agent.status != AgentStatus.TO_CHARGER:
                if agent.current_task and agent.current_task.id != -1:
                    task_manager.return_task(agent.current_task)
                
                agent.status = AgentStatus.TO_CHARGER
                nearest = self._find_nearest_charger_pos(agent.pos)
                agent.current_task = Task(-1, agent.pos, nearest, 1.0)
                
                # Если это просто подзарядка, а не авария, не задираем приоритет до небес
                if need_proactive_charge and not is_emergency:
                    is_emergency = False # Пусть планируется в общем порядке, но с целью "Зарядка"

            # РАСЧЕТ ПРИОРИТЕТА
            p_safe = 1.0 - self._sigmoid(omega)
            p_task = agent.current_task.importance if agent.current_task else 0.0
            
            # Если авария - приоритет 2000. Если проактивная зарядка - обычный приоритет.
            val = (2000 + p_safe) if is_emergency else (self.gamma * p_safe + (1-self.gamma)*p_task)
            val += agent.id * 0.001 
            
            priorities.append((val, agent))
        
        priorities.sort(key=lambda x: x[0], reverse=True)
        return [a for p, a in priorities]

    def plan(self, agents: List[AgentState], current_time: int, task_manager: TaskManager):
        sorted_agents = self.update_priorities_and_goals(agents, task_manager)
        
        # Пересоздаем таблицу резервации на этот тик
        self.rt = ReservationTable()
        self.solver.reservation_table = self.rt
        
        # Словарь для хранения "статичных" путей (если агент не сможет найти путь)
        static_reservations = {}

        # 1. Сначала резервируем места для всех как "бетон" (на случай неудачи)
        for a in agents:
            if a.is_dead: continue
            # Резервируем текущую позицию на весь горизонт
            static_path = [(a.pos[0], a.pos[1], current_time + t) for t in range(self.PLAN_HORIZON)]
            static_reservations[a.id] = static_path
            self.rt.reserve_path(static_path, a.id)

        # 2. Планируем по приоритету
        for agent in sorted_agents:
            if agent.is_dead: continue

            # Снимаем статическую бронь для текущего агента
            self.rt.clear_path(static_reservations[agent.id], agent.id)

            # --- СЦЕНАРИЙ: ЗАРЯДКА (Уже на станции) ---
            if agent.status == AgentStatus.CHARGING:
                needed = agent.profile.battery_capacity - agent.battery
                ticks = int(needed / agent.profile.charging_speed) + 2
                wait_path = [(agent.pos[0], agent.pos[1], current_time + t) for t in range(ticks)]
                self.rt.reserve_path(wait_path, agent.id)
                agent.assign_path(wait_path)
                continue

            # --- СЦЕНАРИЙ: ЕСТЬ ВАЛИДНЫЙ ПУТЬ ---
            # Если у агента уже есть путь, и он ведет куда надо, попробуем его оставить
            # Это экономит CPU и делает поведение стабильным
            if agent.path and agent.current_task:
                # Проверяем, не устарел ли путь (начинается ли он с current_time)
                # В agent.path хранятся (x, y, t). Нам нужно проверить соответствие времени.
                # Для простоты в этой реализации мы перепланируем, если путь пуст или изменилась цель.
                # Но чтобы LNS работал, мы должны доверять agent.path, если он был задан LNS.
                # В данной реализации plan() вызывается ДО LNS, поэтому мы строим базовые пути.
                pass 

            # --- СЦЕНАРИЙ: НЕТ ЗАДАЧИ ---
            if not agent.current_task or (agent.current_task.id == -1 and agent.pos == agent.current_task.goal_pos):
                # Стоим на месте (возвращаем статику)
                self.rt.reserve_path(static_reservations[agent.id], agent.id)
                agent.assign_path([]) 
                continue

            # --- СЦЕНАРИЙ: ПОИСК ПУТИ ---
            is_working = (agent.status == AgentStatus.WORKING)
            
            path = self.solver.find_path(
                agent.id, agent.pos, agent.current_task.goal_pos, agent.profile,
                current_time, agent.battery, self.safety_buffer, 
                energy_lambda=0.5, charger_capacity=self.charger_capacity,
                has_payload=is_working
            )

            if path:
                # 1. Генерируем "хвост" ожидания (Wait Extension)
                last_x, last_y, last_t = path[-1]
                horizon_end_time = current_time + self.PLAN_HORIZON
                wait_extension = []
                
                # Продлеваем бронь до конца горизонта планирования
                for t in range(last_t + 1, horizon_end_time):
                    wait_extension.append((last_x, last_y, t))
                
                # 2. Объединяем путь движения и ожидание
                full_path = path + wait_extension
                
                # 3. Резервируем ПОЛНЫЙ путь в таблице
                self.rt.reserve_path(full_path, agent.id)

                # 4. Присваиваем ПОЛНЫЙ путь агенту (исключая текущую позицию t=0)
                # Теперь LNS сможет корректно очистить весь зарезервированный хвост,
                # так как он хранится внутри agent.path
                if len(full_path) > 1:
                    agent.assign_path(full_path[1:])
                else:
                    agent.assign_path([])

            else:
                # 1. Резервируем в таблице, что агент стоит на месте
                self.rt.reserve_path(static_reservations[agent.id], agent.id)
                
                # 2. [ВАЖНО] Присваиваем этот "стоячий" путь агенту.
                # Теперь агент знает, что он планирует стоять.
                # И LNS, если выберет этого агента, увидит этот путь и сможет его очистить перед перепланированием.
                agent.assign_path(static_reservations[agent.id][1:])

    def _find_nearest_charger_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        # Используем предрасчитанную карту для мгновенного поиска
        nearest = self.grid.get_nearest_charger_pos(pos)
        if nearest:
            return nearest
        
        # Fallback (на случай если агент замурован и BFS не дошел)
        return pos