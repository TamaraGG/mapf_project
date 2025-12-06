import math
from typing import List, Tuple, Dict
from common.agent_state import AgentState, AgentStatus
from common.grid_map import GridMap
from centralized.reservation_table import ReservationTable
from centralized.a_star_solver import EnergyAwareAStar
from common.task_manager import TaskManager, Task

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
        self.PLAN_HORIZON = 60 # Ограничиваем горизонт для скорости

    def _sigmoid(self, x: float) -> float:
        if x < -100: return 0.0
        if x > 100: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def update_priorities_and_goals(self, agents: List[AgentState], task_manager: TaskManager):
        priorities = []
        for agent in agents:
            if agent.is_dead: continue
            
            # 1. Оценка критичности заряда
            dist_to_charger = self.grid.get_nearest_charger_dist(agent.pos)
            if dist_to_charger == float('inf'): dist_to_charger = 1000.0 

            # Расход на возврат считаем "пустым" (без груза)
            consumption = agent.profile.calculate_move_cost(has_payload=False)
            ret_cost = dist_to_charger * consumption
            
            # Omega: сколько "буферов безопасности" у нас осталось
            omega = (agent.battery - ret_cost) / (self.safety_buffer + 1e-5)

            is_emergency = False
            
            # Логика переключения в режим зарядки
            if agent.status == AgentStatus.TO_CHARGER: 
                is_emergency = True # Уже едем, приоритет высокий
            elif omega < 1.0 and agent.status != AgentStatus.CHARGING:
                is_emergency = True # Критически мало заряда
            elif agent.status == AgentStatus.IDLE:
                # Проактивная зарядка: если делать нечего и заряд < 60%
                if agent.battery < agent.profile.battery_capacity * 0.6:
                    is_emergency = True # Не совсем emergency, но отправим на зарядку

            # Назначение цели "Зарядка"
            if is_emergency and agent.status != AgentStatus.CHARGING and agent.status != AgentStatus.TO_CHARGER:
                # Если была задача - возвращаем в пул
                if agent.current_task and agent.current_task.id != -1:
                    task_manager.return_task(agent.current_task)
                    agent.has_picked_up = False
                
                agent.status = AgentStatus.TO_CHARGER
                nearest = self._find_nearest_charger_pos(agent.pos)
                if nearest:
                    # Создаем фиктивную задачу "ехать к розетке"
                    agent.current_task = Task(-1, agent.pos, nearest, 1.0)
            
            # Расчет приоритета P(a)
            p_safe = 1.0 - self._sigmoid(omega) # Чем меньше омега, тем выше p_safe
            p_task = agent.current_task.importance if agent.current_task else 0.0
            
            # Если авария - приоритет сверхвысокий (>1000)
            val = (2000.0 + p_safe) if is_emergency else (self.gamma * p_safe + (1-self.gamma)*p_task)
            
            priorities.append((val, agent))
        
        # Сортировка: от большего приоритета к меньшему
        priorities.sort(key=lambda x: x[0], reverse=True)
        return [a for p, a in priorities]

    def plan(self, agents: List[AgentState], current_time: int, task_manager: TaskManager):
        # 1. Обновляем приоритеты и цели (кто едет на зарядку, кто работает)
        sorted_agents = self.update_priorities_and_goals(agents, task_manager)
        
        # 2. Сброс таблицы занятости
        self.rt = ReservationTable()
        self.solver.rt = self.rt 
        
        static_reservations = {}

        # 3. [ВАЖНО] Инициализация: Все агенты считаются статическими препятствиями
        # Мы резервируем их текущие позиции на весь горизонт планирования.
        # Это гарантирует, что если агент с низким приоритетом еще не получил план,
        # агент с высоким приоритетом не проедет сквозь него.
        for a in agents:
            if a.is_dead:
                path = [(a.pos[0], a.pos[1], current_time + t) for t in range(self.PLAN_HORIZON)]
            else:
                path = [(a.pos[0], a.pos[1], current_time + t) for t in range(self.PLAN_HORIZON)]
            
            self.rt.reserve_path(path, a.id)
            static_reservations[a.id] = path

        # 4. Последовательное планирование
        for agent in sorted_agents:
            if agent.is_dead: continue

            # А. Удаляем "стену", которую мы поставили на шаге 3 для этого агента
            self.rt.clear_path(static_reservations[agent.id], agent.id)

            # Б. Если агент уже на зарядке - он просто занимает слот
            if agent.status == AgentStatus.CHARGING:
                needed = agent.profile.battery_capacity - agent.battery
                ticks = int(needed / agent.profile.charging_speed) + 5
                wait_path = [(agent.pos[0], agent.pos[1], current_time + t) for t in range(ticks)]
                self.rt.reserve_path(wait_path, agent.id)
                agent.assign_path(wait_path)
                continue

            # В. Если нет задачи - стоим на месте (резервируем точку)
            target_pos = agent.current_task.goal_pos if agent.current_task else agent.pos
            
            # Определяем, куда ехать: к старту задачи или к финишу?
            if agent.status == AgentStatus.WORKING:
                if not agent.has_picked_up:
                    target_pos = agent.current_task.start_pos
                else:
                    target_pos = agent.current_task.goal_pos
            
            # Если мы уже в цели (например, приехали на погрузку, но еще не забрали)
            if agent.pos == target_pos:
                # Путь длиной 1 (стоять)
                path = [(agent.pos[0], agent.pos[1], current_time + 1)]
            else:
                # Г. Поиск пути A*
                carrying = (agent.status == AgentStatus.WORKING and agent.has_picked_up)
                
                path = self.solver.find_path(
                    agent.id, agent.pos, target_pos, agent.profile,
                    current_time, agent.battery, self.safety_buffer, 
                    energy_lambda=0.5, charger_capacity=self.charger_capacity,
                    has_payload=carrying
                )

            # Д. Обработка результата
            if path:
                # Резервируем найденный путь
                self.rt.reserve_path(path, agent.id)
                
                # Резервируем конечную точку на будущее (чтобы никто не занял место, куда мы приедем)
                last_x, last_y, last_t = path[-1]
                wait_extension = [(last_x, last_y, t) for t in range(last_t + 1, current_time + self.PLAN_HORIZON)]
                self.rt.reserve_path(wait_extension, agent.id)
                
                # Назначаем путь агенту (без текущей точки t=0, так как агент уже там)
                if len(path) > 1:
                    agent.assign_path(path[1:])
                else:
                    agent.assign_path([])
            else:
                # Е. Путь не найден (тупик/пробка) -> Восстанавливаем "стену"
                # Агент будет стоять и ждать следующего тика
                self.rt.reserve_path(static_reservations[agent.id], agent.id)
                agent.assign_path([])

    def _find_nearest_charger_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        nearest = self.grid.get_nearest_charger_pos(pos)
        if nearest: return nearest
        return pos