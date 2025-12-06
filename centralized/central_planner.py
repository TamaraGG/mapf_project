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
        self.PLAN_HORIZON = 100

    def _sigmoid(self, x: float) -> float:
        if x < -100: return 0.0
        if x > 100: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def update_priorities_and_goals(self, agents: List[AgentState], task_manager: TaskManager):
        priorities = []
        for agent in agents:
            if agent.is_dead: continue
            
            dist_to_charger = self.grid.get_nearest_charger_dist(agent.pos)
            if dist_to_charger == float('inf'): dist_to_charger = 1000.0 

            has_payload = (agent.current_task is not None and agent.status == AgentStatus.WORKING)
            consumption = agent.profile.calculate_move_cost(has_payload)
            ret_cost = dist_to_charger * consumption
            
            omega = (agent.battery - ret_cost) / (self.safety_buffer * 2.0)

            is_emergency = False
            need_proactive_charge = False

            if agent.status == AgentStatus.TO_CHARGER: 
                is_emergency = True
            elif omega < 1.0 and agent.status != AgentStatus.CHARGING:
                is_emergency = True
            elif agent.status == AgentStatus.IDLE:
                work_threshold = agent.profile.battery_capacity * 0.45 
                if agent.battery < work_threshold:
                    need_proactive_charge = True

            if (is_emergency or need_proactive_charge) and agent.status != AgentStatus.CHARGING and agent.status != AgentStatus.TO_CHARGER:
                if agent.current_task and agent.current_task.id != -1:
                    task_manager.return_task(agent.current_task)
                
                agent.status = AgentStatus.TO_CHARGER
                nearest = self._find_nearest_charger_pos(agent.pos)
                agent.current_task = Task(-1, agent.pos, nearest, 1.0)
                
                if need_proactive_charge and not is_emergency:
                    is_emergency = False

            p_safe = 1.0 - self._sigmoid(omega)
            p_task = agent.current_task.importance if agent.current_task else 0.0
            
            val = (2000 + p_safe) if is_emergency else (self.gamma * p_safe + (1-self.gamma)*p_task)
            val += agent.id * 0.001 
            
            priorities.append((val, agent))
        
        priorities.sort(key=lambda x: x[0], reverse=True)
        return [a for p, a in priorities]

    def plan(self, agents: List[AgentState], current_time: int, task_manager: TaskManager):
        sorted_agents = self.update_priorities_and_goals(agents, task_manager)
        
        self.rt = ReservationTable()
        self.solver.rt = self.rt 
        
        static_reservations = {}

        # 1. Резервируем препятствия
        for a in agents:
            # Мертвые агенты - вечные стены
            if a.is_dead:
                static_path = [(a.pos[0], a.pos[1], current_time + t) for t in range(self.PLAN_HORIZON)]
                self.rt.reserve_path(static_path, a.id)
                static_reservations[a.id] = static_path
            else:
                # [ИСПРАВЛЕНИЕ] Живые агенты резервируются ТОЛЬКО на текущий момент (t=0) и следующий (t=1).
                # Это не дает другим пройти сквозь них прямо сейчас, но позволяет планировать путь "сквозь" них в будущем,
                # предполагая, что они уедут.
                short_static = [(a.pos[0], a.pos[1], current_time + t) for t in range(2)]
                self.rt.reserve_path(short_static, a.id)
                
                # Но если агент не найдет путь, мы зарезервируем его надолго (fallback)
                full_static = [(a.pos[0], a.pos[1], current_time + t) for t in range(self.PLAN_HORIZON)]
                static_reservations[a.id] = full_static

        # 2. Планируем
        for agent in sorted_agents:
            if agent.is_dead: continue

            # Снимаем короткую бронь, чтобы планировать для себя
            # (Мы не хотим врезаться в самих себя)
            current_short_static = [(agent.pos[0], agent.pos[1], current_time + t) for t in range(2)]
            self.rt.clear_path(current_short_static, agent.id)

            if agent.status == AgentStatus.CHARGING:
                needed = agent.profile.battery_capacity - agent.battery
                ticks = int(needed / agent.profile.charging_speed) + 2
                wait_path = [(agent.pos[0], agent.pos[1], current_time + t) for t in range(ticks)]
                self.rt.reserve_path(wait_path, agent.id)
                agent.assign_path(wait_path)
                continue

            if not agent.current_task or (agent.current_task.id == -1 and agent.pos == agent.current_task.goal_pos):
                self.rt.reserve_path(static_reservations[agent.id], agent.id)
                agent.assign_path(static_reservations[agent.id][1:]) 
                continue

            is_working = (agent.status == AgentStatus.WORKING)
            
            path = self.solver.find_path(
                agent.id, agent.pos, agent.current_task.goal_pos, agent.profile,
                current_time, agent.battery, self.safety_buffer, 
                energy_lambda=0.5, charger_capacity=self.charger_capacity,
                has_payload=is_working
            )

            if path:
                last_x, last_y, last_t = path[-1]
                horizon_end_time = current_time + self.PLAN_HORIZON
                wait_extension = []
                for t in range(last_t + 1, horizon_end_time):
                    wait_extension.append((last_x, last_y, t))
                
                full_path = path + wait_extension
                self.rt.reserve_path(full_path, agent.id)

                if len(full_path) > 1:
                    agent.assign_path(full_path[1:])
                else:
                    agent.assign_path([])
            else:
                # Путь не найден -> резервируем себя как стену надолго
                self.rt.reserve_path(static_reservations[agent.id], agent.id)
                agent.assign_path(static_reservations[agent.id][1:])

    def _find_nearest_charger_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        nearest = self.grid.get_nearest_charger_pos(pos)
        if nearest: return nearest
        return posы