import numpy as np
from typing import List
from agent_state import AgentState, AgentStatus

class ObservationBuilder:
    def __init__(self, grid_map, task_manager, config):
        self.grid = grid_map
        self.tm = task_manager
        self.width = config["MAP_WIDTH"]
        self.height = config["MAP_HEIGHT"]
        self.max_dim = self.width + self.height
        self.k_tasks = config.get("K_VISIBLE_TASKS", 5)
        
        self.MAX_BATTERY = 2000.0
        self.MAX_RHO = 10.0
        self.MAX_SPEED = 2.0

    def encode(self, agent: AgentState, all_agents: List[AgentState]) -> np.ndarray:
        # 1. Self
        x_norm = agent.pos[0] / self.width
        y_norm = agent.pos[1] / self.height
        b_norm = np.clip(agent.battery / agent.profile.battery_capacity, 0.0, 1.0)
        
        # [ИСПРАВЛЕНИЕ] Груз есть, если есть задача, независимо от статуса (едем мы или заряжаемся)
        has_payload = 1.0 if agent.current_task is not None else 0.0

        # 2. Profile
        rho_norm = agent.profile.base_consumption / self.MAX_RHO
        cap_norm = agent.profile.battery_capacity / self.MAX_BATTERY
        speed_norm = agent.profile.speed / self.MAX_SPEED

        # 3. Charger
        chg_pos = self.grid.get_nearest_charger_pos(agent.pos)
        if chg_pos:
            dist_c = self.grid.get_heuristic(agent.pos, chg_pos) / self.max_dim
            dx_c = (chg_pos[0] - agent.pos[0]) / self.width
            dy_c = (chg_pos[1] - agent.pos[1]) / self.height
            
            queue_count = 0
            for a in all_agents:
                if not a.is_dead and a.id != agent.id:
                    if a.pos == chg_pos: queue_count += 1
                    elif a.status == AgentStatus.TO_CHARGER:
                        if self.grid.get_nearest_charger_pos(a.pos) == chg_pos:
                            queue_count += 1
            q_norm = np.clip(queue_count / 5.0, 0.0, 1.0)
        else:
            dist_c, dx_c, dy_c, q_norm = 1.0, 0.0, 0.0, 1.0

        # 4. Tasks / Goal
        current_goal_dist = 0.0
        current_goal_dx = 0.0
        current_goal_dy = 0.0
        task_features = []

        if has_payload > 0.5:
            goal = agent.current_task.goal_pos
            current_goal_dist = self.grid.get_heuristic(agent.pos, goal) / self.max_dim
            current_goal_dx = (goal[0] - agent.pos[0]) / self.width
            current_goal_dy = (goal[1] - agent.pos[1]) / self.height
            task_features = [1.0, 0.0] * self.k_tasks 
        else:
            visible_tasks = self.find_k_nearest_tasks(agent)
            for t in visible_tasks:
                d_t = self.grid.get_heuristic(agent.pos, t.start_pos) / self.max_dim
                imp_t = t.importance
                task_features.extend([d_t, imp_t])
            
            while len(task_features) < self.k_tasks * 2:
                task_features.extend([1.0, 0.0])

        obs = np.array([
            x_norm, y_norm, b_norm, has_payload,      
            rho_norm, cap_norm, speed_norm,           
            dist_c, dx_c, dy_c, q_norm,               
            current_goal_dist, current_goal_dx, current_goal_dy 
        ] + task_features, dtype=np.float32)

        return obs

    def find_k_nearest_tasks(self, agent):
        if not self.tm.pending_tasks: return []
        tasks = list(self.tm.pending_tasks)
        tasks.sort(key=lambda t: self.grid.get_heuristic(agent.pos, t.start_pos))
        return tasks[:self.k_tasks]