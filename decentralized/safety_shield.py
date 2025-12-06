import math
from typing import List, Tuple, Optional
from decentralized.agent_state import AgentState, AgentStatus

class SafetyShield:
    def __init__(self, grid_map, config):
        self.grid = grid_map
        self.safety_buffer = config.get("SAFETY_BUFFER", 20.0)
        self.pessimism_factor = 1.1 

    def check_action(self, agent: AgentState, action: int, visible_tasks: List) -> Tuple[int, bool]:
        if action == 1: return 1, False

        target_pos, will_have_payload = self._predict_outcome(agent, action, visible_tasks)
        if target_pos is None:
            target_pos = agent.pos
            will_have_payload = (agent.current_task is not None)

        dist_to_target = self.grid.get_heuristic(agent.pos, target_pos) * self.pessimism_factor
        
        cost_to_target = 0.0
        if target_pos != agent.pos:
            current_payload_status = (agent.current_task is not None and agent.status == AgentStatus.WORKING)
            cost_to_target = dist_to_target * agent.profile.calculate_move_cost(current_payload_status)
        else:
            cost_to_target = agent.profile.idle_consumption

        nearest_charger = self.grid.get_nearest_charger_pos(target_pos)
        if nearest_charger is None: return action, False

        dist_return = self.grid.get_heuristic(target_pos, nearest_charger) * self.pessimism_factor
        cost_return = dist_return * agent.profile.calculate_move_cost(will_have_payload)

        predicted_battery = agent.battery - cost_to_target
        required_reserve = cost_return + self.safety_buffer

        if predicted_battery < required_reserve:
            return 1, True
        
        return action, False

    def _predict_outcome(self, agent: AgentState, action: int, visible_tasks: List) -> Tuple[Optional[Tuple[int, int]], bool]:
        if action == 0:
            has_payload = (agent.current_task is not None and agent.status == AgentStatus.WORKING)
            return agent.pos, has_payload

        if agent.current_task and agent.status == AgentStatus.WORKING:
            return agent.current_task.goal_pos, True

        task_idx = action - 2
        if task_idx < len(visible_tasks):
            task = visible_tasks[task_idx]
            return task.start_pos, False
        
        return None, False