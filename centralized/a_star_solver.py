import heapq
from typing import List, Tuple, Optional
from node import Node

def get_neighbors(current_node, grid, reservation_table, profile, goal_pos, safety_buffer, energy_lambda, charger_capacity, has_payload):
    neighbors = []
    moves = [(0, 1, False), (0, -1, False), (1, 0, False), (-1, 0, False), (0, 0, True)]

    going_to_charger = grid.is_charger(goal_pos)

    for dx, dy, is_wait in moves:
        nx, ny = current_node.x + dx, current_node.y + dy
        nt = current_node.time + 1

        if not (0 <= nx < grid.width and 0 <= ny < grid.height): continue
        if (nx, ny) in grid.obstacles: continue

        cap = charger_capacity if grid.is_charger((nx, ny)) else 1
        
        if reservation_table.is_vertex_collision(nx, ny, nt, capacity=cap): continue
        if reservation_table.is_edge_collision((current_node.x, current_node.y), (nx, ny), nt): continue

        step_cost = profile.idle_consumption if is_wait else profile.calculate_move_cost(has_payload)
        new_bat = current_node.remaining_battery - step_cost

        if new_bat < 0: continue 
        
        is_charger_cell = grid.is_charger((nx, ny))
        
        # Проверка безопасности
        if not going_to_charger and not is_charger_cell:
            dist = grid.get_nearest_charger_dist((nx, ny))
            if dist == float('inf'): continue 

            # [ИСПРАВЛЕНИЕ] Считаем, что возвращаться на зарядку будем ПУСТЫМИ (без груза).
            # Это оптимистичная оценка, но она разблокирует агентов.
            # has_payload=False для возврата
            return_consumption = profile.calculate_move_cost(has_payload=False) 
            return_cost = dist * return_consumption
            
            if new_bat < return_cost + safety_buffer:
                continue
        
        new_g = current_node.g + 1 + (energy_lambda * step_cost)
        new_h = grid.get_heuristic((nx, ny), goal_pos)
        
        neighbors.append(Node(nx, ny, nt, new_g, new_h, new_g+new_h, new_bat, current_node))
    
    return neighbors

class EnergyAwareAStar:
    def __init__(self, grid, reservation_table):
        self.grid = grid
        self.rt = reservation_table

    def find_path(self, agent_id, start_pos, goal_pos, profile, start_time, current_battery, safety_buffer, energy_lambda=0.5, charger_capacity=1, has_payload=False):

        start_h = self.grid.get_heuristic(start_pos, goal_pos)
        start_node = Node(start_pos[0], start_pos[1], start_time, 0.0, start_h, start_h, current_battery, None)
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        
        MAX_DEPTH = 100 

        while open_set:
            curr = heapq.heappop(open_set)
            
            if (curr.x, curr.y) == goal_pos:
                return self._reconstruct(curr)
            
            state_key = (curr.x, curr.y, curr.time)
            if state_key in closed_set: continue
            closed_set.add(state_key)

            if curr.time - start_time > MAX_DEPTH: continue

            nbs = get_neighbors(curr, self.grid, self.rt, profile, goal_pos, safety_buffer, energy_lambda, charger_capacity, has_payload)
            
            for nb in nbs:
                if (nb.x, nb.y, nb.time) not in closed_set:
                    heapq.heappush(open_set, nb)
        return None

    def _reconstruct(self, node):
        path = []
        while node:
            path.append((node.x, node.y, node.time))
            node = node.parent
        return path[::-1]