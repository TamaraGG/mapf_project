from typing import List, Tuple, Set, Dict, Optional
from collections import deque

Node = Tuple[int, int]

class GridMap:
    def __init__(self, grid_matrix: List[List[int]]):
        if not grid_matrix: raise ValueError("Empty grid")
        self.height = len(grid_matrix)
        self.width = len(grid_matrix[0])
        self.obstacles: Set[Node] = set()
        self.chargers_list: List[Node] = []
        self.chargers_set: Set[Node] = set()
        
        # Кэши
        self.charger_dist_map: Dict[Node, int] = {}
        self.nearest_charger_map: Dict[Node, Node] = {} 

        for y in range(self.height):
            for x in range(self.width):
                val = grid_matrix[y][x]
                if val == 1: self.obstacles.add((x, y))
                elif val == 2:
                    self.chargers_list.append((x, y))
                    self.chargers_set.add((x, y))
        
        self._compute_charger_distances()

    def _compute_charger_distances(self):
        """BFS от всех зарядок одновременно."""
        queue = deque()
        for c in self.chargers_list:
            queue.append((c, 0))
            self.charger_dist_map[c] = 0
            self.nearest_charger_map[c] = c
        
        visited = set(self.chargers_list)
        
        while queue:
            curr_node, dist = queue.popleft()
            root_charger = self.nearest_charger_map[curr_node]
            
            cx, cy = curr_node
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        self.charger_dist_map[(nx, ny)] = dist + 1
                        self.nearest_charger_map[(nx, ny)] = root_charger
                        queue.append(((nx, ny), dist + 1))

    def get_heuristic(self, a: Node, b: Node) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_nearest_charger_pos(self, node: Node) -> Optional[Node]:
        if not self.chargers_list: return None
        return self.nearest_charger_map.get(node, None)

    def is_charger(self, node: Node) -> bool:
        return node in self.chargers_set

    @staticmethod
    def from_string(map_str: str) -> 'GridMap':
        rows = map_str.strip().split('\n')
        matrix = []
        for row in rows:
            r = []
            for char in row.strip():
                if char == '@': r.append(1)
                elif char == 'C': r.append(2)
                else: r.append(0)
            matrix.append(r)
        return GridMap(matrix)