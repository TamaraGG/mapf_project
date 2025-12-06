from typing import List, Tuple, Set, Dict
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
        
        # Кэш реальных расстояний до ближайшей зарядки
        self.charger_dist_map: Dict[Node, int] = {}
        # Кэш координат ближайшей зарядки: (x, y) -> (charger_x, charger_y)
        self.nearest_charger_map: Dict[Node, Node] = {} 

        for y in range(self.height):
            for x in range(self.width):
                val = grid_matrix[y][x]
                if val == 1: self.obstacles.add((x, y))
                elif val == 2:
                    self.chargers_list.append((x, y))
                    self.chargers_set.add((x, y))
        
        # Предварительный расчет расстояний (BFS)
        self._compute_charger_distances()

    def _compute_charger_distances(self):
        """Запускает BFS от всех зарядок одновременно, чтобы найти реальное расстояние и координаты ближайшей."""
        queue = deque()
        
        # Инициализация BFS от всех зарядок
        for c in self.chargers_list:
            queue.append((c, 0))
            self.charger_dist_map[c] = 0
            self.nearest_charger_map[c] = c # Ближайшая зарядка к самой зарядке - это она сама
        
        visited = set(self.chargers_list)
        
        while queue:
            curr_node, dist = queue.popleft()
            cx, cy = curr_node
            
            # Узнаем, какая зарядка является ближайшей для текущей клетки
            root_charger = self.nearest_charger_map[curr_node]
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        
                        # Записываем дистанцию
                        self.charger_dist_map[(nx, ny)] = dist + 1
                        # Записываем ту же зарядку, что и у родителя
                        self.nearest_charger_map[(nx, ny)] = root_charger
                        
                        queue.append(((nx, ny), dist + 1))

    def get_neighbors(self, node: Node) -> List[Node]:
        x, y = node
        res = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in self.obstacles:
                    res.append((nx, ny))
        return res

    def get_heuristic(self, a: Node, b: Node) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_nearest_charger_dist(self, node: Node) -> float:
        """Возвращает реальное расстояние до ближайшей зарядки."""
        if not self.chargers_list: return float('inf')
        return self.charger_dist_map.get(node, float('inf'))
    
    def get_nearest_charger_pos(self, node: Node) -> Node:
        """Возвращает координаты ближайшей зарядки."""
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