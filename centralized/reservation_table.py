from collections import defaultdict
from typing import List, Tuple, Set, Dict

class ReservationTable:
    def __init__(self):
        # time -> (x,y) -> {agent_ids}
        self._reserved: Dict[int, Dict[Tuple[int, int], Set[int]]] = defaultdict(lambda: defaultdict(set))
        # time -> ((u_x, u_y), (v_x, v_y)) -> {agent_ids}
        self._reserved_edges: Dict[int, Dict[Tuple[Tuple[int, int], Tuple[int, int]], Set[int]]] = defaultdict(lambda: defaultdict(set))

    def reserve_path(self, path: List[Tuple[int, int, int]], agent_id: int):
        for i in range(len(path)):
            x, y, t = path[i]
            self._reserved[t][(x, y)].add(agent_id)
            
            if i > 0:
                prev_x, prev_y, prev_t = path[i-1]
                edge = ((prev_x, prev_y), (x, y))
                self._reserved_edges[t][edge].add(agent_id)

    def clear_path(self, path: List[Tuple[int, int, int]], agent_id: int):
        for i in range(len(path)):
            x, y, t = path[i]
            if t in self._reserved and (x, y) in self._reserved[t]:
                self._reserved[t][(x, y)].discard(agent_id)
                if not self._reserved[t][(x, y)]: del self._reserved[t][(x, y)]
            
            if i > 0:
                prev_x, prev_y, prev_t = path[i-1]
                edge = ((prev_x, prev_y), (x, y))
                if t in self._reserved_edges and edge in self._reserved_edges[t]:
                    self._reserved_edges[t][edge].discard(agent_id)

    def is_vertex_collision(self, x: int, y: int, t: int, capacity: int = 1) -> bool:
        return len(self._reserved[t].get((x, y), set())) >= capacity

    def is_edge_collision(self, prev_node: Tuple[int, int], curr_node: Tuple[int, int], t: int) -> bool:
        reverse_edge = (curr_node, prev_node)
        return len(self._reserved_edges[t].get(reverse_edge, set())) > 0