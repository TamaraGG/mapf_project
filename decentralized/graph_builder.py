import numpy as np
import torch
from scipy.spatial import distance
from typing import List
from common.agent_state import AgentState

class GraphBuilder:
    def __init__(self, config):
        self.comm_radius = config.get("COMM_RADIUS", 15.0)
        self.device = config.get("DEVICE", "cpu")

    def build_edge_index(self, agents: List[AgentState]) -> torch.LongTensor:
        n = len(agents)
        if n < 2:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        positions = np.array([a.pos for a in agents], dtype=np.float32)
        alive_mask = np.array([not a.is_dead for a in agents], dtype=bool)

        # Матрица расстояний
        dists = distance.cdist(positions, positions, metric='euclidean')

        # Маска связности: (dist < R) AND (dist > 0) AND (alive_i) AND (alive_j)
        adj_matrix = (dists <= self.comm_radius) & (dists > 0)
        alive_grid = alive_mask[:, None] & alive_mask[None, :]
        final_mask = adj_matrix & alive_grid

        sources, targets = np.where(final_mask)
        edge_index = np.vstack((sources, targets))
        
        return torch.from_numpy(edge_index).long().to(self.device)