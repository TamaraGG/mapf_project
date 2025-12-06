import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)
        
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            neighbor_mean = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            neighbor_sum = torch.zeros_like(x)
            neighbor_sum.index_add_(0, dst, x[src])
            
            ones = torch.ones(src.size(0), 1, device=x.device)
            degree = torch.zeros(num_nodes, 1, device=x.device)
            degree.index_add_(0, dst, ones)
            
            neighbor_mean = neighbor_sum / (degree + 1e-6)

        combined = torch.cat([x, neighbor_mean], dim=1)
        out = F.relu(self.linear(combined))
        return out

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.gnn = GraphSAGELayer(hidden_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, obs, edge_index):
        x = self.encoder(obs)
        h = self.gnn(x, edge_index)
        return h

    def get_action(self, obs, edge_index, action_masks=None, deterministic=False):
        h = self.forward(obs, edge_index)
        value = self.critic(h)
        logits = self.actor(h)
        
        if action_masks is not None:
            HUGE_NEG = -1e8
            logits = torch.where(action_masks > 0.5, logits, torch.tensor(HUGE_NEG).to(logits.device))

        probs = Categorical(logits=logits)
        action = torch.argmax(logits, dim=1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, obs, edge_index):
        h = self.forward(obs, edge_index)
        return self.critic(h)