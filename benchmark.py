import time
import random
import numpy as np
import torch
import csv
from copy import deepcopy
from collections import deque

# --- ИМПОРТЫ ИЗ COMMON ---
from common.grid_map import GridMap
from common.agent_profile import AGENT_PROFILES
from common.agent_state import AgentState, AgentStatus
from centralized.reservation_table import ReservationTable # Если перенесли, иначе из centralized
from common.task_manager import TaskManager, Task

# --- ИМПОРТЫ АЛГОРИТМОВ ---
# Обратите внимание: Python должен видеть эти папки как пакеты
from centralized.central_planner import CentralPlanner
from centralized.lns_optimizer import LNSOptimizer
from centralized.reservation_table import ReservationTable # Если оставили там

from decentralized.rl_environment import RLEnvironment
from decentralized.model import ActorCritic
from decentralized.safety_shield import SafetyShield
from decentralized.observation_builder import ObservationBuilder

# --- КОНФИГУРАЦИЯ ---
BENCHMARK_CONFIG = {
    "MAP_WIDTH": 20,
    "MAP_HEIGHT": 20,
    "NUM_AGENTS": 10,
    "WALL_DENSITY": 0.1,
    "MAX_STEPS": 400,
    "SHOCK_TICK": 150,
    "SEED": 42,
    # Укажите правильный путь к модели
    "MODEL_PATH": "decentralized/models/model_step_972800.pth" 
}

class Scenario:
    def __init__(self, config):
        random.seed(config["SEED"])
        np.random.seed(config["SEED"])
        
        self.width = config["MAP_WIDTH"]
        self.height = config["MAP_HEIGHT"]
        self.map_str = self._generate_map(config["WALL_DENSITY"])
        temp_grid = GridMap.from_string(self.map_str)
        
        self.agent_configs = []
        types = ["HeavyTruck", "FastDrone", "StandardBot"]
        used_pos = set()
        for i in range(config["NUM_AGENTS"]):
            p_name = types[i % 3]
            while True:
                rx, ry = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (rx, ry) not in temp_grid.obstacles and (rx, ry) not in used_pos and not temp_grid.is_charger((rx, ry)):
                    used_pos.add((rx, ry))
                    break
            bat = AGENT_PROFILES[p_name].battery_capacity
            self.agent_configs.append((i, p_name, (rx, ry), bat))
            
        self.task_schedule = []
        for i in range(300): # Генерируем с запасом
            tick = i * 2 
            t = self._generate_task(temp_grid, i, tick)
            self.task_schedule.append(t)

    def _generate_map(self, density):
        w, h = self.width, self.height
        grid = [['.' for _ in range(w)] for _ in range(h)]
        for _ in range(int(w * h * density)):
            grid[random.randint(0, h-1)][random.randint(0, w-1)] = '@'
        chargers = [(1,1), (w-2, h-2), (w//2, h//2), (w-2, 1), (1, h-2)]
        for cx, cy in chargers:
            if 0 <= cx < w and 0 <= cy < h: grid[cy][cx] = 'C'
        return "\n".join("".join(row) for row in grid)

    def _generate_task(self, grid, task_id, tick):
        while True:
            start = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            goal = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if start not in grid.obstacles and not grid.is_charger(start) and \
               goal not in grid.obstacles and not grid.is_charger(goal) and start != goal:
                return Task(task_id, start, goal, random.random(), tick)

class DeterministicTaskManager(TaskManager):
    def __init__(self, grid, task_schedule):
        super().__init__(grid)
        self.schedule = deque(task_schedule)
        self.pending_tasks = deque()
    
    def spawn_random_task(self, current_time, agents=None):
        while self.schedule and self.schedule[0].creation_time <= current_time:
            task = self.schedule.popleft()
            if task.start_pos not in self.grid.obstacles and task.goal_pos not in self.grid.obstacles:
                self.pending_tasks.append(task)

# --- CENTRALIZED RUNNER ---
def run_centralized(scenario, config):
    print(">>> Запуск Centralized CEA-MAPF...")
    grid = GridMap.from_string(scenario.map_str)
    
    agents = []
    for aid, type_name, pos, bat in scenario.agent_configs:
        a = AgentState(aid, AGENT_PROFILES[type_name], pos)
        a.battery = bat
        agents.append(a)
        
    rt = ReservationTable()
    tm = DeterministicTaskManager(grid, deepcopy(scenario.task_schedule))
    
    planner = CentralPlanner(grid, rt, gamma=0.8, safety_buffer=15.0, charger_capacity=2)
    lns = LNSOptimizer(planner)
    
    total_completed = 0
    start_time = time.time()
    
    for tick in range(config["MAX_STEPS"]):
        if tick == config["SHOCK_TICK"]:
            apply_shock_event(grid, agents)
            for a in agents: a.path = []
            
        tm.spawn_random_task(tick)
        tm.assign_tasks(agents) # Теперь этот метод есть в базовом классе!
        
        planner.plan(agents, tick, tm)
        if tick % 5 == 0: lns.step(agents, tick)
        
                # Шаг исполнения (ИСПРАВЛЕННАЯ ЛОГИКА)
        for a in agents:
            # Логика задач (Pickup -> Delivery)
            if a.status == AgentStatus.WORKING and a.current_task:
                # 1. Прибыли на точку старта (PICKUP)
                if not a.has_picked_up and a.pos == a.current_task.start_pos:
                    a.has_picked_up = True
                    a.path = [] # Сброс пути, чтобы перепланировать к финишу
                
                # 2. Прибыли на точку финиша (DELIVERY)
                elif a.has_picked_up and a.pos == a.current_task.goal_pos:
                    if a.current_task.id != -1: # Не считаем доезд до зарядки
                        total_completed += 1
                    a.has_picked_up = False
                    a.current_task = None
                    a.status = AgentStatus.IDLE
                    a.path = []
            
            # Физический шаг
            a.step(grid)
            
    duration = time.time() - start_time
    dead_agents = sum(1 for a in agents if a.is_dead)
    total_energy = sum(a.total_energy_consumed for a in agents)
    
    return {"Throughput": total_completed, "Dead": dead_agents, "Energy": total_energy, "Time": duration}

# --- DECENTRALIZED RUNNER ---
def run_decentralized(scenario, config):
    print(">>> Запуск Decentralized DEG-RL...")
    
    rl_config = {
        "MAP_WIDTH": config["MAP_WIDTH"], "MAP_HEIGHT": config["MAP_HEIGHT"],
        "NUM_AGENTS": config["NUM_AGENTS"], "WALL_DENSITY": config["WALL_DENSITY"],
        "COMM_RADIUS": 10.0, "K_VISIBLE_TASKS": 5, "DEVICE": "cpu"
    }
    
    env = RLEnvironment(rl_config)
    env.grid = GridMap.from_string(scenario.map_str)
    env.task_manager = DeterministicTaskManager(env.grid, deepcopy(scenario.task_schedule))
    env.obs_builder = ObservationBuilder(env.grid, env.task_manager, rl_config)
    
    env.agents = []
    for aid, type_name, pos, bat in scenario.agent_configs:
        a = AgentState(aid, AGENT_PROFILES[type_name], pos)
        a.battery = bat
        a.target_pos = None
        a.prev_target_pos = None
        env.agents.append(a)
        
    obs_dim = env.obs_builder.encode(env.agents[0], env.agents).shape[0]
    action_dim = env.action_dim
    model = ActorCritic(obs_dim, action_dim)
    
    try:
        model.load_state_dict(torch.load(config["MODEL_PATH"], map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        print(f"[WARNING] Модель {config['MODEL_PATH']} не найдена. Используются случайные веса.")

    shield = SafetyShield(env.grid, {"SAFETY_BUFFER": 15.0})
    
    obs, edge_index, mask = env._get_full_state()
    obs_t = torch.tensor(obs)
    mask_t = torch.tensor(mask)
    
    total_completed = 0
    start_time = time.time()
    
    for tick in range(config["MAX_STEPS"]):
        if tick == config["SHOCK_TICK"]:
            apply_shock_event(env.grid, env.agents)
            env.grid._compute_charger_distances()

        env.task_manager.spawn_random_task(tick)
        
        with torch.no_grad():
            action, _, _, _ = model.get_action(obs_t, edge_index, mask_t, deterministic=True)
        
        cpu_actions = action.numpy().tolist()
        safe_actions = []
        for i, agent in enumerate(env.agents):
            if agent.is_dead:
                safe_actions.append(0)
                continue
            visible = env.obs_builder.find_k_nearest_tasks(agent)
            final_act, _ = shield.check_action(agent, cpu_actions[i], visible)
            safe_actions.append(final_act)
            
        next_obs, rewards, dones, infos = env.step(safe_actions)
        total_completed += sum(1 for r in rewards if r > 10.0)
        
        obs_t = torch.tensor(next_obs[0])
        edge_index = next_obs[1]
        mask_t = torch.tensor(next_obs[2])
        
    duration = time.time() - start_time
    dead_agents = sum(1 for a in env.agents if a.is_dead)
    total_energy = sum(a.total_energy_consumed for a in env.agents)
    
    return {"Throughput": total_completed, "Dead": dead_agents, "Energy": total_energy, "Time": duration}

def apply_shock_event(grid, agents):
    center_x, center_y = grid.width // 2, grid.height // 2
    radius = 3
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if 0 <= x < grid.width and 0 <= y < grid.height:
                if not grid.is_charger((x, y)):
                    grid.add_dynamic_obstacle(x, y)
    for a in agents:
        if a.pos in grid.obstacles:
            a.status = AgentStatus.DEAD
            a.battery = 0

if __name__ == "__main__":
    print("=== ГЕНЕРАЦИЯ СЦЕНАРИЯ ===")
    scenario = Scenario(BENCHMARK_CONFIG)
    print(f"Сценарий создан. Задач в пуле: {len(scenario.task_schedule)}")
    
    res_central = run_centralized(scenario, BENCHMARK_CONFIG)
    res_decentral = run_decentralized(scenario, BENCHMARK_CONFIG)
    
    print("\n" + "="*60)
    print(f"{'METRIC':<20} | {'CENTRALIZED':<15} | {'DECENTRALIZED':<15}")
    print("-" * 60)
    print(f"{'Throughput':<20} | {res_central['Throughput']:<15} | {res_decentral['Throughput']:<15}")
    print(f"{'Dead Agents':<20} | {res_central['Dead']:<15} | {res_decentral['Dead']:<15}")
    print(f"{'Total Energy':<20} | {int(res_central['Energy']):<15} | {int(res_decentral['Energy']):<15}")
    
    e_eff_c = res_central['Energy'] / res_central['Throughput'] if res_central['Throughput'] else 0
    e_eff_d = res_decentral['Energy'] / res_decentral['Throughput'] if res_decentral['Throughput'] else 0
    
    print(f"{'Energy/Task':<20} | {e_eff_c:.2f}{'':<11} | {e_eff_d:.2f}")
    print(f"{'Sim Time (s)':<20} | {res_central['Time']:.2f}{'':<11} | {res_decentral['Time']:.2f}")
    print("="*60)