import time
import random
import numpy as np
import torch
import csv
from copy import deepcopy

# Импорты общих модулей
from common.grid_map import GridMap
from common.agent_profile import AGENT_PROFILES
from decentralized.agent_state import AgentState, AgentStatus
from centralized.reservation_table import ReservationTable
from decentralized.task_manager import TaskManager, Task

# Импорты алгоритмов
from centralized.central_planner import CentralPlanner
from centralized.lns_optimizer import LNSOptimizer
from decentralized.rl_environment import RLEnvironment
from decentralized.model import ActorCritic
from decentralized.safety_shield import SafetyShield

# --- КОНФИГУРАЦИЯ БЕНЧМАРКА ---
BENCHMARK_CONFIG = {
    "MAP_WIDTH": 20,
    "MAP_HEIGHT": 20,
    "NUM_AGENTS": 10,
    "WALL_DENSITY": 0.1,
    "MAX_STEPS": 400,
    "SHOCK_TICK": 150,
    "SEED": 42,  # <--- ГЛАВНОЕ: Фиксированный сид
    "MODEL_PATH": "decentralized\models\model_step_972800.pth" # Укажите ваш путь
}

class Scenario:
    """Хранит замороженное состояние мира для воспроизводимости."""
    def __init__(self, config):
        random.seed(config["SEED"])
        np.random.seed(config["SEED"])
        
        self.width = config["MAP_WIDTH"]
        self.height = config["MAP_HEIGHT"]
        
        # 1. Генерация карты
        self.map_str = self._generate_map(config["WALL_DENSITY"])
        temp_grid = GridMap.from_string(self.map_str)
        
        # 2. Генерация стартовых позиций агентов
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
            # Сохраняем: (id, type_name, start_pos, start_battery)
            bat = AGENT_PROFILES[p_name].battery_capacity # Все полные на старте для честности
            self.agent_configs.append((i, p_name, (rx, ry), bat))
            
        # 3. Генерация потока задач (заранее на весь эпизод)
        self.task_schedule = []
        # Генерируем 200 задач заранее
        for i in range(200):
            tick = i * 2 # Каждые 2 тика новая задача (примерно)
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
    """Подменяет случайную генерацию на выдачу задач из списка."""
    def __init__(self, grid, task_schedule):
        super().__init__(grid)
        self.schedule = deque(task_schedule) # Копия списка
        self.pending_tasks = deque()
    
    def spawn_random_task(self, current_time, agents=None):
        # Переопределяем метод: вместо рандома берем из расписания
        while self.schedule and self.schedule[0].creation_time <= current_time:
            task = self.schedule.popleft()
            # Проверяем валидность (вдруг там теперь стена после шока)
            if task.start_pos not in self.grid.obstacles and task.goal_pos not in self.grid.obstacles:
                self.pending_tasks.append(task)
            else:
                # Если задача попала в завал - пропускаем
                pass

# --- ЗАПУСК ЦЕНТРАЛИЗОВАННОГО ---
def run_centralized(scenario, config):
    print(">>> Запуск Centralized CEA-MAPF...")
    grid = GridMap.from_string(scenario.map_str)
    
    agents = []
    for aid, type_name, pos, bat in scenario.agent_configs:
        a = AgentState(aid, AGENT_PROFILES[type_name], pos)
        a.battery = bat
        agents.append(a)
        
    rt = ReservationTable()
    # Используем детерминированный менеджер
    tm = DeterministicTaskManager(grid, deepcopy(scenario.task_schedule))
    
    planner = CentralPlanner(grid, rt, gamma=0.8, safety_buffer=15.0, charger_capacity=2)
    lns = LNSOptimizer(planner)
    
    total_completed = 0
    dead_agents = 0
    start_time = time.time()
    
    for tick in range(config["MAX_STEPS"]):
        # Шок
        if tick == config["SHOCK_TICK"]:
            apply_shock_event(grid, agents) # Функция из main.py (нужно скопировать или импортировать)
            for a in agents: a.path = []
            
        # Задачи
        tm.spawn_random_task(tick) # Теперь это берет из списка
        tm.assign_tasks(agents)
        
        # План
        planner.plan(agents, tick, tm)
        if tick % 5 == 0: lns.step(agents, tick)
        
        # Шаг
        for a in agents:
            if a.current_task and a.pos == a.current_task.goal_pos:
                if a.current_task.id != -1: total_completed += 1
                a.current_task = None
                a.status = AgentStatus.IDLE
                a.path = []
            a.step(grid)
            
    duration = time.time() - start_time
    dead_agents = sum(1 for a in agents if a.is_dead)
    total_energy = sum(a.total_energy_consumed for a in agents)
    
    return {
        "Method": "Centralized",
        "Throughput": total_completed,
        "Dead": dead_agents,
        "Energy": total_energy,
        "Time": duration
    }

# --- ЗАПУСК ДЕЦЕНТРАЛИЗОВАННОГО ---
def run_decentralized(scenario, config):
    print(">>> Запуск Decentralized DEG-RL...")
    
    # Настройка среды вручную, чтобы внедрить сценарий
    rl_config = {
        "MAP_WIDTH": config["MAP_WIDTH"], "MAP_HEIGHT": config["MAP_HEIGHT"],
        "NUM_AGENTS": config["NUM_AGENTS"], "WALL_DENSITY": config["WALL_DENSITY"],
        "COMM_RADIUS": 10.0, "K_VISIBLE_TASKS": 5,
        "DEVICE": "cpu"
    }
    
    env = RLEnvironment(rl_config)
    # Взлом reset: подменяем генерацию на наш сценарий
    env.grid = GridMap.from_string(scenario.map_str)
    env.task_manager = DeterministicTaskManager(env.grid, deepcopy(scenario.task_schedule))
    from observation_builder import ObservationBuilder
    env.obs_builder = ObservationBuilder(env.grid, env.task_manager, rl_config)
    
    env.agents = []
    for aid, type_name, pos, bat in scenario.agent_configs:
        a = AgentState(aid, AGENT_PROFILES[type_name], pos)
        a.battery = bat
        a.target_pos = None
        a.prev_target_pos = None
        env.agents.append(a)
        
    # Загрузка модели
    obs_dim = env.obs_builder.encode(env.agents[0], env.agents).shape[0]
    action_dim = env.action_dim
    model = ActorCritic(obs_dim, action_dim)
    try:
        model.load_state_dict(torch.load(config["MODEL_PATH"], map_location='cpu'))
        model.eval()
    except:
        print("Модель не найдена! Результаты будут случайными.")

    shield = SafetyShield(env.grid, {"SAFETY_BUFFER": 15.0})
    
    # Получаем начальное наблюдение
    obs, edge_index, mask = env._get_full_state()
    obs_t = torch.tensor(obs)
    mask_t = torch.tensor(mask)
    
    total_completed = 0
    start_time = time.time()
    
    for tick in range(config["MAX_STEPS"]):
        if tick == config["SHOCK_TICK"]:
            # Копируем логику шока
            center_x, center_y = env.width // 2, env.height // 2
            radius = 3
            for y in range(center_y - radius, center_y + radius + 1):
                for x in range(center_x - radius, center_x + radius + 1):
                    if not env.grid.is_charger((x, y)):
                        env.grid.add_dynamic_obstacle(x, y)
            # Убиваем попавших
            for a in env.agents:
                if a.pos in env.grid.obstacles: a.status = AgentStatus.DEAD
            env.grid._compute_charger_distances()

        # RL Step
        env.task_manager.spawn_random_task(tick) # Из списка
        
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
        
        # Считаем выполненные (бонус > 10)
        total_completed += sum(1 for r in rewards if r > 10.0)
        
        obs_t = torch.tensor(next_obs[0])
        edge_index = next_obs[1]
        mask_t = torch.tensor(next_obs[2])
        
    duration = time.time() - start_time
    dead_agents = sum(1 for a in env.agents if a.is_dead)
    total_energy = sum(a.total_energy_consumed for a in env.agents)
    
    return {
        "Method": "Decentralized (RL)",
        "Throughput": total_completed,
        "Dead": dead_agents,
        "Energy": total_energy,
        "Time": duration
    }

# Вспомогательная функция для шока (копия из main)
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
    from collections import deque # Нужно для DeterministicTaskManager
    
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