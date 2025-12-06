import time
import torch
import numpy as np
import csv
import statistics
import random
from typing import List
import pygame # Нужен для визуализации

from rl_environment import RLEnvironment
from model import ActorCritic
from safety_shield import SafetyShield
from common.agent_state import AgentStatus
from common.visualizer import PygameVisualizer # Импортируем визуализатор

# --- КОНФИГУРАЦИЯ СЦЕНАРИЕВ ---
SCENARIO_CONFIG = {
    "BASELINE": {
        "SHOCK_EVENT": False,
        "AGENTS": 8,          # 8 агентов, как при обучении
        "MAP_SIZE": 20,
        "FILENAME": "results_rl_baseline.csv"
    },
    "ADAPTABILITY_TEST": {
        "SHOCK_EVENT": True,
        "SHOCK_TICK": 150,
        "AGENTS": 8,
        "MAP_SIZE": 20,
        "FILENAME": "results_rl_shock.csv"
    }
}

# !!! ВЫБЕРИТЕ СЦЕНАРИЙ ЗДЕСЬ !!!
# CURRENT_SCENARIO = 'BASELINE'
CURRENT_SCENARIO = 'ADAPTABILITY_TEST' 

# Путь к модели
MODEL_PATH = "decentralized\models\model_step_972800.pth" 

# Конфигурация среды
RL_CONFIG = {
    "MAP_WIDTH": SCENARIO_CONFIG[CURRENT_SCENARIO]["MAP_SIZE"],
    "MAP_HEIGHT": SCENARIO_CONFIG[CURRENT_SCENARIO]["MAP_SIZE"],
    "NUM_AGENTS": SCENARIO_CONFIG[CURRENT_SCENARIO]["AGENTS"],
    "WALL_DENSITY": 0.1,
    "COMM_RADIUS": 10.0,
    "K_VISIBLE_TASKS": 5,
    
    # --- ИСПРАВЛЕНИЕ: ВЕРНУЛИ БОНУС ---
    "W1_PROGRESS": 0, "W2_ENERGY": 0, "W3_FEAR": 0, 
    "REWARD_BONUS": 50.0, # Чтобы детектировать выполнение задачи
    "BATTERY_THRESHOLD": 0.3, "SHIELD_PENALTY": 0,
    "DEVICE": "cpu"
}

MAX_STEPS = 400
ENABLE_VIZ = True # Включить картинку? (True/False)

class MetricsLogger:
    def __init__(self, filename):
        self.filename = filename
        self.history = []
        self.total_completed = 0
        self.inference_latencies = []

    def log_step(self, tick, agents, completed_now, latency):
        self.total_completed += completed_now
        self.inference_latencies.append(latency)
        total_energy = sum(a.total_energy_consumed for a in agents)
        dead_count = sum(1 for a in agents if a.is_dead)
        
        self.history.append({
            'tick': tick,
            'completed_cumulative': self.total_completed,
            'completed_now': completed_now,
            'total_energy': total_energy,
            'dead_agents': dead_count,
            'latency_ms': latency * 1000
        })

    def save_to_csv(self):
        if not self.history: return
        keys = self.history[0].keys()
        try:
            with open(self.filename, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.history)
            print(f"\n[INFO] Логи RL сохранены: {self.filename}")
        except Exception as e:
            print(f"[ERROR] Ошибка записи CSV: {e}")

    def print_final_report(self, num_agents, total_ticks):
        if self.total_completed == 0:
            eta_sys = 0
        else:
            final_energy = self.history[-1]['total_energy']
            eta_sys = final_energy / self.total_completed

        throughput = (self.total_completed / total_ticks) * 1000
        final_dead = self.history[-1]['dead_agents']
        sr = ((num_agents - final_dead) / num_agents) * 100.0
        
        avg_lat = statistics.mean(self.inference_latencies) * 1000
        max_lat = max(self.inference_latencies) * 1000

        print("\n" + "="*50)
        print(f"ИТОГОВЫЙ ОТЧЕТ DEG-RL ({CURRENT_SCENARIO})")
        print("="*50)
        print(f"1. Энергоэффективность (η_sys): {eta_sys:.2f} ед./задачу")
        print(f"2. Пропускная способность:      {throughput:.2f} задач/1000 тиков")
        print(f"3. Выживаемость (SR):           {sr:.1f}% (Умерло: {final_dead})")
        print(f"4. Задержка (Inference):        Ср: {avg_lat:.2f} мс | Макс: {max_lat:.2f} мс")
        print("="*50 + "\n")

def apply_shock_event(env):
    print(f"\n[{time.strftime('%H:%M:%S')}] !!! SHOCK EVENT: ОБВАЛ В ЦЕНТРЕ !!!")
    center_x, center_y = env.width // 2, env.height // 2
    radius = 2
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if not env.grid.is_charger((x, y)):
                agent_here = any(a.pos == (x, y) for a in env.agents)
                if not agent_here:
                    env.grid.obstacles.add((x, y))
    env.grid._compute_charger_distances()

def main():
    env = RLEnvironment(RL_CONFIG)
    obs, edge_index, mask = env.reset()
    
    obs_dim = env.obs_builder.encode(env.agents[0], env.agents).shape[0]
    action_dim = env.action_dim
    model = ActorCritic(obs_dim, action_dim)
    
    try:
        print(f"Загрузка модели из {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        print("Файл модели не найден! Проверьте путь.")
        return

    shield = SafetyShield(env.grid, RL_CONFIG)
    logger = MetricsLogger(SCENARIO_CONFIG[CURRENT_SCENARIO]["FILENAME"])
    
    # Визуализатор
    viz = None
    if ENABLE_VIZ:
        viz = PygameVisualizer(env.grid)

    obs_t = torch.tensor(obs)
    mask_t = torch.tensor(mask)
    
    print(f"Запуск симуляции RL... Шагов: {MAX_STEPS}")

    for tick in range(MAX_STEPS):
        if SCENARIO_CONFIG[CURRENT_SCENARIO]["SHOCK_EVENT"] and tick == SCENARIO_CONFIG[CURRENT_SCENARIO]["SHOCK_TICK"]:
            apply_shock_event(env)

        t_start = time.perf_counter()
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
            
            # Хак для визуализации намерений (куда едет агент)
            if ENABLE_VIZ:
                if final_act == 1: agent.target_pos = env.grid.get_nearest_charger_pos(agent.pos)
                elif final_act >= 2:
                    t_idx = final_act - 2
                    if t_idx < len(visible): agent.target_pos = visible[t_idx].start_pos
                    elif agent.current_task: agent.target_pos = agent.current_task.goal_pos

        t_end = time.perf_counter()
        latency = t_end - t_start

        next_obs, rewards, dones, infos = env.step(safe_actions)
        
        # Теперь rewards содержат 50.0 при успехе, проверка сработает
        completed_this_tick = sum(1 for r in rewards if r > 10.0)

        logger.log_step(tick, env.agents, completed_this_tick, latency)
        
        if ENABLE_VIZ:
            stats = {'completed': logger.total_completed, 'dead': sum(1 for a in env.agents if a.is_dead)}
            viz.render(env.agents, tick, stats, env.task_manager)
            time.sleep(0.05) # Раскомментируйте, если слишком быстро

        obs_t = torch.tensor(next_obs[0])
        edge_index = next_obs[1]
        mask_t = torch.tensor(next_obs[2])
        
        if all(dones): break

    logger.save_to_csv()
    logger.print_final_report(RL_CONFIG["NUM_AGENTS"], MAX_STEPS)

if __name__ == "__main__":
    main()