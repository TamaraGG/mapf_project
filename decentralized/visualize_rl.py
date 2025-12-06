import torch
import time
import numpy as np
import pygame
import sys

# Импорты ваших модулей
from rl_environment import RLEnvironment
from model import ActorCritic
from safety_shield import SafetyShield
from visualizer import PygameVisualizer 

# --- КОНФИГУРАЦИЯ (Должна совпадать с обучением!) ---
CONFIG = {
    "MAP_WIDTH": 10,          
    "MAP_HEIGHT": 10,         
    "NUM_AGENTS": 4,          
    "WALL_DENSITY": 0.1,      
    "COMM_RADIUS": 10.0,
    "K_VISIBLE_TASKS": 5,
    
    # Сброшены до 0, т.к. не влияют на детерминированную визуализацию
    "W1_PROGRESS": 0.0, "W2_ENERGY": 0.0, "W3_FEAR": 0.0, 
    "REWARD_BONUS": 50.0, "BATTERY_THRESHOLD": 0.3, "SHIELD_PENALTY": 0.0,
    "DEVICE": "cpu" 
}

# ПУТЬ К ВАШЕМУ "ЗОЛОТОМУ" ЧЕКПОИНТУ
MODEL_PATH = "decentralized\models\model_step_972800.pth" 

def main():
    # 1. Инициализация среды и модели
    env = RLEnvironment(CONFIG)
    obs, edge_index, mask = env.reset()
    
    dummy_obs = env.obs_builder.encode(env.agents[0], env.agents)
    obs_dim = dummy_obs.shape[0]
    action_dim = env.action_dim
    
    model = ActorCritic(obs_dim, action_dim)
    
    # Загрузка весов
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() 
        print("Model loaded successfully! Starting visualization...")
    except FileNotFoundError:
        print(f"Error: File {MODEL_PATH} not found. Check the path.")
        return

    # 4. Инициализация Shield и Визуализатора
    shield = SafetyShield(env.grid, CONFIG)
    viz = PygameVisualizer(env.grid)
    
    # Переменные для отрисовки статистики
    stats = {'completed': 0, 'dead': 0}
    
    # Тензоры на CPU
    obs_t = torch.tensor(obs)
    mask_t = torch.tensor(mask)

    # --- ГЛАВНЫЙ ЦИКЛ ---
    for tick in range(10000):
        
        # 1. Получаем действие от нейросети
        with torch.no_grad():
            action, _, _, _ = model.get_action(obs_t, edge_index, mask_t, deterministic=True)
        
        cpu_actions = action.numpy().tolist()
        
        # 2. Пропускаем через Safety Shield (ВАЖНО!)
        safe_actions = []
        for i, agent in enumerate(env.agents):
            if agent.is_dead:
                safe_actions.append(0)
                continue
            
            # Shield должен получить правильный список задач для проверки
            visible = env.obs_builder.find_k_nearest_tasks(agent)
            final_act, corrected = shield.check_action(agent, cpu_actions[i], visible)
            safe_actions.append(final_act)
            
            # --- СИНХРОНИЗАЦИЯ: ВРЕМЕННО СТАВИМ TARGET ДЛЯ ВИЗУАЛИЗАЦИИ ---
            # Это нужно, чтобы _move_greedy знал, куда идти, и чтобы визуализатор знал, что рисовать
            if final_act == 1:
                agent.target_pos = env.grid.get_nearest_charger_pos(agent.pos)
            elif final_act >= 2 and agent.current_task:
                agent.target_pos = agent.current_task.goal_pos
            elif final_act >= 2: # Выбрал задачу, но еще не взял
                 visible = env.obs_builder.find_k_nearest_tasks(agent)
                 task_idx = final_act - 2
                 if task_idx < len(visible):
                    agent.target_pos = visible[task_idx].start_pos
            else:
                agent.target_pos = agent.pos


        # 3. Шаг среды (Используем safe_actions)
        # Env.step выполняет всю сложную логику движения, взятия груза, зарядки
        next_obs, rewards, dones, infos = env.step(safe_actions)

        # 4. СБОР СТАТИСТИКИ (Вернули корректный сбор)
        completed_now = sum(1 for r in rewards if r > 10.0) # Если награда > 10, значит была доставка (бонус 20)
        stats['completed'] += completed_now
        stats['dead'] = sum(1 for a in env.agents if a.is_dead)

        # 5. ОТРИСОВКА
        viz.render(env.agents, tick, stats, env.task_manager)
        
        time.sleep(0.1)
        
        # 6. Подготовка к следующему шагу
        obs_t = torch.tensor(next_obs[0])
        edge_index = next_obs[1]
        mask_t = torch.tensor(next_obs[2])
        
        if all(dones):
            print("All agents dead. Resetting...")
            obs, edge_index, mask = env.reset()
            obs_t = torch.tensor(obs)
            mask_t = torch.tensor(mask)
            stats = {'completed': 0, 'dead': 0}

if __name__ == "__main__":
    main()