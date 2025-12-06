import os
import time
import torch
import numpy as np
from collections import deque

# Попытка импорта WandB для логирования
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Logging to console only.")

from ppo_trainer import PPOTrainer

# --- КОНФИГУРАЦИЯ ---
# --- КОНФИГУРАЦИЯ ДЛЯ ДИПЛОМА (HARD MODE) ---
CONFIG = {
    # Среда
    "MAP_WIDTH": 20,          # Большая карта
    "MAP_HEIGHT": 20,
    "NUM_AGENTS": 8,          # Плотный трафик
    "WALL_DENSITY": 0.1,     # Лабиринт (стены есть)
    "COMM_RADIUS": 10.0,      # Ограниченная связь (GraphSAGE станет важен)
    "K_VISIBLE_TASKS": 5,     # Видят больше задач
    
    # Обучение (Сделаем его более стабильным)
    "TOTAL_TIMESTEPS": 1_000_000, # Миллион шагов (на ночь)
    "NUM_STEPS": 512,         # Длиннее эпизоды
    "BATCH_SIZE": 128,        # Больше батч для стабильности
    "LR": 2.5e-4,             # Чуть медленнее учимся, чтобы не "забывать"
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.03,         # Умеренная энтропия
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 4,
    
    # --- ФИЗИКА И НАГРАДЫ ---
    "W1_PROGRESS": 2.0,       # Двигаться к цели - хорошо
    "W2_ENERGY": 0.05,         # Тратить энергию - ПЛОХО (Включили физику!)
    "W3_FEAR": 5.0,           # Страх смерти высокий
    "REWARD_BONUS": 50.0,     # Бонус за доставку
    "BATTERY_THRESHOLD": 0.3,
    "SHIELD_PENALTY": 2.0,    # Штраф за помощь Shield (чтобы учились сами)
    
    # Системные
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "CHECKPOINT_FREQ": 100,   # Реже сохраняем
    "VALIDATION_FREQ": 50,
    "EXP_NAME": "DEG-RL_Hard_Physics_v2" # Новое имя папки
}

def run_validation(trainer, num_steps=200):
    """
    Запускает прогон без шума (детерминированный) для оценки качества.
    """
    env = trainer.env
    obs, edge_index, mask = env.reset()
    
    # Перенос на устройство
    obs = torch.tensor(obs).to(trainer.device)
    mask = torch.tensor(mask).to(trainer.device)
    
    total_rewards = 0
    completed_tasks = 0
    dead_agents = 0
    
    # Для валидации отключаем Shield Penalty в статистике (но Shield работает физически)
    
    for _ in range(num_steps):
        with torch.no_grad():
            # deterministic=True -> выбираем действие с макс. вероятностью
            action, _, _, _ = trainer.model.get_action(
                obs, edge_index, mask, deterministic=True
            )
        
        # Конвертация для среды
        cpu_actions = action.cpu().numpy().tolist()
        
        # Проверка Shield (он все равно нужен для безопасности)
        safe_actions = []
        for i, agent in enumerate(env.agents):
            if agent.is_dead:
                safe_actions.append(0)
                continue
            visible = env.obs_builder.find_k_nearest_tasks(agent)
            final_act, _ = trainer.shield.check_action(agent, cpu_actions[i], visible)
            safe_actions.append(final_act)

        # Шаг среды
        next_obs_np, rewards, dones, infos = env.step(safe_actions)
        
        # Сбор статистики
        total_rewards += np.sum(rewards)
        
        # Подсчет выполненных задач (через бонус в награде или состояние агента)
        # Здесь упрощенно: если награда большая, значит выполнил задачу
        completed_tasks += np.sum(rewards > 10.0) 
        
        obs = torch.tensor(next_obs_np[0]).to(trainer.device)
        edge_index = next_obs_np[1] # Уже тензор
        mask = torch.tensor(next_obs_np[2]).to(trainer.device)
        
        if all(dones):
            break
            
    dead_agents = sum([1 for a in env.agents if a.is_dead])
    survival_rate = (env.n_agents - dead_agents) / env.n_agents
    
    return {
        "val_mean_reward": total_rewards / num_steps,
        "val_survival_rate": survival_rate,
        "val_completed_tasks": completed_tasks
    }

def main():
    # 1. Инициализация
    if WANDB_AVAILABLE:
        wandb.init(
            project="MAPF_Energy_RL",
            name=CONFIG["EXP_NAME"],
            config=CONFIG
        )
    
    # Создаем папку для моделей
    # Создаем уникальную папку для этого эксперимента
    save_dir = os.path.join("checkpoints", CONFIG["EXP_NAME"])
    os.makedirs(save_dir, exist_ok=True)    
    
    # Инициализация тренера
    trainer = PPOTrainer(CONFIG)
    
    # Начальный сброс среды
    obs, edge_index, mask = trainer.env.reset()
    obs = torch.tensor(obs).to(trainer.device)
    mask = torch.tensor(mask).to(trainer.device)
    
    num_updates = CONFIG["TOTAL_TIMESTEPS"] // CONFIG["NUM_STEPS"]
    start_time = time.time()
    
    print(f"Starting training for {num_updates} updates...")

    # --- ВНЕШНИЙ ЦИКЛ ОБУЧЕНИЯ ---
    for update in range(1, num_updates + 1):
        
        # 2. Сбор данных (Rollout)
        # Передаем текущее состояние, получаем буфер и новое состояние
        buffer = trainer.collect_rollouts(obs, edge_index, mask)
        
        # Обновляем переменные состояния для следующего цикла
        obs = buffer['next_obs']
        edge_index = buffer['next_edge_index']
        mask = buffer['next_action_mask']
        
        # 3. Расчет GAE
        advantages, returns = trainer.compute_gae(buffer)
        
        # 4. Обновление политики (PPO Update)
        train_metrics = trainer.update_policy(buffer, advantages, returns)
        
        # Расчет метрик обучения (из буфера)
        train_rewards = [t.cpu().numpy() for t in buffer['rewards']]
        avg_train_reward = np.mean(np.sum(train_rewards, axis=0)) # Сумма за эпизод (роллаут)
        
        # 5. Валидация
        val_metrics = {}
        if update % CONFIG["VALIDATION_FREQ"] == 0:
            print(f"Validating at update {update}...")
            val_metrics = run_validation(trainer)
            print(f"Validation Results: {val_metrics}")
            

        # 6. Логирование
        logs = {
            "global_step": update * CONFIG["NUM_STEPS"],
            "train/loss": train_metrics["loss"],
            "train/value_loss": train_metrics["v_loss"],
            "train/entropy": train_metrics["entropy"],
            "train/mean_reward": avg_train_reward,
            "time/fps": int((update * CONFIG["NUM_STEPS"]) / (time.time() - start_time))
        }
        logs.update(val_metrics) # Добавляем метрики валидации, если они есть
        
        if WANDB_AVAILABLE:
            wandb.log(logs)
        else:
            if update % 10 == 0:
                print(f"Update {update} | Loss: {train_metrics['loss']:.3f} | Reward: {avg_train_reward:.1f}")

        # 7. Сохранение
        if update % CONFIG["CHECKPOINT_FREQ"] == 0:
            # Сохраняем внутрь save_dir
            path = os.path.join(save_dir, f"model_step_{logs['global_step']}.pth")
            trainer.save_model(path)

    # Финальное сохранение
    trainer.save_model(os.path.join(save_dir, "model_final.pth"))

if __name__ == "__main__":
    main()