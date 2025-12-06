import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from typing import Dict, Any

from model import ActorCritic
from rl_environment import RLEnvironment
from safety_shield import SafetyShield

class PPOTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("DEVICE", "cpu"))
        
        # Гиперпараметры PPO
        self.lr = config.get("LR", 3e-4)
        self.gamma = config.get("GAMMA", 0.99)
        self.gae_lambda = config.get("GAE_LAMBDA", 0.95)
        self.clip_coef = config.get("CLIP_COEF", 0.2)
        self.ent_coef = config.get("ENT_COEF", 0.01)
        self.vf_coef = config.get("VF_COEF", 0.5)
        self.max_grad_norm = config.get("MAX_GRAD_NORM", 0.5)
        self.num_steps = config.get("NUM_STEPS", 128) # Длина роллаута
        self.batch_size = config.get("BATCH_SIZE", 64)
        self.update_epochs = config.get("UPDATE_EPOCHS", 4)
        
        # Штраф за срабатывание Shield
        self.shield_penalty = config.get("SHIELD_PENALTY", 5.0)

        # Инициализация среды и моделей
        self.env = RLEnvironment(config)
        
        # Получаем размерности из среды после первого сброса (или рассчитываем)
        # Временный сброс для инициализации obs_builder
        self.env.reset() 
        self.obs_dim = self.env.obs_builder.encode(self.env.agents[0], self.env.agents).shape[0]
        self.action_dim = self.env.action_dim
        
        self.model = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        
        # Инициализация Shield (ему нужна карта из env)
        self.shield = SafetyShield(self.env.grid, config)

    def train(self, total_timesteps: int):
        """
        Основной цикл обучения.
        """
        num_updates = total_timesteps // self.num_steps
        start_time = time.time()
        
        # Метрики для логирования
        ep_rewards = deque(maxlen=10)
        ep_survival = deque(maxlen=10)
        
        # Начальное состояние
        obs, edge_index, action_mask = self.env.reset()
        
        # Перенос на устройство
        obs = torch.tensor(obs).to(self.device)
        # edge_index уже тензор на нужном устройстве из GraphBuilder
        action_mask = torch.tensor(action_mask).to(self.device)

        print(f"Starting training on {self.device}...")
        print(f"Obs Dim: {self.obs_dim}, Action Dim: {self.action_dim}")

        global_step = 0
        for update in range(1, num_updates + 1):
            # 1. Сбор данных (Rollout)
            buffer = self.collect_rollouts(obs, edge_index, action_mask)
            
            # Обновляем текущее состояние для следующей итерации (последнее из роллаута)
            obs = buffer['next_obs']
            edge_index = buffer['next_edge_index']
            action_mask = buffer['next_action_mask']
            
            # 2. Расчет GAE
            advantages, returns = self.compute_gae(buffer)
            
            # 3. Обновление политики (PPO Update)
            metrics = self.update_policy(buffer, advantages, returns)
            
            global_step += self.num_steps
            
            # Логирование
            if update % 10 == 0:
                avg_rew = np.mean([sum(r) for r in buffer['rewards']]) # Средняя награда за шаг (суммарно по агентам)
                # Для корректного отображения лучше считать среднюю награду на эпизод, 
                # но здесь у нас непрерывная среда.
                
                # Считаем выживаемость в текущем батче
                dones_count = np.sum(buffer['dones'])
                survival_rate = 1.0 - (dones_count / (self.num_steps * self.env.n_agents))
                
                print(f"Step {global_step} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Val: {metrics['v_loss']:.4f} | "
                      f"Ent: {metrics['entropy']:.4f} | "
                      f"Srv: {survival_rate:.1%} | "
                      f"FPS: {int(global_step / (time.time() - start_time))}")

            # Сохранение модели
            if update % 50 == 0:
                self.save_model(f"ppo_deg_rl_step_{global_step}.pth")

    def collect_rollouts(self, obs, edge_index, action_mask):
        """
        Запускает агентов в среде и собирает траектории.
        """
        buffer = {
            'obs': [], 'edge_index': [], 'masks': [],
            'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []
        }
        
        for step in range(self.num_steps):
            with torch.no_grad():
                # Получаем действие от нейросети
                action, logprob, _, value = self.model.get_action(
                    obs, edge_index, action_mask
                )

            # --- SAFETY SHIELD BLOCK ---
            # Конвертируем действия в CPU список для симуляции
            cpu_actions = action.cpu().numpy().tolist()
            safe_actions = []
            shield_penalties = np.zeros(self.env.n_agents)
            
            # Проверяем каждого агента
            # Нам нужно передать видимые задачи в Shield. 
            # Это немного накладно, но необходимо для корректности.
            # В оптимизированной версии Shield был бы встроен в env.step.
            
            for i, agent in enumerate(self.env.agents):
                if agent.is_dead:
                    safe_actions.append(0) # Dead -> Wait
                    continue
                
                # Получаем задачи, которые видит агент (для Shield)
                visible_tasks = self.env.obs_builder.find_k_nearest_tasks(agent)
                
                # Проверка
                final_act, corrected = self.shield.check_action(agent, cpu_actions[i], visible_tasks)
                safe_actions.append(final_act)
                
                if corrected:
                    shield_penalties[i] = self.shield_penalty

            # --- ENV STEP ---
            next_obs_np, rewards, dones, infos = self.env.step(safe_actions)
            
            # Применяем штраф за Shield к награде
            total_rewards = rewards - shield_penalties
            
            # Подготовка следующего шага
            next_obs = torch.tensor(next_obs_np[0]).to(self.device)
            next_edge_index = next_obs_np[1] # Уже тензор
            next_action_mask = torch.tensor(next_obs_np[2]).to(self.device)

            # Сохранение в буфер
            # Важно: сохраняем ОРИГИНАЛЬНОЕ действие (action) и logprob,
            # чтобы PPO учился не делать ошибок.
            buffer['obs'].append(obs)
            buffer['edge_index'].append(edge_index)
            buffer['masks'].append(action_mask)
            buffer['actions'].append(action)
            buffer['logprobs'].append(logprob)
            buffer['values'].append(value.flatten())
            buffer['rewards'].append(torch.tensor(total_rewards).to(self.device))
            buffer['dones'].append(torch.tensor(dones).to(self.device))
            
            obs = next_obs
            edge_index = next_edge_index
            action_mask = next_action_mask
            
            # Если все умерли - ресет (опционально, для lifelong можно не делать)
            if all(dones):
                obs_np, edge_idx, mask_np = self.env.reset()
                obs = torch.tensor(obs_np).to(self.device)
                edge_index = edge_idx
                action_mask = torch.tensor(mask_np).to(self.device)

        # Сохраняем состояние "после последнего шага" для GAE
        buffer['next_obs'] = obs
        buffer['next_edge_index'] = edge_index
        buffer['next_action_mask'] = action_mask
        
        return buffer

    def compute_gae(self, buffer):
        """
        Считает Generalized Advantage Estimation.
        """
        with torch.no_grad():
            next_value = self.model.get_value(
                buffer['next_obs'], buffer['next_edge_index']
            ).flatten()
            
        rewards = buffer['rewards']
        dones = buffer['dones']
        values = buffer['values']
        
        advantages = torch.zeros_like(torch.stack(rewards)).to(self.device)
        lastgaelam = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - dones[t].float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t].float()
                nextvalues = values[t+1]
                
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + torch.stack(values)
        return advantages, returns

    def update_policy(self, buffer, advantages, returns):
        """
        PPO Update Epochs.
        Из-за переменных графов мы не можем просто сделать flatten и batch.
        Придется итерироваться хитрее.
        """
        # Flattening list of tensors -> Tensor
        # Obs: (Num_Steps, N_Agents, Dim) -> (Batch, Dim)
        b_obs = torch.cat(buffer['obs']) 
        b_actions = torch.cat(buffer['actions'])
        b_logprobs = torch.cat(buffer['logprobs'])
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = torch.cat(buffer['values'])
        b_masks = torch.cat(buffer['masks'])
        
        # Edge Indices - это список списков, их нельзя сплющить в один тензор.
        # Нам нужно сохранить соответствие: b_obs[i] <-> b_edge_index[i]
        # Но b_obs сплющен по времени и агентам.
        # А edge_index один на всех агентов в один момент времени.
        
        # РЕШЕНИЕ:
        # Мы будем итерироваться по ВРЕМЕННЫМ ШАГАМ (Steps), а не по отдельным агентам.
        # В один шаг времени у нас N агентов и 1 граф.
        # Это "Mini-Batch" размером N_Agents.
        
        idxs = np.arange(self.num_steps)
        
        clip_fracs = []
        total_loss = 0
        v_loss = 0
        entropy_loss = 0
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(idxs)
            
            for t in idxs:
                # Берем данные за один временной шаг t
                # Это батч из N_Agents примеров
                
                # Данные из буфера (они уже на GPU)
                mb_obs = buffer['obs'][t]
                mb_edge_index = buffer['edge_index'][t]
                mb_masks = buffer['masks'][t]
                mb_actions = buffer['actions'][t]
                mb_logprobs = buffer['logprobs'][t]
                mb_advantages = advantages[t]
                mb_returns = returns[t]
                mb_values = buffer['values'][t]

                # Forward pass (Evaluate)
                # Получаем новые логиты и значения для ВСЕХ агентов в этот шаг времени
                _, new_logprob, entropy, new_value = self.model.get_action(
                    mb_obs, mb_edge_index, mb_masks
                )
                
                new_value = new_value.view(-1)

                # Policy Loss
                logratio = new_logprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    new_value - mb_values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss_val = 0.5 * v_loss_max.mean()

                # Entropy Loss
                entropy_loss_val = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss_val + self.vf_coef * v_loss_val

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                v_loss += v_loss_val.item()
                entropy_loss += entropy_loss_val.item()

        return {
            "loss": total_loss / (self.num_steps * self.update_epochs),
            "v_loss": v_loss / (self.num_steps * self.update_epochs),
            "entropy": entropy_loss / (self.num_steps * self.update_epochs)
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")