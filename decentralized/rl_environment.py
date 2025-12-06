import numpy as np
import random
from typing import List, Dict, Any

from grid_map import GridMap
from agent_state import AgentState, AgentStatus
from agent_profile import AGENT_PROFILES
from task_manager import TaskManager
from graph_builder import GraphBuilder
from observation_builder import ObservationBuilder

class RLEnvironment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config["MAP_WIDTH"]
        self.height = config["MAP_HEIGHT"]
        self.n_agents = config["NUM_AGENTS"]
        
        self.w1_prog = config.get("W1_PROGRESS", 1.0)
        self.w2_energy = config.get("W2_ENERGY", 0.5)
        self.w3_fear = config.get("W3_FEAR", 5.0)
        self.bonus = config.get("REWARD_BONUS", 20.0)
        self.b_thr = config.get("BATTERY_THRESHOLD", 0.3)

        self.k_visible_tasks = config.get("K_VISIBLE_TASKS", 5)
        self.action_dim = 2 + self.k_visible_tasks 

        self.grid = None
        self.agents: List[AgentState] = []
        self.task_manager = None
        self.graph_builder = GraphBuilder(config)
        self.obs_builder = None

    @staticmethod
    def _generate_random_map_str(w, h, density):
        grid = [['.' for _ in range(w)] for _ in range(h)]
        for _ in range(int(w * h * density)):
            grid[random.randint(0, h-1)][random.randint(0, w-1)] = '@'
        chargers = [(1,1), (w-2, h-2), (w//2, h//2), (w-2, 1), (1, h-2)]
        for cx, cy in chargers:
            if 0 <= cx < w and 0 <= cy < h: grid[cy][cx] = 'C'
        return "\n".join("".join(row) for row in grid)

    def reset(self):
        map_str = self._generate_random_map_str(self.width, self.height, self.config["WALL_DENSITY"])
        self.grid = GridMap.from_string(map_str)
        self.task_manager = TaskManager(self.grid)
        self.obs_builder = ObservationBuilder(self.grid, self.task_manager, self.config)

        self.agents = []
        types = list(AGENT_PROFILES.keys())
        used_pos = set()
        
        for i in range(self.n_agents):
            p_name = types[i % len(types)]
            profile = AGENT_PROFILES[p_name]
            while True:
                rx, ry = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (rx, ry) not in self.grid.obstacles and (rx, ry) not in used_pos:
                    used_pos.add((rx, ry))
                    break
            
            agent = AgentState(i, profile, (rx, ry))
            agent.battery = profile.battery_capacity
            agent.target_pos = None
            agent.prev_target_pos = None
            self.agents.append(agent)

        for _ in range(self.n_agents * 2):
            self.task_manager.spawn_random_task(0)

        return self._get_full_state()

    def step(self, actions: List[int]):
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        dones = np.zeros(self.n_agents, dtype=bool)
        infos = [{} for _ in range(self.n_agents)]
        
        # 1. Интерпретация действий
        for i, agent in enumerate(self.agents):
            if agent.is_dead: continue
            
            # --- ИСПРАВЛЕНИЕ 1: Блокировка на зарядке ---
            # Если агент уже заряжается и не полон - он игнорирует нейросеть и стоит
            if agent.status == AgentStatus.CHARGING:
                if agent.battery < agent.profile.battery_capacity:
                    agent.target_pos = agent.pos # Стоять на месте
                    continue # Пропускаем выбор действия
                else:
                    agent.status = AgentStatus.IDLE # Зарядился, можно ехать
            
            action = actions[i]
            target = None
            
            if action == 0: # Wait
                target = agent.pos
            elif action == 1: # Charger
                target = self.grid.get_nearest_charger_pos(agent.pos)
                if target is None: target = agent.pos
                agent.status = AgentStatus.TO_CHARGER 
            else: # Task
                if agent.current_task:
                    if agent.status == AgentStatus.WORKING:
                        target = agent.current_task.goal_pos
                    else:
                        target = agent.current_task.start_pos
                else:
                    task_idx = action - 2
                    visible = self.obs_builder.find_k_nearest_tasks(agent)
                    if task_idx < len(visible):
                        selected_task = visible[task_idx]
                        if selected_task in self.task_manager.pending_tasks:
                            agent.current_task = selected_task
                            self.task_manager.claim_task(selected_task)
                            target = selected_task.start_pos
                            agent.status = AgentStatus.IDLE 
                        else:
                            target = agent.pos 
                    else:
                        target = agent.pos

            agent.target_pos = target

        # 2. Движение и Физика
        for i, agent in enumerate(self.agents):
            if agent.is_dead: 
                dones[i] = True
                continue

            dist_before = 0
            if agent.target_pos:
                dist_before = self.grid.get_heuristic(agent.pos, agent.target_pos)

            moved = False
            # Пытаемся двигаться, только если цель не достигнута
            if agent.target_pos and agent.target_pos != agent.pos:
                moved = self._move_greedy(agent, agent.target_pos)

            has_payload = (agent.status == AgentStatus.WORKING)
            e_cost = agent.profile.calculate_move_cost(has_payload) if moved else agent.profile.idle_consumption

            agent.battery -= e_cost
            agent.total_energy_consumed += e_cost
            
            if agent.battery <= 0:
                agent.battery = 0
                agent.status = AgentStatus.DEAD
                dones[i] = True
                rewards[i] -= 100.0
                continue 

            # --- БЛОК 3: ОБРАБОТКА СОБЫТИЙ ---
            reward_bonus = 0.0
            
            # А. ЗАРЯДКА (Исправленная логика)
            if self.grid.is_charger(agent.pos):
                # Если мы приехали на зарядку (по действию или нужде)
                should_charge = (actions[i] == 1) or \
                                (agent.status == AgentStatus.TO_CHARGER) or \
                                (agent.battery < agent.profile.battery_capacity * 0.95) # Гистерезис
                
                if should_charge:
                    agent.status = AgentStatus.CHARGING # Входим в режим зарядки
                    agent.battery = min(agent.battery + agent.profile.charging_speed, agent.profile.battery_capacity)
                    
                    # Если зарядились полностью - освобождаемся
                    if agent.battery >= agent.profile.battery_capacity:
                        agent.status = AgentStatus.IDLE

            # Б. PICKUP / DELIVERY
            if agent.current_task:
                if agent.status == AgentStatus.WORKING and agent.pos == agent.current_task.goal_pos:
                    reward_bonus = self.bonus
                    agent.current_task = None
                    agent.status = AgentStatus.IDLE
                    self.task_manager.spawn_random_task(0) 

                elif agent.status != AgentStatus.WORKING and agent.pos == agent.current_task.start_pos:
                    agent.status = AgentStatus.WORKING 
                    reward_bonus += 1.0 

            # 4. Награда
            dist_after = 0
            if agent.target_pos:
                dist_after = self.grid.get_heuristic(agent.pos, agent.target_pos)
            
            progress = 0.0
            if agent.prev_target_pos == agent.target_pos:
                progress = dist_before - dist_after
            
            agent.prev_target_pos = agent.target_pos

            b_norm = agent.battery / agent.profile.battery_capacity
            fear = (self.b_thr - b_norm) ** 2 if b_norm < self.b_thr else 0.0

            r = (self.w1_prog * progress) - (self.w2_energy * (e_cost / 10.0)) - (self.w3_fear * fear) + reward_bonus
            rewards[i] += r
            infos[i] = {"battery": agent.battery, "dead": agent.is_dead}

        if len(self.task_manager.pending_tasks) < self.n_agents * 2:
            self.task_manager.spawn_random_task(0)

        return self._get_full_state(), rewards, dones, infos

    def _get_full_state(self):
        obs_list = []
        for agent in self.agents:
            obs_list.append(self.obs_builder.encode(agent, self.agents))
        
        obs_tensor = np.array(obs_list, dtype=np.float32)
        edge_index = self.graph_builder.build_edge_index(self.agents)
        
        masks = np.ones((self.n_agents, self.action_dim), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            if agent.is_dead:
                masks[i, :] = 0.0
                masks[i, 0] = 1.0 # Разрешаем только Wait
                continue
            
            if agent.current_task:
                # Если есть задача, маскируем выбор новых задач (оставляем Wait, Charge, Continue Task)
                # Action 2 (индекс 0) зарезервирован как "Продолжить текущую задачу"
                masks[i, 3:] = 0.0 
            else:
                visible = len(self.obs_builder.find_k_nearest_tasks(agent))
                for k in range(self.k_visible_tasks):
                    if k >= visible:
                        masks[i, 2 + k] = 0.0
        
        return (obs_tensor, edge_index, masks)

        
    def _move_greedy(self, agent, target):
        if agent.pos == target: return False
        
        # Все возможные ходы
        candidates = []
        cx, cy = agent.pos
        
        # Перемешиваем направления, чтобы при равных условиях не было приоритета "вверх"
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(deltas)
        
        for dx, dy in deltas:
            nx, ny = cx + dx, cy + dy
            
            # 1. Проверка границ
            if not (0 <= nx < self.width and 0 <= ny < self.height): continue
            # 2. Проверка стен
            if (nx, ny) in self.grid.obstacles: continue
            
            # 3. Проверка других агентов (Коллизии)
            collision = False
            for other in self.agents:
                if other.id != agent.id and not other.is_dead:
                    if other.pos == (nx, ny): # Клетка занята
                        collision = True
                        break
                    # Простейшая защита от swap-конфликта (обмен местами)
                    # Если другой агент хочет пойти в мою клетку, а я в его - это авария.
                    # Но здесь мы знаем только текущие позиции.
            if collision: continue
            
            # Считаем эвристику до цели
            h = self.grid.get_heuristic((nx, ny), target)
            candidates.append(((nx, ny), h))
        
        if not candidates:
            return False # Тупик, некуда идти
            
        # Сортируем кандидатов по эвристике (от меньшего к большему)
        candidates.sort(key=lambda x: x[1])
        
        # Текущее расстояние
        current_h = self.grid.get_heuristic(agent.pos, target)
        
        # Берем лучший вариант
        best_pos, best_h = candidates[0]
        
        # --- ИСПРАВЛЕНИЕ 2: Разрешаем движение, если не стало хуже ---
        # Раньше было: if best_h < current_h.
        # Теперь: if best_h <= current_h.
        # Это позволяет агенту делать шаги в сторону (обходить препятствия),
        # не приближаясь, но и не отдаляясь.
        
        if best_h <= current_h:
            agent.pos = best_pos
            return True
            
        # Если все варианты ведут назад (best_h > current_h),
        # агент попал в "чашу" (локальный минимум).
        # В простом greedy он застрянет.
        # Можно разрешить случайный шаг с малой вероятностью (Random Walk),
        # чтобы выбраться из тупика.
        if random.random() < 0.1: # 10% шанс сделать "плохой" ход, чтобы разблокироваться
             agent.pos = best_pos
             return True
             
        return False

  