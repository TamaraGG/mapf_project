import random
import copy
from typing import List
from agent_state import AgentState, AgentStatus

class LNSOptimizer:
    def __init__(self, planner, energy_lambda=0.5):
        self.planner = planner
        self.energy_lambda = energy_lambda

    def step(self, agents, current_time, k=3, iterations=5):
        candidates = self._select(agents, k)
        if not candidates: return
        
        new_paths = self._repair(agents, candidates, current_time)
        
        if new_paths:
            # Применяем новые пути
            for aid, path in new_paths.items():
                agent = next(a for a in agents if a.id == aid)
                # path[0] - это текущая позиция (время t), path[1] - следующий шаг
                if len(path) > 1:
                    agent.assign_path(path[1:])
                else:
                    agent.assign_path([])

    def _select(self, agents, k):
        cands = []
        for a in agents:
            if (a.is_dead or 
                not a.path or 
                a.current_task is None or 
                a.status in [AgentStatus.CHARGING]): 
                continue
            
            # Чем длиннее путь, тем больше смысла его оптимизировать
            score = len(a.path)
            cands.append((score, a.id))
        
        cands.sort(key=lambda x: x[0], reverse=True)
        # Берем топ-k + немного случайности
        top = [x[1] for x in cands[:k*2]]
        if not top: return []
        return random.sample(top, min(len(top), k))

    def _repair(self, agents, ids, time):
        rt = self.planner.rt
        targets = [a for a in agents if a.id in ids]
        
        # 1. Сохраняем старые пути и удаляем их из RT
        backup_paths = {} 
        
        for a in targets:
            # Восстанавливаем полный путь: текущая позиция + будущие шаги
            current_pos_path = [(a.pos[0], a.pos[1], time)]
            full_future_path = current_pos_path + a.path
            
            # Также нужно учесть "хвост" ожидания, который ставит planner
            # Но для простоты LNS удаляем только активную часть.
            # В идеале нужно чистить всё.
            # Здесь мы используем clear_path для пути движения.
            
            backup_paths[a.id] = full_future_path
            rt.clear_path(full_future_path, a.id)

        # 2. Перемешиваем порядок планирования
        random.shuffle(targets)
        new_paths = {}
        success = True
        
        for a in targets:
            if a.current_task is None: 
                success = False; break

            # Планируем
            path = self.planner.solver.find_path(
                a.id, a.pos, a.current_task.goal_pos, a.profile, 
                time, a.battery, self.planner.safety_buffer, 
                self.energy_lambda, self.planner.charger_capacity
            )
            
            if path:
                # [ВАЖНО] Сразу резервируем, чтобы следующий агент видел этот путь
                rt.reserve_path(path, a.id)
                new_paths[a.id] = path
            else:
                success = False
                break
        
        if not success:
            # Rollback: очищаем то, что успели запланировать
            for aid, path in new_paths.items():
                rt.clear_path(path, aid)
            
            # Возвращаем старые пути
            for a in targets:
                old_path = backup_paths[a.id]
                rt.reserve_path(old_path, a.id)
            
            return None
        
        return new_paths