from collections import deque
import random
from dataclasses import dataclass
from typing import Tuple, List
# Обратите внимание на импорт внутри пакета
from common.agent_state import AgentState, AgentStatus

@dataclass
class Task:
    id: int
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    importance: float = 0.5
    creation_time: int = 0

class TaskManager:
    def __init__(self, grid):
        self.grid = grid
        self.pending_tasks = deque()
        self.task_counter = 0

    def spawn_random_task(self, current_time: int, agents: List[AgentState] = None):
        attempts = 0
        while attempts < 100:
            start = (random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1))
            if start in self.grid.obstacles or self.grid.is_charger(start):
                attempts += 1; continue
            
            goal = (random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1))
            if goal in self.grid.obstacles or self.grid.is_charger(goal) or goal == start:
                attempts += 1; continue

            if agents:
                collision = False
                for a in agents:
                    if a.pos == start or a.pos == goal: collision = True
                if collision: attempts += 1; continue
            
            self.task_counter += 1
            self.pending_tasks.append(Task(self.task_counter, start, goal, random.random(), current_time))
            return

    # --- МЕТОД ДЛЯ CENTRALIZED ---
    def assign_tasks(self, agents: List[AgentState]):
        for agent in agents:
            if agent.status == AgentStatus.IDLE and not agent.is_dead:
                threshold = agent.profile.battery_capacity * 0.3
                if agent.battery < threshold:
                    continue

                if self.pending_tasks:
                    task = self.pending_tasks.popleft()
                    agent.current_task = task
                    agent.status = AgentStatus.WORKING

    # --- МЕТОД ДЛЯ DECENTRALIZED (RL) ---
    def claim_task(self, task: Task):
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)

    def return_task(self, task: Task):
        if task and task.id != -1: 
            self.pending_tasks.appendleft(task)

    def clean_invalid_tasks(self):
        valid_tasks = deque()
        while self.pending_tasks:
            task = self.pending_tasks.popleft()
            if (task.start_pos in self.grid.obstacles or 
                task.goal_pos in self.grid.obstacles):
                # print(f"[TaskMgr] Задача {task.id} отменена (зона заблокирована)")
                continue
            valid_tasks.append(task)
        self.pending_tasks = valid_tasks