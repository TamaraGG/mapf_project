import time
import random
import csv
import statistics
from typing import List

from common.grid_map import GridMap
from common.agent_profile import AGENT_PROFILES
from agent_state import AgentState, AgentStatus
from reservation_table import ReservationTable
from task_manager import TaskManager
from central_planner import CentralPlanner
from lns_optimizer import LNSOptimizer
from visualizer import PygameVisualizer

# --- КОНФИГУРАЦИЯ СЦЕНАРИЕВ ---
SCENARIO_CONFIG = {
    "BASELINE": {
        "SHOCK_EVENT": False,
        "AGENTS": 10,
        "MAP_SIZE": 20,
        "FILENAME": "results_baseline.csv"
    },
    "ADAPTABILITY_TEST": {
        "SHOCK_EVENT": True,     # Включить "обвал"
        "SHOCK_TICK": 150,       # Когда происходит событие
        "AGENTS": 15,            # Чуть больше агентов для теста пробок
        "MAP_SIZE": 25,
        "FILENAME": "results_shock.csv"
    }
}

# !!! ВЫБЕРИТЕ СЦЕНАРИЙ ЗДЕСЬ !!!
# CURRENT_SCENARIO = 'BASELINE'
CURRENT_SCENARIO = 'ADAPTABILITY_TEST'

CONFIG = {
    "MAP_WIDTH": SCENARIO_CONFIG[CURRENT_SCENARIO]["MAP_SIZE"],
    "MAP_HEIGHT": SCENARIO_CONFIG[CURRENT_SCENARIO]["MAP_SIZE"],
    "NUM_AGENTS": SCENARIO_CONFIG[CURRENT_SCENARIO]["AGENTS"],
    "MAX_STEPS": 500,
    "SLEEP_TIME": 0.1, # Задержка для визуализации (0.0 для макс скорости)
    "WALL_DENSITY": 0.1,
    "SAFETY_BUFFER": 15.0, 
    "GAMMA": 0.8,
    "CHARGER_CAPACITY": 2, 
    "USE_LNS": True
}

class MetricsLogger:
    def __init__(self, filename):
        self.filename = filename
        self.history = []
        self.total_completed = 0
        self.planning_latencies = []

    def log_step(self, tick, agents, completed_now, latency):
        self.total_completed += completed_now
        self.planning_latencies.append(latency)
        
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
            print(f"\n[INFO] Детальные логи сохранены в файл: {self.filename}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить CSV: {e}")

    def print_final_report(self, num_agents, total_ticks):
        if self.total_completed == 0:
            eta_sys = 0
        else:
            final_energy = self.history[-1]['total_energy']
            eta_sys = final_energy / self.total_completed

        throughput = (self.total_completed / total_ticks) * 1000
        final_dead = self.history[-1]['dead_agents']
        sr = ((num_agents - final_dead) / num_agents) * 100.0
        avg_lat = statistics.mean(self.planning_latencies) * 1000 if self.planning_latencies else 0
        max_lat = max(self.planning_latencies) * 1000 if self.planning_latencies else 0

        print("\n" + "="*50)
        print(f"ИТОГОВЫЙ ОТЧЕТ ({CURRENT_SCENARIO})")
        print("="*50)
        print(f"1. Энергоэффективность (η_sys): {eta_sys:.2f} ед./задачу")
        print(f"2. Пропускная способность:      {throughput:.2f} задач/1000 тиков")
        print(f"3. Выживаемость (SR):           {sr:.1f}% (Умерло: {final_dead})")
        print(f"4. Вычислительная задержка:     Ср: {avg_lat:.2f} мс | Макс: {max_lat:.2f} мс")
        print("="*50 + "\n")

def generate_map(w, h, density):
    grid = [['.' for _ in range(w)] for _ in range(h)]
    for _ in range(int(w * h * density)):
        grid[random.randint(0, h-1)][random.randint(0, w-1)] = '@'
    
    # Гарантированные зарядки по углам и в центре
    chargers = [(1,1), (w-2, h-2), (w//2, h//2), (w-2, 1), (1, h-2)]
    for cx, cy in chargers:
        if 0 <= cx < w and 0 <= cy < h: grid[cy][cx] = 'C'
    return "\n".join("".join(row) for row in grid)

def apply_shock_event(grid, agents, task_manager):
    print(f"\n[{time.strftime('%H:%M:%S')}] !!! SHOCK EVENT: ОБВАЛ В ЦЕНТРЕ КАРТЫ !!!")
    center_x, center_y = grid.width // 2, grid.height // 2
    radius = 3
    
    affected_cells = []
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if 0 <= x < grid.width and 0 <= y < grid.height:
                if not grid.is_charger((x, y)):
                    grid.add_dynamic_obstacle(x, y)
                    affected_cells.append((x, y))

    for a in agents:
        if a.pos in affected_cells:
            if not a.is_dead:
                print(f"[Shock] Агент {a.id} погиб под завалом в {a.pos}!")
                a.status = AgentStatus.DEAD
                a.path = []
                a.current_task = None
                a.battery = 0

    for a in agents:
        if not a.is_dead and a.current_task:
            if a.current_task.goal_pos in grid.obstacles:
                a.current_task = None
                a.status = AgentStatus.IDLE
                a.path = []

    task_manager.clean_invalid_tasks()

def main():
    map_str = generate_map(CONFIG["MAP_WIDTH"], CONFIG["MAP_HEIGHT"], CONFIG["WALL_DENSITY"])
    grid = GridMap.from_string(map_str)
    viz = PygameVisualizer(grid)

    agents = []
    types = ["HeavyTruck", "FastDrone", "StandardBot"]
    for i in range(CONFIG["NUM_AGENTS"]):
        prof = AGENT_PROFILES[types[i % 3]]
        while True:
            sx, sy = random.randint(0, grid.width-1), random.randint(0, grid.height-1)
            if (sx, sy) not in grid.obstacles and not grid.is_charger((sx, sy)):
                if not any(a.pos == (sx, sy) for a in agents): break
        
        a = AgentState(i, prof, (sx, sy))
        # Старт с разным зарядом, чтобы они не поехали на зарядку все одновременно
        a.battery = prof.battery_capacity * random.uniform(0.5, 1.0)
        agents.append(a)

    rt = ReservationTable()
    tm = TaskManager(grid)
    planner = CentralPlanner(
        grid, rt, 
        gamma=CONFIG["GAMMA"], 
        safety_buffer=CONFIG["SAFETY_BUFFER"],
        charger_capacity=CONFIG["CHARGER_CAPACITY"]
    )
    lns = LNSOptimizer(planner)
    
    scenario_data = SCENARIO_CONFIG[CURRENT_SCENARIO]
    logger = MetricsLogger(scenario_data["FILENAME"])

    print(f"Запуск сценария: {CURRENT_SCENARIO}")
    print(f"Агентов: {CONFIG['NUM_AGENTS']}, Карта: {CONFIG['MAP_WIDTH']}x{CONFIG['MAP_HEIGHT']}")

    # [ИСПРАВЛЕНИЕ] Предварительная генерация задач, чтобы всем хватило на старте
    print("Генерация стартового пула задач...")
    while len(tm.pending_tasks) < CONFIG["NUM_AGENTS"] * 1.5:
        tm.spawn_random_task(0, agents)
    
    tm.assign_tasks(agents)

    # --- ГЛАВНЫЙ ЦИКЛ ---
    for tick in range(CONFIG["MAX_STEPS"]):
        
        # 1. Шок-событие
        if scenario_data["SHOCK_EVENT"] and tick == scenario_data["SHOCK_TICK"]:
            apply_shock_event(grid, agents, tm)
            for a in agents: a.path = []

        # 2. Генерация новых задач
        if len(tm.pending_tasks) < CONFIG["NUM_AGENTS"]:
            tm.spawn_random_task(tick, agents)
        
        tm.assign_tasks(agents)

        # 3. ПЛАНИРОВАНИЕ
        t_start = time.perf_counter()
        planner.plan(agents, tick, tm)
        if CONFIG["USE_LNS"] and tick % 5 == 0:
            lns.step(agents, tick, k=3, iterations=5)
        latency = time.perf_counter() - t_start

        # 4. ИСПОЛНЕНИЕ И ЛОГИКА ЗАДАЧ
        completed_this_tick = 0
        for a in agents:
            # Проверка выполнения этапов задачи
            if a.status == AgentStatus.WORKING and a.current_task:
                # А. Прибыли на точку старта (PICKUP)
                if not a.has_picked_up and a.pos == a.current_task.start_pos:
                    a.has_picked_up = True
                    a.path = [] # Сброс пути, чтобы планировщик построил путь к финишу
                
                # Б. Прибыли на точку финиша (DELIVERY)
                elif a.has_picked_up and a.pos == a.current_task.goal_pos:
                    completed_this_tick += 1
                    a.has_picked_up = False
                    a.current_task = None
                    a.status = AgentStatus.IDLE
                    a.path = []

            # Физический шаг
            a.step(grid)

        # 5. Логирование и Визуализация
        logger.log_step(tick, agents, completed_this_tick, latency)
        stats = {'completed': logger.total_completed, 'dead': sum(1 for a in agents if a.is_dead)}
        viz.render(agents, tick, stats, tm)
        
        if CONFIG["SLEEP_TIME"] > 0:
            time.sleep(CONFIG["SLEEP_TIME"])

    logger.save_to_csv()
    logger.print_final_report(CONFIG["NUM_AGENTS"], CONFIG["MAX_STEPS"])
    
    time.sleep(5)

if __name__ == "__main__":
    main()