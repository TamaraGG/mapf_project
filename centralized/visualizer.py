import pygame
import sys
from centralized.agent_state import AgentStatus

CELL = 40
MARGIN = 2
PANEL = 100
COLORS = {
    'BG': (30, 30, 30), 'GRID': (50, 50, 50), 'WALL': (100, 100, 100),
    'CHG': (0, 200, 0), 'TRUCK': (200, 50, 50), 'DRONE': (50, 150, 255),
    'BOT': (200, 200, 200), 'DEAD': (50, 50, 50), 'TASK': (100, 100, 100)
}

class PygameVisualizer:
    def __init__(self, grid):
        pygame.init()
        self.grid = grid
        w = grid.width * (CELL + MARGIN) + MARGIN
        h = grid.height * (CELL + MARGIN) + MARGIN
        self.screen = pygame.display.set_mode((w, h + PANEL))
        pygame.display.set_caption("CEA-MAPF Sim")
        self.font = pygame.font.SysFont("Arial", 12)

    def render(self, agents, tick, stats, task_manager):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        self.screen.fill(COLORS['BG'])

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                rect = pygame.Rect(MARGIN+x*(CELL+MARGIN), MARGIN+y*(CELL+MARGIN), CELL, CELL)
                col = COLORS['GRID']
                if (x,y) in self.grid.obstacles: col = COLORS['WALL']
                elif self.grid.is_charger((x,y)): col = COLORS['CHG']
                pygame.draw.rect(self.screen, col, rect)
                if col == COLORS['CHG']:
                    self.screen.blit(self.font.render("C", True, (0,0,0)), (rect.centerx-4, rect.centery-8))

        for t in task_manager.pending_tasks:
            px, py = self._to_px(t.goal_pos)
            r = pygame.Rect(0,0, CELL-14, CELL-14)
            r.center = (px, py)
            pygame.draw.rect(self.screen, COLORS['TASK'], r, 2)

        for a in agents:
            if a.is_dead: continue
            col = self._get_col(a)
            spx = self._to_px(a.pos)
            
            if a.current_task:
                epx = self._to_px(a.current_task.goal_pos)
                line_col = (255, 255, 0) if a.status == AgentStatus.TO_CHARGER else col
                pygame.draw.line(self.screen, line_col, spx, epx, 2)
                if a.status != AgentStatus.TO_CHARGER:
                    r = pygame.Rect(0,0, CELL-10, CELL-10)
                    r.center = epx
                    pygame.draw.rect(self.screen, col, r, 2)

            pygame.draw.circle(self.screen, col, spx, CELL//2 - 4)
            # Если несет груз - рисуем белую точку в центре
            if a.has_picked_up:
                pygame.draw.circle(self.screen, (255, 255, 255), spx, 4)

            if a.status == AgentStatus.CHARGING:
                pygame.draw.circle(self.screen, (255,255,0), spx, CELL//2 - 4, 3)
            
            self.screen.blit(self.font.render(str(a.id), True, (255,255,255)), (spx[0]-4, spx[1]-8))

            fill = max(0, min(1, a.battery / a.profile.battery_capacity))
            hc = (0,255,0) if fill > 0.5 else (255,0,0)
            pygame.draw.rect(self.screen, (0,0,0), (spx[0]-12, spx[1]+6, 24, 4))
            pygame.draw.rect(self.screen, hc, (spx[0]-12, spx[1]+6, int(24*fill), 4))

        txt = f"Tick: {tick} | Done: {stats['completed']} | Dead: {stats['dead']} | Pending: {len(task_manager.pending_tasks)}"
        self.screen.blit(self.font.render(txt, True, (255,255,255)), (10, self.grid.height*(CELL+MARGIN)+10))
        pygame.display.flip()

    def _to_px(self, pos):
        return (MARGIN + pos[0]*(CELL+MARGIN) + CELL//2, MARGIN + pos[1]*(CELL+MARGIN) + CELL//2)

    def _get_col(self, a):
        if a.status == AgentStatus.DEAD: return COLORS['DEAD']
        if "Truck" in a.profile.type_name: return COLORS['TRUCK']
        if "Drone" in a.profile.type_name: return COLORS['DRONE']
        return COLORS['BOT']