"""Draw food pellets and hydrothermal vent glows."""
import pygame
from sim.config import VENT_RADIUS

_VENT_COLOR = (40, 80, 120)   # dark blue-teal glow
_FOOD_COLOR = (50, 190, 70)


def draw_food(surf, food, vents):
    for vx, vy in vents:
        pygame.draw.circle(surf, _VENT_COLOR, (int(vx), int(vy)), int(VENT_RADIUS), 1)
        pygame.draw.circle(surf, (30, 55, 85), (int(vx), int(vy)), int(VENT_RADIUS // 2), 1)
    for fx, fy in food:
        pygame.draw.circle(surf, _FOOD_COLOR, (int(fx), int(fy)), 3)
