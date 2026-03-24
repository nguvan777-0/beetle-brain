"""Draw food pellets and hydrothermal vent glows."""
import pygame
from sim.config import VENT_RADIUS

_FOOD_COLOR = (50, 190, 70)
# radii at which density has dropped to 1/4, 1/16, 1/64 of centre
_RINGS = [
    (int(VENT_RADIUS * 0.13), (60, 120, 180)),
    (int(VENT_RADIUS * 0.33), (40,  85, 130)),
    (int(VENT_RADIUS),        (22,  48,  80)),
]


def draw_food(surf, food, vents):
    for vx, vy in vents:
        cx, cy = int(vx), int(vy)
        for r, color in _RINGS:
            pygame.draw.circle(surf, color, (cx, cy), r, 1)
    for fx, fy in food:
        pygame.draw.circle(surf, _FOOD_COLOR, (int(fx), int(fy)), 3)
