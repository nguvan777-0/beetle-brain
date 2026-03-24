"""Draw food pellets."""
import pygame


def draw_food(surf, food):
    for fx, fy in food:
        pygame.draw.circle(surf, (50, 190, 70), (int(fx), int(fy)), 3)
