"""Sparkline graph widget for the stats panel."""
import pygame


def draw_sparkline(surf, data, rect, color, mn=None, mx=None):
    if len(data) < 2:
        return
    x0, y0, w, h = rect
    mn  = mn if mn is not None else min(data)
    mx  = mx if mx is not None else max(data)
    rng = mx - mn if mx != mn else 1
    pts = [
        (x0 + int(i / (len(data) - 1) * w),
         y0 + h - int((v - mn) / rng * h))
        for i, v in enumerate(data)
    ]
    pygame.draw.lines(surf, color, False, pts, 1)
    font_tiny = pygame.font.SysFont("monospace", 10)
    lbl = font_tiny.render(f"{data[-1]:.2f}", True, color)
    surf.blit(lbl, (x0 + w + 3, y0 + h // 2 - 5))
