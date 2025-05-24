"""Utility functions for the chess UI."""
import math

def format_time(seconds):
    """Format time in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def ease_out_quad(x):
    """Quadratic easing function for smooth animations"""
    return 1 - (1 - x) * (1 - x)

def ease_out_bounce(x):
    """Bounce easing function for piece landing animation"""
    n1 = 7.5625
    d1 = 2.75

    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * x * x + 0.984375

def lerp_color(color1, color2, t):
    """Linear interpolation between two colors"""
    return tuple(
        int(color1[i] + (color2[i] - color1[i]) * t)
        for i in range(len(color1))
    )

def create_gradient_surface(width, height, color_top, color_bottom):
    """Create a vertical gradient surface"""
    import pygame
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    for y in range(height):
        color = lerp_color(color_top, color_bottom, y / height)
        pygame.draw.line(surface, color, (0, y), (width, y))
        
    return surface
