import pygame
from .config import *

class MoveAnimation:
    def __init__(self, start_pos, end_pos, piece_image, duration=0.2):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.piece_image = piece_image
        self.duration = duration
        self.progress = 0
        
    def update(self, dt):
        self.progress = min(1.0, self.progress + dt/self.duration)
        
    def draw(self, screen):
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * self.progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * self.progress
        screen.blit(self.piece_image, (x, y))