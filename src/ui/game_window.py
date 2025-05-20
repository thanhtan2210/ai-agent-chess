import pygame
import chess
from .config import *
from .board_ui import BoardUI

class GameWindow:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(WINDOW_TITLE)
        
        # Khởi tạo board UI ở giữa màn hình
        board_offset_x = (WINDOW_SIZE[0] - BOARD_SIZE) // 2
        board_offset_y = (WINDOW_SIZE[1] - BOARD_SIZE) // 2
        self.board_ui = BoardUI(BOARD_SIZE, BOARD_SIZE)
        
        # Khởi tạo font chữ trước
        self.font = pygame.font.SysFont(FONTS['default'], FONT_SIZES['medium'])
        
        # Sau đó mới khởi tạo các button
        self.buttons = self._create_buttons()

    def _create_buttons(self):
        buttons = {
            'new_game': Button(50, 50, "New Game", self.font),
            'undo': Button(50, 100, "Undo", self.font),
            'settings': Button(50, 150, "Settings", self.font)
        }
        return buttons

    def draw(self, board, selected_square=None):
        # Vẽ background
        self.screen.fill(COLORS['background'])
        
        # Vẽ bàn cờ
        skip_squares = [] if selected_square is None else [selected_square]
        self.board_ui.draw_board(board, skip_squares)
        
        # Highlight ô được chọn
        if selected_square:
            self.board_ui.highlight_square(selected_square, COLORS['selected'])
            
        # Vẽ các button
        for button in self.buttons.values():
            button.draw(self.screen)
            
        # Cập nhật màn hình
        pygame.display.flip()

class Button:
    def __init__(self, x, y, text, font):
        self.rect = pygame.Rect(x, y, BUTTON_STYLES['default']['width'], BUTTON_STYLES['default']['height'])
        self.text = text
        self.font = font
        self.is_hovered = False

    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=BUTTON_STYLES['default']['border_radius'])
        
        text_surface = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False