import pygame
import chess
import os
import math
from .config import *
from .utils import ease_out_quad, ease_out_bounce

class Animation:
    def __init__(self):
        self.current_animation = None
        self.start_time = 0
        self.fade_animations = []
        
    def start_move_animation(self, move, start_time):
        """Bắt đầu animation di chuyển quân cờ"""
        self.current_animation = {
            'type': 'move',
            'from_square': move.from_square,
            'to_square': move.to_square,
            'start_time': start_time,
            'duration': ANIMATION['move_duration']
        }
        
    def add_fade_animation(self, square, color, duration=None):
        """Thêm hiệu ứng fade cho một ô cờ"""
        if duration is None:
            duration = ANIMATION['fade_duration']
        
        self.fade_animations.append({
            'square': square,
            'color': color,
            'start_time': pygame.time.get_ticks(),
            'duration': duration
        })
        
    def update(self, current_time):
        """Cập nhật trạng thái của tất cả các animation"""
        # Update move animation
        if self.current_animation:
            progress = (current_time - self.current_animation['start_time']) / \
                      self.current_animation['duration']
            if progress >= 1:
                self.current_animation = None
        
        # Update fade animations
        self.fade_animations = [
            anim for anim in self.fade_animations
            if (current_time - anim['start_time']) / anim['duration'] < 1
        ]

class BoardUI:
    def __init__(self, screen):
        """Khởi tạo giao diện bàn cờ.
        
        Args:
            screen: pygame.Surface object để vẽ lên
        """
        self.screen = screen
        self.board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        self.coords_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        
        # Load fonts
        font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
        self.piece_font = pygame.font.Font(font_path, SQUARE_SIZE - 8)
        self.coord_font = pygame.font.SysFont(COORDINATES['font'], COORDINATES['size'])
        
        # Piece symbols
        self.unicode_pieces = {
            (chess.PAWN, chess.WHITE): "♙",
            (chess.KNIGHT, chess.WHITE): "♘",
            (chess.BISHOP, chess.WHITE): "♗",
            (chess.ROOK, chess.WHITE): "♖",
            (chess.QUEEN, chess.WHITE): "♕",
            (chess.KING, chess.WHITE): "♔",
            (chess.PAWN, chess.BLACK): "♟",
            (chess.KNIGHT, chess.BLACK): "♞",
            (chess.BISHOP, chess.BLACK): "♝",
            (chess.ROOK, chess.BLACK): "♜",
            (chess.QUEEN, chess.BLACK): "♛",
            (chess.KING, chess.BLACK): "♚",
        }
        
        # Animation state
        self.animation = Animation()
        self.hover_square = None
        self.selected_square = None
        self.valid_moves = []
        self.last_move = None
        
        # Pre-render coordinates
        self._render_coordinates()
        
    def _render_coordinates(self):
        """Pre-render tọa độ bàn cờ"""
        self.coords_surface.fill((0, 0, 0, 0))
        
        for i in range(8):
            # File coordinates (a-h)
            text = chr(ord('a') + i)
            surf = self.coord_font.render(text, True, COORDINATES['color'])
            x = i * SQUARE_SIZE + SQUARE_SIZE//2 - surf.get_width()//2
            y = BOARD_SIZE - COORDINATES['size'] - COORDINATES['margin']
            self.coords_surface.blit(surf, (x, y))
            
            # Rank coordinates (1-8)
            text = str(8 - i)
            surf = self.coord_font.render(text, True, COORDINATES['color'])
            x = COORDINATES['margin']
            y = i * SQUARE_SIZE + SQUARE_SIZE//2 - surf.get_height()//2
            self.coords_surface.blit(surf, (x, y))

    def square_to_pixel(self, square):
        """Chuyển đổi tọa độ ô cờ sang pixel"""
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = rank * SQUARE_SIZE
        return x, y

    def pixel_to_square(self, pos):
        """Chuyển đổi tọa độ pixel sang ô cờ"""
        x, y = pos
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None

    def draw_board(self):
        """Vẽ bàn cờ với các hiệu ứng đẹp"""
        # Vẽ border cho bàn cờ
        border_rect = pygame.Rect(-2, -2, BOARD_SIZE + 4, BOARD_SIZE + 4)
        pygame.draw.rect(self.board_surface, COLORS['sidebar_border'], border_rect)
        
        # Vẽ các ô cờ
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)
                x = file * SQUARE_SIZE
                y = rank * SQUARE_SIZE
                
                # Màu ô cờ
                is_light = (file + rank) % 2 == 0
                color = COLORS['light_square'] if is_light else COLORS['dark_square']
                
                # Vẽ ô cờ với góc bo tròn nhẹ
                pygame.draw.rect(self.board_surface, color, 
                               (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Vẽ hiệu ứng hover với gradient
                if square == self.hover_square:
                    hover_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    for i in range(SQUARE_SIZE):
                        alpha = int(40 * (1 - i/SQUARE_SIZE))
                        pygame.draw.rect(hover_surface, (*COLORS['hover'][:3], alpha),
                                      (0, i, SQUARE_SIZE, 1))
                    self.board_surface.blit(hover_surface, (x, y))
                    
                # Vẽ viền mảnh cho từng ô
                pygame.draw.rect(self.board_surface, (*COLORS['sidebar_border'], 30),
                               (x, y, SQUARE_SIZE, SQUARE_SIZE), 1)

    def draw_piece(self, piece, square):
        """Vẽ quân cờ với hiệu ứng đẹp"""
        if piece is None:
            return
            
        x, y = self.square_to_pixel(square)
        piece_symbol = self.unicode_pieces[(piece.piece_type, piece.color)]
        
        # Kiểm tra animation
        if square in self.animation.fade_animations:
            anim = self.animation.fade_animations[square]
            progress = (pygame.time.get_ticks() - anim['start_time']) / anim['duration']
            progress = min(1.0, progress)
            
            # Tính toán vị trí với animation
            start_x, start_y = self.square_to_pixel(anim['from_square'])
            end_x, end_y = self.square_to_pixel(anim['to_square'])
            
            # Áp dụng easing function
            t = self._ease_out_quad(progress)
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            
            # Thêm hiệu ứng bounce nhẹ
            if progress > 0.8:
                bounce = math.sin((progress - 0.8) * 5 * math.pi) * 3
                y += bounce
        
        # Vẽ bóng đổ với độ trong suốt
        for offset in range(3):
            alpha = 100 - offset * 30
            shadow = self.piece_font.render(piece_symbol, True, (*COLORS['shadow'][:3], alpha))
            self.board_surface.blit(shadow, (x + offset, y + offset))
        
        # Calculate scale based on hover and animation
        scale = 1.0
        if square == self.hover_square:
            scale = ANIMATION['hover_scale']
        if square in self.animation.fade_animations:
            anim_progress = (pygame.time.get_ticks() - self.animation.fade_animations[square]['start_time']) / self.animation.fade_animations[square]['duration']
            if anim_progress < 0.5:  # Scale up slightly during first half of move
                scale *= 1.1
                
        # Calculate piece size and position with scale
        piece_size = int(SQUARE_SIZE * scale)
        piece_surface = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
        
        # Piece color based on side
        color = COLORS['piece_white'] if piece.color else COLORS['piece_black']
        
        # Draw piece outline for white pieces
        if piece.color:
            outline_size = 2
            outline = self.piece_font.render(piece_symbol, True, (0, 0, 0, 100))
            outline = pygame.transform.scale(outline, (piece_size, piece_size))
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,-1), (0,1), (-1,0), (1,0)]:
                piece_surface.blit(outline, (outline_size * dx, outline_size * dy))
        
        # Draw main piece
        main_piece = self.piece_font.render(piece_symbol, True, color)
        main_piece = pygame.transform.scale(main_piece, (piece_size, piece_size))
        
        # Center the piece in its surface
        x_offset = (piece_size - SQUARE_SIZE) // 2
        y_offset = (piece_size - SQUARE_SIZE) // 2
        
        piece_surface.blit(main_piece, (0, 0))
        self.board_surface.blit(piece_surface, (x - x_offset, y - y_offset))

    def draw_highlights(self, board):
        """Vẽ các hiệu ứng highlight"""
        # Highlight nước đi cuối
        if self.last_move:
            for square in (self.last_move.from_square, self.last_move.to_square):
                x, y = self.square_to_pixel(square)
                highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                highlight_surface.fill(COLORS['last_move'])
                self.board_surface.blit(highlight_surface, (x, y))
        
        # Highlight ô được chọn
        if self.selected_square is not None:
            x, y = self.square_to_pixel(self.selected_square)
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(COLORS['highlight'])
            self.board_surface.blit(highlight_surface, (x, y))
        
        # Highlight các nước đi hợp lệ
        for move in self.valid_moves:
            x, y = self.square_to_pixel(move.to_square)
            move_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            
            # Vẽ dấu chấm cho nước đi hợp lệ
            pygame.draw.circle(
                move_surface, COLORS['valid_move'],
                (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//6
            )
            self.board_surface.blit(move_surface, (x, y))
        
        # Highlight tình huống chiếu
        if board.is_check():
            king_square = board.king(board.turn)
            if king_square is not None:
                x, y = self.square_to_pixel(king_square)
                check_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                check_surface.fill(COLORS['check'])
                self.board_surface.blit(check_surface, (x, y))

    def draw(self, board):
        """Vẽ toàn bộ bàn cờ"""
        # Vẽ bàn cờ cơ bản
        self.draw_board()
        
        # Vẽ các hiệu ứng highlight
        self.draw_highlights(board)
        
        # Vẽ quân cờ
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                self.draw_piece(piece, square)
        
        # Vẽ tọa độ
        self.board_surface.blit(self.coords_surface, (0, 0))
        
        # Vẽ lên màn hình chính
        self.screen.blit(
            self.board_surface,
            (BOARD_OFFSET_X, BOARD_OFFSET_Y)
        )

    def update_hover(self, pos):
        """Cập nhật ô đang hover"""
        x = pos[0] - BOARD_OFFSET_X
        y = pos[1] - BOARD_OFFSET_Y
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            self.hover_square = self.pixel_to_square((x, y))
        else:
            self.hover_square = None

    def handle_click(self, pos, board):
        """Xử lý click chuột"""
        x = pos[0] - BOARD_OFFSET_X
        y = pos[1] - BOARD_OFFSET_Y
        
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            square = self.pixel_to_square((x, y))
            return square
        return None
    
    def reset(self):
        """Reset trạng thái của bàn cờ"""
        self.selected_square = None
        self.hover_square = None
        self.valid_moves = []
        self.last_move = None
        self.animation.current_animation = None
        self.animation.fade_animations = []
        
    def update_animations(self, current_time):
        """Cập nhật trạng thái animation"""
        self.animation.update(current_time)
        
    def start_move_animation(self, move):
        """Bắt đầu animation cho một nước đi"""
        self.animation.start_move_animation(move, pygame.time.get_ticks())
        
    def is_animating(self):
        """Kiểm tra xem có animation đang chạy không"""
        return self.animation.current_animation is not None or len(self.animation.fade_animations) > 0