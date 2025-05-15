import pygame
import chess
import os

class BoardUI:
    def __init__(self, width=800, height=800):
        """Khởi tạo giao diện bàn cờ.
        
        Args:
            width: Chiều rộng cửa sổ
            height: Chiều cao cửa sổ
        """
        pygame.init()
        self.width = width
        self.height = height
        self.square_size = min(width, height) // 8
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chess Game")
        
        # Màu gỗ cho bàn cờ
        self.WHITE = (240, 217, 181)  # Gỗ nhạt
        self.BLACK = (181, 136, 99)   # Gỗ đậm
        self.HIGHLIGHT = (124, 252, 0, 128)  # Màu xanh lá nhạt cho highlight
        
        # Dùng font DejaVuSans.ttf với kích thước nhỏ hơn để quân cờ không bị tràn ô
        font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
        self.piece_font = pygame.font.Font(font_path, self.square_size - 4)

        self.unicode_pieces = {
            (chess.PAWN, chess.WHITE): "\u2659",
            (chess.KNIGHT, chess.WHITE): "\u2658",
            (chess.BISHOP, chess.WHITE): "\u2657",
            (chess.ROOK, chess.WHITE): "\u2656",
            (chess.QUEEN, chess.WHITE): "\u2655",
            (chess.KING, chess.WHITE): "\u2654",
            (chess.PAWN, chess.BLACK): "\u265F",
            (chess.KNIGHT, chess.BLACK): "\u265E",
            (chess.BISHOP, chess.BLACK): "\u265D",
            (chess.ROOK, chess.BLACK): "\u265C",
            (chess.QUEEN, chess.BLACK): "\u265B",
            (chess.KING, chess.BLACK): "\u265A",
        }
        
    def square_to_pixel(self, square):
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        x = col * self.square_size
        y = row * self.square_size
        return x, y

    def draw_piece_at(self, piece, x, y):
        uni = self.unicode_pieces.get((piece.piece_type, piece.color))
        if uni:
            # Đổi màu: quân trắng dùng màu đen, quân đen dùng màu trắng
            color = (0, 0, 0) if piece.color == chess.BLACK else (255, 255, 255)
            text = self.piece_font.render(uni, True, color)
            text_rect = text.get_rect(center=(x + self.square_size // 2, y + self.square_size // 2))
            self.screen.blit(text, text_rect)

    def draw_board(self, board, skip_squares=None):
        if skip_squares is None:
            skip_squares = []
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size
                color = self.WHITE if (row + col) % 2 == 0 else self.BLACK
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                square = chess.square(col, 7 - row)
                if square in skip_squares:
                    continue
                piece = board.piece_at(square)
                if piece:
                    uni = self.unicode_pieces.get((piece.piece_type, piece.color))
                    if uni:
                        # Đổi màu: quân trắng dùng màu đen, quân đen dùng màu trắng
                        color = (0, 0, 0) if piece.color == chess.BLACK else (255, 255, 255)
                        text = self.piece_font.render(uni, True, color)
                        text_rect = text.get_rect(center=(x + self.square_size // 2, y + self.square_size // 2))
                        self.screen.blit(text, text_rect)
                        
    def highlight_square(self, square, color=None):
        """Highlight một ô cờ.
        
        Args:
            square: chess.Square object hoặc tọa độ (x, y)
            color: Màu highlight (mặc định là màu xanh lá nhạt)
        """
        if isinstance(square, chess.Square):
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
        else:
            col, row = square
            
        x = col * self.square_size
        y = row * self.square_size
        
        if color is None:
            color = self.HIGHLIGHT
            
        s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        pygame.draw.rect(s, color, s.get_rect())
        self.screen.blit(s, (x, y))
        
    def get_square_from_pos(self, pos):
        """Chuyển đổi vị trí chuột thành tọa độ ô cờ.
        
        Args:
            pos: Tuple (x, y) vị trí chuột
            
        Returns:
            Tuple (col, row) tọa độ ô cờ
        """
        x, y = pos
        col = x // self.square_size
        row = y // self.square_size
        return col, row
        
    def update(self):
        """Cập nhật màn hình."""
        pygame.display.flip()
        
    def quit(self):
        """Đóng pygame."""
        pygame.quit() 