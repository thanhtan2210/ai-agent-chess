# filepath: d:\Bon Bon\project 1\git\ai-agent-chess\src\ui\game_ui.py
import os
import sys
import pygame
import chess
import time
import math
from .board_ui import BoardUI
from .config import *
from .utils import format_time, create_gradient_surface
from ..game.chess_game import ChessGame
from .setup_dialog import select_difficulty_and_color

def format_time(seconds):
    """Format time in seconds to mm:ss format"""
    if seconds is None:
        return "--:--"
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

class GameUI:
    def __init__(self, game, white_agent=None, black_agent=None):
        """Khởi tạo giao diện game.
        
        Args:
            game: ChessGame instance
            white_agent: Agent cho quân trắng
            black_agent: Agent cho quân đen
        """
        # Initialize pygame and display
        pygame.init()
        if not pygame.get_init():
            raise RuntimeError("Failed to initialize pygame")
        
        pygame.display.set_caption(WINDOW_TITLE)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        if not self.screen:
            raise RuntimeError("Failed to create display surface")
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        
        # Initialize fonts
        self._init_fonts()
        
        # Game settings
        self.settings = {
            'sound_enabled': True,
            'animations_enabled': True,
            'coordinates_enabled': True,
            'highlights_enabled': True,
            'move_confirmation': False
        }
        
        # Initialize undo/redo stacks
        self.move_history = []
        self.redo_stack = []
        
        # Player info: cập nhật đúng vị trí AI/Player theo màu
        if white_agent is None:
            # Người chơi là trắng
            self.player_info = {
                'white': {'name': 'Player'},
                'black': {'name': 'AI'}
            }
        elif black_agent is None:
            # Người chơi là đen
            self.player_info = {
                'white': {'name': 'AI'},
                'black': {'name': 'Player'}
            }
        else:
            self.player_info = {
                'white': {'name': 'Player'},
                'black': {'name': 'AI'}
            }
        # Khởi tạo captured_pieces để tránh lỗi
        self.captured_pieces = {'white': [], 'black': []}
        
        # Initialize game state
        self.game = game
        self.white_agent = white_agent
        self.black_agent = black_agent
        self.selected_square = None
        self.valid_moves = []
        self.last_move = None
        self.game_over = False
        self.show_promotion_dialog = False
        self.show_settings = False
        
        # Initialize UI components
        self.board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        self.sidebar_surface = pygame.Surface((SIDE_PANEL_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        self.board_ui = BoardUI(self.screen)
        self.buttons = self.create_buttons()
    
    def _init_fonts(self):
        """Khởi tạo các font chữ"""
        self.font = pygame.font.SysFont(FONTS['default'], FONT_SIZES['medium'])
        self.header_font = pygame.font.SysFont(FONTS['default'], FONT_SIZES['large'])
        self.title_font = pygame.font.SysFont(FONTS['default'], FONT_SIZES['title'])
        self.small_font = pygame.font.SysFont(FONTS['default'], FONT_SIZES['small'])

    def create_buttons(self):
        """Tạo các nút điều khiển game"""
        button_width = BUTTON_STYLE['width']
        button_height = BUTTON_STYLE['height']
        start_y = 120  # Đặt nút New Game thấp hơn
        button_gap = 20
        buttons = {}
        button_configs = [
            ('new_game', 'New Game', self.new_game, start_y, button_width, button_height),
            ('undo', 'Undo', self.undo_move, start_y + button_height + button_gap, button_width, button_height),
            ('resign', 'Resign', self.resign_game, start_y + 2 * (button_height + button_gap), button_width, button_height)
        ]
        for id, text, action, y, w, h in button_configs:
            buttons[id] = {
                'rect': pygame.Rect(self.width - SIDE_PANEL_WIDTH + 20, y, w, h),
                'text': text,
                'action': action,
                'hover': False
            }
        return buttons
        
    def draw_button(self, button, surface=None):
        """Vẽ một nút với hiệu ứng hiện đại giống chess.com"""
        if surface is None:
            surface = self.screen
        
        rect = button['rect']
        text = button['text']
        hover = button.get('hover', False)
        
        # Draw shadow
        shadow_rect = rect.copy()
        shadow_rect.y += UI_STYLE['shadow_offset']
        pygame.draw.rect(surface, COLORS['shadow'], shadow_rect, 
                        border_radius=UI_STYLE['button_radius'])
        
        # Nút New Game nổi bật hơn
        if text == 'New Game':
            color = (102, 204, 102) if not hover else (120, 220, 120)  # Xanh lá sáng
            font = pygame.font.SysFont(FONTS['default'], FONT_SIZES['large'], bold=True)
        else:
            color = COLORS['button_hover'] if hover else COLORS['button_bg']
            font = self.font
        
        pygame.draw.rect(surface, color, rect, 
                        border_radius=UI_STYLE['button_radius'])
        
        # Draw text
        text_surf = font.render(text, True, COLORS['text_primary'])
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)
        
    def draw(self):
        """Vẽ toàn bộ giao diện game giống chess.com"""
        # Clear screen
        self.screen.fill(COLORS['background'])
        
        # Draw chess board
        self.board_ui.draw(self.game.board)
        
        # Draw sidebar elements
        self.draw_sidebar()
        
        # Draw player info panels
        self.draw_player_info(is_top=True)  # Black player
        self.draw_player_info(is_top=False)  # White player
        
        # Draw move list
        self.draw_move_list()
        
        # Draw captured pieces
        self.draw_captured_pieces()
        
        # Draw promotion dialog if active
        if self.show_promotion_dialog:
            self.draw_promotion_dialog()
        
        # Draw game over screen if game is over
        if self.game_over:
            self.draw_game_over()
        
        # Draw settings panel if open
        if self.show_settings:
            self.draw_settings()
        
        # Update display
        pygame.display.flip()

    def draw_sidebar(self):
        """Vẽ thanh bên với gradient và đổ bóng hiện đại giống chess.com"""
        sidebar_rect = pygame.Rect(
            WINDOW_WIDTH - SIDE_PANEL_WIDTH,
            0,
            SIDE_PANEL_WIDTH,
            WINDOW_HEIGHT
        )
        # Nền sidebar tối hiện đại
        pygame.draw.rect(self.screen, COLORS['sidebar_bg'], sidebar_rect)
        # Gradient nhẹ phía trên
        gradient = pygame.Surface((SIDE_PANEL_WIDTH, 120), pygame.SRCALPHA)
        for y in range(120):
            alpha = int(80 * (1 - y / 120))
            pygame.draw.line(gradient, (0,0,0,alpha), (0, y), (SIDE_PANEL_WIDTH, y))
        self.screen.blit(gradient, (WINDOW_WIDTH - SIDE_PANEL_WIDTH, 0))
        # Tạo lại buttons
        self.buttons = self.create_buttons()
        for button in self.buttons.values():
            self.draw_button(button)

    def draw_settings(self):
        """Vẽ panel cài đặt giống chess.com"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Settings panel
        panel_width = 400
        panel_height = 500
        panel_x = (WINDOW_WIDTH - panel_width) // 2
        panel_y = (WINDOW_HEIGHT - panel_height) // 2
        
        # Draw panel background with shadow
        shadow_rect = pygame.Rect(
            panel_x + UI_STYLE['shadow_offset'],
            panel_y + UI_STYLE['shadow_offset'],
            panel_width,
            panel_height
        )
        pygame.draw.rect(self.screen, COLORS['shadow'], shadow_rect,
                        border_radius=UI_STYLE['panel_radius'])
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, COLORS['sidebar_bg'], panel_rect,
                        border_radius=UI_STYLE['panel_radius'])
        
        # Draw title
        title = self.header_font.render("Settings", True, COLORS['text_primary'])
        title_rect = title.get_rect(centerx=panel_rect.centerx, y=panel_y + 20)
        self.screen.blit(title, title_rect)
        
        # Draw settings options
        y = panel_y + 80
        for setting, enabled in self.settings.items():
            # Setting name
            name = setting.replace('_', ' ').title()
            text = self.font.render(name, True, COLORS['text_primary'])
            self.screen.blit(text, (panel_x + 30, y))
            
            # Toggle button
            toggle_rect = pygame.Rect(panel_x + 300, y, 50, 26)
            toggle_color = COLORS['button_active'] if enabled else COLORS['sidebar_border']
            pygame.draw.rect(self.screen, toggle_color, toggle_rect,
                           border_radius=13)
            
            # Toggle switch
            switch_x = toggle_rect.right - 23 if enabled else toggle_rect.x + 3
            switch_rect = pygame.Rect(switch_x, y + 3, 20, 20)
            pygame.draw.rect(self.screen, COLORS['text_primary'], switch_rect,
                           border_radius=10)
            
            y += 50
    
    def draw_player_info(self, is_top=True):
        """Vẽ thông tin người chơi giống chess.com"""
        player = 'black' if is_top else 'white'
        y = 20 if is_top else WINDOW_HEIGHT - 80
        
        # Background panel
        panel_rect = pygame.Rect(
            WINDOW_WIDTH - SIDE_PANEL_WIDTH + 10,
            y,
            SIDE_PANEL_WIDTH - 20,
            50
        )
        pygame.draw.rect(self.screen, COLORS['sidebar_bg'], panel_rect,
                        border_radius=UI_STYLE['panel_radius'])
        
        # Player name
        name = self.player_info[player]['name']
        name_surf = self.font.render(name, True, COLORS['text_primary'])
        self.screen.blit(name_surf, (panel_rect.x + 10, y + 16))
        
        # Time left (if using chess clock)
        if hasattr(self, 'clock'):
            time_left = self.clock.get_time(player)
            time_color = COLORS['time_warning'] if time_left < 30 else COLORS['text_primary']
            time_surf = self.font.render(format_time(time_left), True, time_color)
            self.screen.blit(time_surf, (panel_rect.right - 70, y + 15))

    def draw_move_list(self):
        """Vẽ danh sách nước đi giống chess.com"""
        # Panel background
        # Đặt panel move list thấp xuống dưới các nút
        move_panel_y = 60 + 4 * (BUTTON_STYLE['height'] + 20) + 20  # 60 là start_y, 4 nút, 20 là button_gap, +20 padding
        panel_rect = pygame.Rect(
            WINDOW_WIDTH - SIDE_PANEL_WIDTH + 10,
            move_panel_y,
            SIDE_PANEL_WIDTH - 20,
            300
        )
        pygame.draw.rect(self.screen, COLORS['sidebar_bg'], panel_rect,
                        border_radius=UI_STYLE['panel_radius'])
        # Title
        title = self.header_font.render("Moves", True, COLORS['text_primary'])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))
        # Move list
        y = panel_rect.y + 50
        for i in range(0, len(self.move_history), 2):
            # Move number
            num_surf = self.small_font.render(str(f"{i//2 + 1}."), True, COLORS['text_secondary'])
            self.screen.blit(num_surf, (panel_rect.x + 10, y))
            # White's move
            if i < len(self.move_history):
                white_surf = self.small_font.render(str(self.move_history[i]), True, COLORS['text_primary'])
                self.screen.blit(white_surf, (panel_rect.x + 40, y))
            # Black's move
            if i + 1 < len(self.move_history):
                black_surf = self.small_font.render(str(self.move_history[i + 1]), True, COLORS['text_primary'])
                self.screen.blit(black_surf, (panel_rect.x + 100, y))
            y += 25
            if y > panel_rect.bottom - 30:
                break

    def draw_captured_pieces(self):
        """Vẽ quân cờ bị bắt giống chess.com"""
        # White's captured pieces
        y = WINDOW_HEIGHT - 120
        self.draw_captured_row(self.captured_pieces['white'], y, True)
        
        # Black's captured pieces
        y = 100
        self.draw_captured_row(self.captured_pieces['black'], y, False)

    def draw_captured_row(self, pieces, y, is_white):
        """Vẽ một hàng quân cờ bị bắt"""
        if not pieces:
            return
            
        x = WINDOW_WIDTH - SIDE_PANEL_WIDTH + 20
        piece_size = FONT_SIZES['medium']
        
        for symbol in pieces:
            color = COLORS['piece_white'] if is_white else COLORS['piece_black']
            piece_surf = self.font.render(symbol, True, color)
            
            # Draw shadow
            shadow_surf = self.font.render(symbol, True, COLORS['shadow'])
            self.screen.blit(shadow_surf, (x + 1, y + 1))
            
            # Draw piece
            self.screen.blit(piece_surf, (x, y))
            x += piece_size * 0.8  # Overlap pieces slightly
    
    def play_sound(self, sound_name):
        pass

    def new_game(self):
        """Start a new game with difficulty/color/AI vs AI/agent selection"""
        depth, color, ai_vs_ai, agent_white_name, agent_black_name = select_difficulty_and_color()
        from src.agents import MinimaxAgent, AlphaBetaAgent, MCTSAgent, DeepLearningAgent, RandomAgent
        from src.game.chess_game import ChessGame
        game = ChessGame()
        def get_agent(agent_name, color, depth):
            if agent_name == "MinimaxAgent":
                return MinimaxAgent(color, max_depth=depth)
            elif agent_name == "AlphaBetaAgent":
                return AlphaBetaAgent(color, depth=depth)
            elif agent_name == "MCTSAgent":
                return MCTSAgent(color, max_time=5.0)
            elif agent_name == "DeepLearningAgent":
                return DeepLearningAgent(color)
            elif agent_name == "RandomAgent":
                return RandomAgent(color)
            else:
                return None
        if ai_vs_ai:
            white_agent = get_agent(agent_white_name, chess.WHITE, depth)
            black_agent = get_agent(agent_black_name, chess.BLACK, depth)
        elif color == 0:
            white_agent = None
            black_agent = get_agent(agent_black_name, chess.BLACK, depth)
        else:
            white_agent = get_agent(agent_white_name, chess.WHITE, depth)
            black_agent = None
        self.__init__(game, white_agent, black_agent)
    
    def resign_game(self):
        """Người chơi từ bỏ game"""
        self.game_over = True
        # Có thể thêm hiệu ứng hoặc thông báo ở đây
    
    def toggle_settings(self):
        """Chuyển đổi hiển thị menu cài đặt"""
        self.show_settings = not self.show_settings
    
    def make_agent_move(self):
        """Cho agent thực hiện nước đi."""
        if self.white_agent and not self.game_over:
            move = self.white_agent.get_move(self.game.board)
            if move:
                san = self.game.board.san(move)
                self.game.board.push(move)
                self.move_history.append(san)
                self._check_game_over()
                
    def _check_game_over(self):
        """Kiểm tra kết thúc game."""
        if self.game.board.is_game_over():
            self.game_over = True
            
    def handle_square_click(self, square):
        """Xử lý sự kiện click vào ô cờ.
        
        Args:
            square: Tuple (col, row) tọa độ ô cờ
        """
        if self.game_over or self.game.board.turn != chess.WHITE:
            return
            
        col, row = square
        chess_square = chess.square(col, 7 - row)
        
        # Nếu đã chọn ô
        if self.selected_square is not None:
            # Tạo move từ ô đã chọn đến ô mới
            move = chess.Move(self.selected_square, chess_square)
            
            # Kiểm tra nếu là nước đi hợp lệ
            if move in self.game.board.legal_moves:
                san = self.game.board.san(move)
                self.game.board.push(move)
                self.move_history.append(san)
                self._check_game_over()
                
                # Cho agent đi nước tiếp theo nếu chưa kết thúc game
                if not self.game_over and self.game.board.turn != chess.WHITE:
                    self.make_agent_move()
                
            # Reset trạng thái chọn
            self.selected_square = None
            self.valid_moves = []
            
        # Nếu chưa chọn ô và ô được click có quân cờ
        elif self.game.board.piece_at(chess_square):
            piece = self.game.board.piece_at(chess_square)
            # Chỉ cho phép chọn quân của người chơi
            if piece.color == chess.WHITE:
                self.selected_square = chess_square
                self.valid_moves = [move for move in self.game.board.legal_moves 
                                  if move.from_square == chess_square]
                
    def handle_button_click(self, pos):
        """Xử lý sự kiện click vào nút.
        
        Args:
            pos: Tuple (x, y) vị trí click
        """
        for name, button in self.buttons.items():
            rect = button['rect']
            if rect.collidepoint(pos):
                button['action']()
                break
    
    def handle_promotion_choice(self, piece_type):
        """Xử lý chọn quân cờ khi phong cấp"""
        if self.promotion_move:
            move = chess.Move.from_uci(
                self.promotion_move.from_square + 
                self.promotion_move.to_square + 
                piece_type
            )
            if self.game.is_valid_move(move):
                self.make_move(move)
                self.show_promotion_dialog = False
                self.promotion_move = None
    
    def draw_game_over(self):
        """Vẽ màn hình kết thúc game"""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Draw game over message
        message = "Game Over"
        result = "Draw!" if self.game.board.is_stalemate() else \
                "White wins!" if not self.game.board.turn else "Black wins!"
                
        # Draw messages
        game_over_text = self.title_font.render(message, True, COLORS['text_primary'])
        result_text = self.header_font.render(result, True, COLORS['text_primary'])
        
        # Position text
        game_over_rect = game_over_text.get_rect(centerx=self.width//2, centery=self.height//2 - 40)
        result_rect = result_text.get_rect(centerx=self.width//2, centery=self.height//2 + 40)
        
        # Draw text with shadow
        shadow_offset = 2
        shadow_color = COLORS['shadow']
        
        # Draw shadows
        game_over_shadow = self.title_font.render(message, True, shadow_color)
        result_shadow = self.header_font.render(result, True, shadow_color)
        
        self.screen.blit(game_over_shadow, (game_over_rect.x + shadow_offset, 
                                          game_over_rect.y + shadow_offset))
        self.screen.blit(result_shadow, (result_rect.x + shadow_offset,
                                       result_rect.y + shadow_offset))
        
        # Draw text
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(result_text, result_rect)
        
        # Draw restart button
        button_width = 200
        button_height = 50
        button_rect = pygame.Rect((self.width - button_width)//2,
                                self.height//2 + 120,
                                button_width, button_height)
        
        # Draw button shadow
        shadow_rect = button_rect.copy()
        shadow_rect.y += UI_STYLE['shadow_offset']
        pygame.draw.rect(self.screen, COLORS['shadow'], shadow_rect,
                        border_radius=UI_STYLE['button_radius'])
        
        # Draw button
        pygame.draw.rect(self.screen, COLORS['button_bg'], button_rect,
                        border_radius=UI_STYLE['button_radius'])
        
        # Draw button text
        button_text = self.font.render("New Game", True, COLORS['text_primary'])
        text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, text_rect)
        
        # Lưu lại button_rect để kiểm tra click
        self._game_over_button_rect = button_rect
        return button_rect

    def handle_promotion(self, piece_type):
        """Xử lý việc chọn quân cờ khi phong cấp"""
        if self.promotion_move:
            move = chess.Move(
                from_square=self.promotion_move.from_square,
                to_square=self.promotion_move.to_square,
                promotion=piece_type
            )
            if move in self.game.board.legal_moves:
                self.make_move(move)
                self.show_promotion_dialog = False
                self.promotion_move = None
                
    def draw_promotion_dialog(self):
        """Vẽ hộp thoại chọn quân cờ khi phong cấp"""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        # Dialog dimensions
        dialog_width = 400
        dialog_height = 200
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2
        # Draw dialog shadow
        shadow_rect = pygame.Rect(dialog_x + UI_STYLE['shadow_offset'],
                                dialog_y + UI_STYLE['shadow_offset'],
                                dialog_width, dialog_height)
        pygame.draw.rect(self.screen, COLORS['shadow'], shadow_rect,
                        border_radius=UI_STYLE['panel_radius'])
        # Draw dialog background
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        pygame.draw.rect(self.screen, COLORS['sidebar_bg'], dialog_rect,
                        border_radius=UI_STYLE['panel_radius'])
        # Draw title
        title = self.header_font.render("Choose Promotion", True, COLORS['text_primary'])
        title_rect = title.get_rect(centerx=self.width//2, y=dialog_y + 20)
        self.screen.blit(title, title_rect)
        # Draw piece options
        pieces = [(chess.QUEEN, 'q'), (chess.ROOK, 'r'), (chess.BISHOP, 'b'), (chess.KNIGHT, 'n')]
        piece_size = 60
        total_width = len(pieces) * (piece_size + 20) - 20
        start_x = (self.width - total_width) // 2
        piece_rects = []
        color = self.game.board.turn
        # Dùng font DejaVu Sans cho biểu tượng quân cờ
        font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
        piece_font = pygame.font.Font(font_path, piece_size - 8)
        for i, (piece_type, symbol) in enumerate(pieces):
            piece_x = start_x + i * (piece_size + 20)
            piece_y = dialog_y + 80
            piece_rect = pygame.Rect(piece_x, piece_y, piece_size, piece_size)
            pygame.draw.rect(self.screen, COLORS['light_square'], piece_rect,
                           border_radius=UI_STYLE['button_radius'])
            # Lấy ký hiệu quân cờ đúng
            piece_symbol = self.board_ui.unicode_pieces[(piece_type, color)]
            piece_text = piece_font.render(piece_symbol, True, COLORS['text_primary'])
            text_rect = piece_text.get_rect(center=piece_rect.center)
            self.screen.blit(piece_text, text_rect)
            piece_rects.append((piece_rect, piece_type))
            # Draw hover effect
            if piece_rect.collidepoint(pygame.mouse.get_pos()):
                hover_surface = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
                hover_surface.fill(COLORS['hover'])
                self.screen.blit(hover_surface, piece_rect)
        return piece_rects  # Return the piece rects for click detection

    def handle_click(self, pos):
        # Nếu đang show promotion dialog, kiểm tra click vào các ô chọn quân
        if self.show_promotion_dialog:
            piece_rects = self.draw_promotion_dialog()
            for rect, piece_type in piece_rects:
                if rect.collidepoint(pos):
                    self.handle_promotion(piece_type)
                    return
        # Nếu game over, kiểm tra click vào nút New Game
        if self.game_over:
            if hasattr(self, '_game_over_button_rect') and self._game_over_button_rect.collidepoint(pos):
                self.new_game()
                self.game_over = False
                return
        # Check if click is on the board
        board_rect = pygame.Rect(BOARD_OFFSET_X, BOARD_OFFSET_Y,
                               BOARD_SIZE, BOARD_SIZE)
        if board_rect.collidepoint(pos):
            # Convert screen coordinates to board coordinates
            board_x = pos[0] - BOARD_OFFSET_X
            board_y = pos[1] - BOARD_OFFSET_Y
            square = self.board_ui.pixel_to_square((board_x, board_y))
            if square is not None:
                self.handle_board_click(square)
        # Check button clicks
        for button in self.buttons.values():
            if button['rect'].collidepoint(pos):
                button['action']()
                return

    def handle_board_click(self, square):
        """Xử lý click trên bàn cờ"""
        if self.game_over or self.show_promotion_dialog:
            return
            
        # Nếu là lượt của AI, không cho phép di chuyển
        current_agent = self.white_agent if self.game.board.turn else self.black_agent
        if current_agent is not None:
            return
            
        if self.selected_square is None:
            # Chọn quân cờ
            piece = self.game.board.piece_at(square)
            if piece and piece.color == self.game.board.turn:
                self.selected_square = square
                self.valid_moves = [
                    move for move in self.game.board.legal_moves
                    if move.from_square == square
                ]
                self.board_ui.selected_square = square
                self.board_ui.valid_moves = self.valid_moves
        else:
            # Thực hiện nước đi
            move = chess.Move(self.selected_square, square)
            
            # Kiểm tra phong cấp
            piece = self.game.board.piece_at(self.selected_square)
            if (piece and piece.piece_type == chess.PAWN and
                ((square >= 56 and self.game.board.turn) or 
                 (square <= 7 and not self.game.board.turn))):
                self.show_promotion_dialog = True
                self.promotion_move = move
            elif move in self.game.board.legal_moves:
                self.make_move(move)
            
            # Reset selection
            self.clear_selection()

    def clear_selection(self):
        """Xóa selection và các highlight"""
        self.selected_square = None
        self.valid_moves = []
        self.board_ui.selected_square = None
        self.board_ui.valid_moves = []
    
    def undo_move(self):
        """Hoàn tác nước đi cuối cùng"""
        if self.game_over or self.show_promotion_dialog:
            return
            
        # Hoàn tác nước đi của AI nếu có
        if (self.white_agent and self.game.board.turn == chess.BLACK) or \
           (self.black_agent and self.game.board.turn == chess.WHITE):
            if len(self.move_history) > 0:
                self.game.board.pop()
                self.move_history.pop()
                self.last_move = None if not self.move_history else self.move_history[-1]
                self.board_ui.reset()
        
        # Hoàn tác nước đi của người chơi
        if len(self.move_history) > 0:
            self.game.board.pop()
            self.move_history.pop()
            self.last_move = None if not self.move_history else self.move_history[-1]
            self.board_ui.reset()
            
            # Cập nhật lại trạng thái game
            self.game_over = False
            self.show_promotion_dialog = False
            self.clear_selection()
    
    def redo_move(self):
        """Làm lại nước đi đã hoàn tác"""
        if len(self.redo_stack) > 0:
            move = self.redo_stack.pop()
            self.game.make_move(move)
            self.move_history.append(move)
            self.last_move = move
            self.board_ui.reset()

    def make_move(self, move):
        """Thực hiện nước đi với animation và âm thanh"""
        if not move in self.game.board.legal_moves:
            return
            
        # Clear redo stack khi có nước đi mới
        self.redo_stack.clear()
        
        # Kiểm tra loại nước đi để phát âm thanh phù hợp
        is_capture = self.game.board.piece_at(move.to_square) is not None
        was_check = self.game.board.is_check()
        
        # Thực hiện nước đi
        self.game.make_move(move)
        self.move_history.append(move)
        self.last_move = move
        
        # Kiểm tra kết thúc game
        if self.game.is_game_over():
            self.game_over = True
    
    def update_animation(self):
        """Cập nhật trạng thái animation"""
        if self.animating_move:
            current_time = pygame.time.get_ticks()
            progress = (current_time - self.animation_start) / self.animation_duration
            
            if progress >= 1:
                self.animating_move = None
            else:
                # Cập nhật animation (sẽ được implement trong BoardUI)
                self.board_ui.update_animation(self.animating_move, progress)
    
    def update(self):
        """Cập nhật trạng thái game"""
        if self.game_over or self.show_promotion_dialog:
            return
            
        # Kiểm tra lượt của AI
        current_agent = self.white_agent if self.game.board.turn else self.black_agent
        if current_agent and not self.board_ui.is_animating():
            move = current_agent.get_move(self.game.board)
            if move:
                self.make_move(move)
    
    def run(self):
        """Chạy game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_hover(event.pos)
            
            # Update game state
            self.update()
            
            # Draw everything
            self.draw()
            
            # Update display
            pygame.display.flip()
            
            # Cap at 60 FPS
            clock.tick(60)
        
        pygame.quit()

    def handle_hover(self, pos):
        if hasattr(self, 'board_ui'):
            self.board_ui.update_hover(pos)