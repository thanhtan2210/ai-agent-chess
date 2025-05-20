# filepath: d:\Bon Bon\project 1\git\ai-agent-chess\src\ui\game_ui.py
import pygame
import chess
import time
from ui.board_ui import BoardUI
from ui.config import *
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.alpha_beta_agent import AlphaBetaAgent
from agents.mcts_agent import MCTSAgent
from agents.deep_learning_agent import DeepLearningAgent

class GameUI:
    def __init__(self, agent_type=None, player_color=None):
        """Khởi tạo giao diện game.
        
        Args:
            agent_type: Loại agent (random, minimax, alphabeta, mcts, deep_learning)
            player_color: Màu quân của người chơi (chess.WHITE hoặc chess.BLACK)
        """
        pygame.init()
        self.width = WINDOW_SIZE[0]
        self.height = WINDOW_SIZE[1]
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(WINDOW_TITLE)
        
        # Khởi tạo font
        self.font = pygame.font.Font(None, FONT_SIZES['medium'])
        self.small_font = pygame.font.Font(None, FONT_SIZES['small'])
        
        # Khởi tạo bàn cờ
        self.board_ui = BoardUI(self.screen, self.width - SIDE_PANEL_WIDTH, self.height)
        self.board = chess.Board()
        
        # Khởi tạo agent
        self.agent_type = agent_type
        self.player_color = player_color
        self.agent = self._create_agent()
        
        # Khởi tạo các biến game
        self.selected_square = None
        self.valid_moves = []
        self.game_over = False
        self.winner = None
        self.move_history = []  # List of tuples (move, san)
        self.start_time = time.time()
        
        # Khởi tạo các nút
        self._create_buttons()
        
        # Khởi tạo menu cài đặt
        self.settings_menu = {
            'visible': False,
            'rect': pygame.Rect(
                self.width - SIDE_PANEL_WIDTH + 20,
                150,
                BUTTON_STYLES['default']['width'],
                200
            ),
            'options': {
                'show_move_history': True,
                'show_valid_moves': True,
                'show_time': True
            }
        }
        
    def _create_buttons(self):
        """Tạo các nút điều khiển."""
        self.buttons = {}
        button_y = 20
        
        # Nút New Game
        self.buttons['new_game'] = pygame.Rect(
            self.width - SIDE_PANEL_WIDTH + 20,
            button_y,
            BUTTON_STYLES['default']['width'],
            BUTTON_STYLES['default']['height']
        )
        button_y += BUTTON_STYLES['default']['height'] + 10
        
        # Nút Undo
        self.buttons['undo'] = pygame.Rect(
            self.width - SIDE_PANEL_WIDTH + 20,
            button_y,
            BUTTON_STYLES['default']['width'],
            BUTTON_STYLES['default']['height']
        )
        button_y += BUTTON_STYLES['default']['height'] + 10
        
        # Nút Settings
        self.buttons['settings'] = pygame.Rect(
            self.width - SIDE_PANEL_WIDTH + 20,
            button_y,
            BUTTON_STYLES['default']['width'],
            BUTTON_STYLES['default']['height']
        )
        
    def _create_agent(self):
        """Tạo agent dựa trên loại được chọn."""
        # Xác định màu của agent (ngược với màu của người chơi)
        agent_color = chess.BLACK if self.player_color == chess.WHITE else chess.WHITE
        
        if self.agent_type == 'random':
            return RandomAgent(agent_color)
        elif self.agent_type.startswith('minimax'):
            # Lấy độ sâu từ tên agent (ví dụ: minimax3 -> depth=3)
            depth = int(self.agent_type[7:]) if len(self.agent_type) > 7 else 3
            return MinimaxAgent(agent_color, max_depth=depth)
        elif self.agent_type == 'alphabeta':
            return AlphaBetaAgent(agent_color, depth=4)
        elif self.agent_type == 'mcts':
            return MCTSAgent(agent_color, max_iterations=100)
        elif self.agent_type == 'deep_learning':
            return DeepLearningAgent(agent_color)
        return None
        
    def make_agent_move(self):
        """Cho agent thực hiện nước đi."""
        if self.agent and not self.game_over:
            move = self.agent.get_move(self.board)
            if move:
                san = self.board.san(move)
                self.board.push(move)
                self.move_history.append((move, san))
                self._check_game_over()
                
    def _check_game_over(self):
        """Kiểm tra kết thúc game."""
        if self.board.is_game_over():
            self.game_over = True
            if self.board.is_checkmate():
                self.winner = chess.BLACK if self.board.turn == chess.WHITE else chess.WHITE
            elif self.board.is_stalemate():
                self.winner = None  # Hòa
                
    def handle_square_click(self, square):
        """Xử lý sự kiện click vào ô cờ.
        
        Args:
            square: Tuple (col, row) tọa độ ô cờ
        """
        if self.game_over or self.board.turn != self.player_color:
            return
            
        col, row = square
        chess_square = chess.square(col, 7 - row)
        
        # Nếu đã chọn ô
        if self.selected_square is not None:
            # Tạo move từ ô đã chọn đến ô mới
            move = chess.Move(self.selected_square, chess_square)
            
            # Kiểm tra nếu là nước đi hợp lệ
            if move in self.board.legal_moves:
                san = self.board.san(move)
                self.board.push(move)
                self.move_history.append((move, san))
                self._check_game_over()
                
                # Cho agent đi nước tiếp theo nếu chưa kết thúc game
                if not self.game_over and self.board.turn != self.player_color:
                    self.make_agent_move()
                
            # Reset trạng thái chọn
            self.selected_square = None
            self.valid_moves = []
            
        # Nếu chưa chọn ô và ô được click có quân cờ
        elif self.board.piece_at(chess_square):
            piece = self.board.piece_at(chess_square)
            # Chỉ cho phép chọn quân của người chơi
            if piece.color == self.player_color:
                self.selected_square = chess_square
                self.valid_moves = [move for move in self.board.legal_moves 
                                  if move.from_square == chess_square]
                
    def handle_button_click(self, pos):
        """Xử lý sự kiện click vào nút.
        
        Args:
            pos: Tuple (x, y) vị trí click
        """
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if name == 'new_game':
                    self.board = chess.Board()
                    self.selected_square = None
                    self.valid_moves = []
                    self.game_over = False
                    self.winner = None
                    self.move_history = []
                    self.start_time = time.time()
                elif name == 'undo':
                    if len(self.move_history) >= 2:  # Undo cả nước của người chơi và agent
                        self.board.pop()
                        self.board.pop()
                        self.move_history.pop()
                        self.move_history.pop()
                        self.game_over = False
                        self.winner = None
                elif name == 'settings':
                    self.settings_menu['visible'] = not self.settings_menu['visible']
                    
        # Xử lý click trong menu cài đặt
        if self.settings_menu['visible']:
            menu_rect = self.settings_menu['rect']
            if menu_rect.collidepoint(pos):
                # Tính toán vị trí click tương đối trong menu
                rel_y = pos[1] - menu_rect.y
                option_height = 40
                
                # Kiểm tra click vào từng option
                for i, (option, value) in enumerate(self.settings_menu['options'].items()):
                    option_rect = pygame.Rect(
                        menu_rect.x,
                        menu_rect.y + i * option_height,
                        menu_rect.width,
                        option_height
                    )
                    if option_rect.collidepoint(pos):
                        self.settings_menu['options'][option] = not value
                        
    def draw(self):
        """Vẽ giao diện game."""
        # Vẽ bàn cờ
        self.board_ui.draw_board(self.board)
        
        # Highlight ô đã chọn và các nước đi hợp lệ
        if self.selected_square is not None and self.settings_menu['options']['show_valid_moves']:
            self.board_ui.highlight_square(self.selected_square)
            for move in self.valid_moves:
                self.board_ui.highlight_square(move.to_square)
                
        # Vẽ các nút
        for name, rect in self.buttons.items():
            pygame.draw.rect(self.screen, BUTTON_STYLES['default']['bg_color'], rect)
            pygame.draw.rect(self.screen, BUTTON_STYLES['default']['border_color'], rect, 2)
            text = self.font.render(name.replace('_', ' ').title(), True, BUTTON_STYLES['default']['text_color'])
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
            
        # Vẽ menu cài đặt
        if self.settings_menu['visible']:
            menu_rect = self.settings_menu['rect']
            pygame.draw.rect(self.screen, PANEL['bg_color'], menu_rect)
            pygame.draw.rect(self.screen, PANEL['border_color'], menu_rect, 2)
            
            for i, (option, value) in enumerate(self.settings_menu['options'].items()):
                option_rect = pygame.Rect(
                    menu_rect.x,
                    menu_rect.y + i * 40,
                    menu_rect.width,
                    40
                )
                text = self.small_font.render(
                    f"{option.replace('_', ' ').title()}: {'On' if value else 'Off'}", 
                    True, 
                    PANEL['text_color']
                )
                text_rect = text.get_rect(midleft=(option_rect.x + 10, option_rect.centery))
                self.screen.blit(text, text_rect)
            
        # Vẽ lịch sử nước đi
        if self.settings_menu['options']['show_move_history']:
            history_y = 250
            history_text = "Move History:"
            text_surface = self.font.render(history_text, True, PANEL['text_color'])
            self.screen.blit(text_surface, (self.width - SIDE_PANEL_WIDTH + 20, history_y))
            
            for i, (move, san) in enumerate(self.move_history):
                move_text = f"{i+1}. {san}"
                text_surface = self.small_font.render(move_text, True, PANEL['text_color'])
                self.screen.blit(text_surface, (self.width - SIDE_PANEL_WIDTH + 20, history_y + 30 + i * 20))
            
        # Vẽ thông tin game
        if self.game_over:
            if self.winner is None:
                text = "Game Over - Stalemate"
            else:
                text = f"Game Over - {'White' if self.winner == chess.WHITE else 'Black'} wins"
            text_surface = self.font.render(text, True, (255, 0, 0))
            self.screen.blit(text_surface, (20, self.height - 40))
            
        # Vẽ thời gian
        if self.settings_menu['options']['show_time']:
            elapsed_time = int(time.time() - self.start_time)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            time_text = f"Time: {minutes:02d}:{seconds:02d}"
            time_surface = self.font.render(time_text, True, (0, 0, 0))
            self.screen.blit(time_surface, (self.width - SIDE_PANEL_WIDTH + 20, self.height - 40))
            
        pygame.display.flip()
        
    def run(self):
        """Chạy game loop."""
        running = True
        clock = pygame.time.Clock()
        
        # Nếu là agent vs agent, tự động chạy
        if self.agent_type and self.player_color is None:
            while running and not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        
                self.make_agent_move()
                self.draw()
                clock.tick(1)  # Giới hạn 1 FPS cho agent vs agent
                
        # Game loop chính cho người chơi
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if pos[0] < self.width - SIDE_PANEL_WIDTH:
                        square = self.board_ui.get_square_from_pos(pos)
                        self.handle_square_click(square)
                    else:
                        self.handle_button_click(pos)
                        
            self.draw()
            clock.tick(60)
            
        pygame.quit()