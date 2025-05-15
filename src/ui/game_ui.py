import pygame
import chess
from .board_ui import BoardUI
from ..agents.minimax_agent import MinimaxAgent
from ..agents.random_agent import RandomAgent

class GameUI:
    def __init__(self, width=1000, height=800, agent_type="minimax2", player_color="white"):
        """Khởi tạo giao diện trò chơi.
        
        Args:
            width: Chiều rộng cửa sổ
            height: Chiều cao cửa sổ
            agent_type: Loại agent (random, minimax4, minimax2)
            player_color: Màu của người chơi (white hoặc black)
        """
        self.board_ui = BoardUI(width - 200, height)  # Để chừa chỗ cho sidebar
        self.width = width
        self.height = height
        self.screen = self.board_ui.screen
        
        # Khởi tạo game
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.animating = False
        self.animation_data = None  # dict: piece, from_square, to_square, start_time, duration
        
        # Thời gian mỗi bên (giây)
        self.time_white = 5 * 60
        self.time_black = 5 * 60
        self.last_tick = pygame.time.get_ticks()
        self.active_color = chess.WHITE
        
        # Chế độ agent vs agent
        if agent_type == "agent_vs_agent":
            self.agent_white = MinimaxAgent(chess.WHITE, max_depth=2)
            self.agent_black = MinimaxAgent(chess.BLACK, max_depth=4)
            self.agent_vs_agent = True
        else:
            self.agent_vs_agent = False
            if agent_type == "random":
                self.agent = RandomAgent(chess.BLACK if player_color == "white" else chess.WHITE)
            elif agent_type.startswith("minimax"):
                try:
                    depth = int(agent_type.replace("minimax", ""))
                except:
                    depth = 2
                self.agent = MinimaxAgent(chess.BLACK if player_color == "white" else chess.WHITE, max_depth=depth)
            else:
                self.agent = MinimaxAgent(chess.BLACK if player_color == "white" else chess.WHITE, max_depth=2)
        
        # Font cho text
        self.font = pygame.font.Font(None, 36)
        
        # Undo/Redo
        self.redo_stack = []
        self.popup_message = None
        
        # Người chơi
        self.player_color = chess.WHITE if player_color == "white" else chess.BLACK
        
    def draw_sidebar(self):
        """Vẽ sidebar với thông tin trò chơi."""
        sidebar_x = self.width - 200
        pygame.draw.rect(self.screen, (200, 200, 200), (sidebar_x, 0, 200, self.height))
        
        # Hiển thị lượt đi
        turn_text = "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
        text = self.font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(text, (sidebar_x + 10, 10))
        
        # Hiển thị trạng thái game
        if self.board.is_checkmate():
            status = "Checkmate!"
        elif self.board.is_stalemate():
            status = "Stalemate"
        elif self.board.is_check():
            status = "Check!"
        else:
            status = "Game in progress"
            
        text = self.font.render(status, True, (0, 0, 0))
        self.screen.blit(text, (sidebar_x + 10, 50))
        
        # Hiển thị đồng hồ đếm thời gian
        font_small = pygame.font.Font(None, 28)
        time_white_str = f"White: {int(self.time_white // 60)}:{int(self.time_white % 60):02d}"
        time_black_str = f"Black: {int(self.time_black // 60)}:{int(self.time_black % 60):02d}"
        text = font_small.render(time_white_str, True, (0, 0, 0))
        self.screen.blit(text, (sidebar_x + 10, 80))
        text = font_small.render(time_black_str, True, (0, 0, 0))
        self.screen.blit(text, (sidebar_x + 10, 110))
        
        # Nút New Game
        pygame.draw.rect(self.screen, (100, 100, 100), (sidebar_x + 10, 140, 180, 40))
        text = self.font.render("New Game", True, (255, 255, 255))
        self.screen.blit(text, (sidebar_x + 50, 150))
        
        # Nút Undo
        pygame.draw.rect(self.screen, (80, 180, 80), (sidebar_x + 10, 190, 85, 35))
        text = font_small.render("Undo", True, (255, 255, 255))
        self.screen.blit(text, (sidebar_x + 25, 195))
        
        # Nút Redo
        pygame.draw.rect(self.screen, (80, 80, 180), (sidebar_x + 105, 190, 85, 35))
        text = font_small.render("Redo", True, (255, 255, 255))
        self.screen.blit(text, (sidebar_x + 120, 195))
        
        # Hiển thị lịch sử nước đi
        move_list = list(self.board.move_stack)
        font_move = pygame.font.Font(None, 24)
        self.screen.blit(font_small.render("Move history:", True, (0,0,0)), (sidebar_x + 10, 240))
        tmp_board = chess.Board()
        for i, move in enumerate(move_list[-15:]):
            san = tmp_board.san(move)
            move_text = f"{i+1}. {san}"
            text = font_move.render(move_text, True, (0, 0, 0))
            self.screen.blit(text, (sidebar_x + 10, 270 + i * 22))
            tmp_board.push(move)
        
    def start_animation(self, move):
        piece = self.board.piece_at(move.from_square)
        self.animating = True
        self.animation_data = {
            "piece": piece,
            "from_square": move.from_square,
            "to_square": move.to_square,
            "move": move,  # Lưu move để push sau khi hoạt ảnh xong
            "start_time": pygame.time.get_ticks(),
            "duration": 300
        }

    def draw_popup(self, message):
        popup_rect = pygame.Rect(self.width//2 - 150, self.height//2 - 60, 300, 120)
        pygame.draw.rect(self.screen, (255, 255, 200), popup_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), popup_rect, 3)
        font_big = pygame.font.Font(None, 36)
        text = font_big.render(message, True, (0, 0, 0))
        text_rect = text.get_rect(center=popup_rect.center)
        self.screen.blit(text, text_rect)

    def handle_click(self, pos):
        """Xử lý sự kiện click chuột.
        
        Args:
            pos: Tuple (x, y) vị trí click
        """
        x, y = pos
        
        # Undo/Redo
        if x > self.width - 200 + 10 and x < self.width - 200 + 95 and y > 190 and y < 225:
            if len(self.board.move_stack) > 0:
                move = self.board.pop()
                self.redo_stack.append(move)
            return
        if x > self.width - 200 + 105 and x < self.width - 200 + 190 and y > 190 and y < 225:
            if len(self.redo_stack) > 0:
                move = self.redo_stack.pop()
                self.board.push(move)
            return
            
        # Kiểm tra click vào nút New Game
        if x > self.width - 200 + 10 and x < self.width - 200 + 190 and y > 140 and y < 180:
            self.new_game()
            return
            
        # Kiểm tra click vào bàn cờ
        if x < self.width - 200:
            col, row = self.board_ui.get_square_from_pos(pos)
            square = chess.square(col, 7 - row)
            
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.valid_moves = [move for move in self.board.legal_moves if move.from_square == square]
            else:
                # Tìm move hợp lệ từ self.valid_moves (bao gồm promotion và nước ăn)
                selected_move = None
                for m in self.valid_moves:
                    if m.to_square == square:
                        selected_move = m
                        break
                if selected_move:
                    if self.is_promotion_move(selected_move):
                        self.show_promotion_popup(selected_move)
                    else:
                        self.start_animation(selected_move)
                        self.redo_stack.clear()
                self.selected_square = None
                self.valid_moves = []
                
    def new_game(self):
        """Bắt đầu ván cờ mới."""
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        
    def is_promotion_move(self, move):
        piece = self.board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and chess.square_rank(move.to_square) == 7) or \
               (piece.color == chess.BLACK and chess.square_rank(move.to_square) == 0):
                return True
        return False

    def show_promotion_popup(self, move):
        # Hiển thị popup chọn quân phong tốt
        running = True
        options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        names = ["Hậu", "Xe", "Tượng", "Mã"]
        font = pygame.font.Font(None, 36)
        popup_rect = pygame.Rect(self.width//2 - 150, self.height//2 - 60, 300, 120)
        while running:
            self.board_ui.draw_board(self.board)
            self.draw_sidebar()
            pygame.draw.rect(self.screen, (255, 255, 200), popup_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), popup_rect, 3)
            for i, name in enumerate(names):
                btn_rect = pygame.Rect(self.width//2 - 130 + i*70, self.height//2, 60, 40)
                pygame.draw.rect(self.screen, (100, 100, 200), btn_rect)
                text = font.render(name, True, (255,255,255))
                text_rect = text.get_rect(center=btn_rect.center)
                self.screen.blit(text, text_rect)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for i in range(4):
                        btn_rect = pygame.Rect(self.width//2 - 130 + i*70, self.height//2, 60, 40)
                        if btn_rect.collidepoint(event.pos):
                            # Tìm move hợp lệ với promotion đúng
                            for m in self.valid_moves:
                                if (m.from_square == move.from_square and
                                    m.to_square == move.to_square and
                                    m.promotion == options[i]):
                                    self.start_animation(m)
                                    self.redo_stack.clear()
                                    running = False
                                    break

    def run(self):
        """Chạy game loop."""
        running = True
        while running:
            # Cập nhật đồng hồ đếm thời gian
            now_tick = pygame.time.get_ticks()
            dt = (now_tick - self.last_tick) / 1000.0
            self.last_tick = now_tick
            if not self.board.is_game_over() and not self.animating:
                if self.board.turn == chess.WHITE:
                    self.time_white -= dt
                else:
                    self.time_black -= dt
                if self.time_white <= 0 or self.time_black <= 0:
                    self.popup_message = "Hết giờ!"
                    break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.agent_vs_agent:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and self.agent_vs_agent:
                    if not self.board.is_game_over():
                        current_board_copy = self.board.copy() # Pass a copy
                        if self.board.turn == chess.WHITE:
                            move = self.agent_white.get_move(current_board_copy)
                        else:
                            move = self.agent_black.get_move(current_board_copy)
                        # Agent tự động phong hậu nếu là promotion
                        if self.is_promotion_move(move):
                            move.promotion = chess.QUEEN
                        self.start_animation(move)
                    
            # Agent vs Agent tự động đi
            if self.agent_vs_agent and not self.board.is_game_over() and not self.animating:
                current_board_copy = self.board.copy() # Pass a copy
                if self.board.turn == chess.WHITE:
                    move = self.agent_white.get_move(current_board_copy)
                else:
                    move = self.agent_black.get_move(current_board_copy)
                # Agent tự động phong hậu nếu là promotion
                if self.is_promotion_move(move):
                    move.promotion = chess.QUEEN
                self.start_animation(move)
                self.redo_stack.clear()
            # Agent đi nước tiếp theo (người vs máy)
            if not self.agent_vs_agent and not self.board.is_game_over() and not self.animating and self.board.turn != self.player_color:
                current_board_copy = self.board.copy() # Pass a copy
                move = self.agent.get_move(current_board_copy)
                if self.is_promotion_move(move):
                    move.promotion = chess.QUEEN
                self.start_animation(move)
                self.redo_stack.clear()
            
            # Hoạt ảnh di chuyển quân cờ
            if self.animating:
                now = pygame.time.get_ticks()
                anim = self.animation_data
                elapsed = now - anim["start_time"]
                t = min(1, elapsed / anim["duration"])
                from_x, from_y = self.board_ui.square_to_pixel(anim["from_square"])
                to_x, to_y = self.board_ui.square_to_pixel(anim["to_square"])
                cur_x = from_x + (to_x - from_x) * t
                cur_y = from_y + (to_y - from_y) * t
                self.board_ui.draw_board(self.board, skip_squares=[anim["from_square"], anim["to_square"]])
                self.board_ui.draw_piece_at(anim["piece"], cur_x, cur_y)
                self.draw_sidebar()
                self.board_ui.update()
                if t >= 1:
                    self.board.push(anim["move"])
                    self.animating = False
                    self.animation_data = None
                continue
            
            # Vẽ bàn cờ
            self.board_ui.draw_board(self.board)
            
            # Highlight ô được chọn và các nước đi hợp lệ
            if self.selected_square is not None:
                self.board_ui.highlight_square(self.selected_square)
                for move in self.valid_moves:
                    self.board_ui.highlight_square(move.to_square)
                    
            # Vẽ sidebar
            self.draw_sidebar()
            
            # Hiển thị popup nếu có
            if self.popup_message or self.board.is_game_over():
                msg = self.popup_message
                if not msg:
                    if self.board.is_checkmate():
                        msg = "Checkmate!"
                    elif self.board.is_stalemate():
                        msg = "Stalemate!"
                    elif self.board.is_check():
                        msg = "Check!"
                    else:
                        msg = "Game Over!"
                self.draw_popup(msg)
            
            # Cập nhật màn hình
            self.board_ui.update()
            
        self.board_ui.quit() 