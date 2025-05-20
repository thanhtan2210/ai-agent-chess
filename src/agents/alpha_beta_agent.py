import chess
import time
from src.game.rules import evaluate_position, PIECE_VALUES
from agents.minimax_agent import MinimaxAgent

class AlphaBetaAgent(MinimaxAgent):
    def __init__(self, max_depth=4, time_limit=None):
        """Khởi tạo AlphaBetaAgent.
        
        Args:
            max_depth: Độ sâu tối đa của cây tìm kiếm
            time_limit: Thời gian tối đa cho mỗi nước đi (giây)
        """
        super().__init__(max_depth=max_depth, time_limit=time_limit)
        
    def get_move(self, board):
        """Tìm nước đi tốt nhất sử dụng alpha-beta pruning.
        
        Args:
            board: chess.Board object
            
        Returns:
            chess.Move object hoặc None nếu không tìm thấy nước đi
        """
        self.start_time = time.time()
        self.nodes_evaluated = 0
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Sắp xếp các nước đi theo thứ tự ưu tiên
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves)
        
        for move in ordered_moves:
            board.push(move)
            value = -self._alpha_beta(board, self.max_depth - 1, -beta, -alpha, False)
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
                
            # Kiểm tra thời gian
            if self.time_limit and time.time() - self.start_time > self.time_limit:
                break
                
        return best_move
        
    def _alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Thuật toán alpha-beta pruning.
        
        Args:
            board: chess.Board object
            depth: Độ sâu còn lại
            alpha: Giá trị alpha
            beta: Giá trị beta
            maximizing_player: True nếu là người chơi tối đa hóa
            
        Returns:
            float: Giá trị đánh giá tốt nhất
        """
        self.nodes_evaluated += 1
        
        # Kiểm tra thời gian
        if self.time_limit and time.time() - self.start_time > self.time_limit:
            return 0
            
        # Kiểm tra kết thúc game
        if board.is_game_over():
            return self._evaluate_game_over(board)
            
        # Đạt độ sâu tối đa
        if depth == 0:
            return evaluate_position(board)
            
        # Sắp xếp các nước đi
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, legal_moves)
        
        if maximizing_player:
            value = float('-inf')
            for move in ordered_moves:
                board.push(move)
                value = max(value, self._alpha_beta(board, depth - 1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in ordered_moves:
                board.push(move)
                value = min(value, self._alpha_beta(board, depth - 1, alpha, beta, True))
                board.pop()
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
            
    def _order_moves(self, board, moves):
        """Sắp xếp các nước đi theo thứ tự ưu tiên.
        
        Args:
            board: chess.Board object
            moves: List các nước đi hợp lệ
            
        Returns:
            List các nước đi đã sắp xếp
        """
        scored_moves = []
        for move in moves:
            score = 0
            
            # Ưu tiên nước bắt quân
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]
                    
            # Ưu tiên nước chiếu
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            # Ưu tiên nước đi an toàn
            board.push(move)
            if not board.is_attacked_by(not board.turn, move.to_square):
                score += 30
            board.pop()
            
            scored_moves.append((score, move))
            
        # Sắp xếp giảm dần theo điểm
        scored_moves.sort(reverse=True)
        return [move for _, move in scored_moves]
        
    def _evaluate_game_over(self, board):
        """Đánh giá trạng thái kết thúc game.
        
        Args:
            board: chess.Board object
            
        Returns:
            float: Giá trị đánh giá
        """
        if board.is_checkmate():
            return float('-inf') if board.turn else float('inf')
        return 0  # Hòa 