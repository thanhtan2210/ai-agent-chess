import chess
import time
from typing import Optional, Dict, Tuple
from src.game.rules import evaluate_position, PIECE_VALUES
from src.agents.base_agent import BaseAgent

# DEBUG_LOG = True # Set to False to disable logs
DEBUG_LOG = False # Tắt log chi tiết nước đi khi chạy agent

class TranspositionTable:
    def __init__(self, max_size=1000000):
        self.table = {}
        self.max_size = max_size
        self.entries = 0
    
    def get(self, board_hash, depth, alpha, beta):
        """Get stored evaluation for position if it exists and is deep enough."""
        if board_hash in self.table:
            stored_depth, stored_eval, flag = self.table[board_hash]
            if stored_depth >= depth:
                if flag == 0:  # EXACT
                    return stored_eval
                elif flag == 1 and stored_eval <= alpha:  # LOWERBOUND
                    return alpha
                elif flag == 2 and stored_eval >= beta:  # UPPERBOUND
                    return beta
        return None
    
    def store(self, board_hash, depth, evaluation, flag):
        """Store evaluation for position with replacement strategy."""
        if self.entries >= self.max_size and self.max_size > 0:
            # Replace oldest entries when table is full
            oldest_key = min(self.table.keys(), key=lambda k: self.table[k][0])
            del self.table[oldest_key]
            self.entries -= 1
        
        self.table[board_hash] = (depth, evaluation, flag)
        self.entries = len(self.table)

class MinimaxAgent(BaseAgent):
    def __init__(self, color, max_depth=5, time_limit=20.0):
        """Initialize the Minimax agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
            max_depth: Maximum search depth (default: 5)
            time_limit: Maximum time to search in seconds (default: 20.0)
        """
        super().__init__(color)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = None
        self.nodes_evaluated = 0
        self.transposition_table = TranspositionTable()
        self.pv_table = {}  # Principal variation table
        self.history_table = {}  # History heuristic table
        self.killer_moves = [[None, None] for _ in range(max_depth)]  # Killer moves for each ply
        self.move_cache = {}  # Cache for move scores
        self.name = "Minimax"
        print(f"MinimaxAgent initialized. Max Depth: {max_depth}, Time Limit: {time_limit}")
    
    def get_move(self, board):
        """Get the best move using iterative deepening with minimax."""
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.move_cache.clear()  # Clear move cache for new position
        
        if DEBUG_LOG: print(f"\n--- Minimax Get Move Called --- FEN: {board.fen()}")
        
        best_move = None
        best_score = float('-inf')
        self.set_best_move(None)  # Reset best move trước khi tìm
        
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break
                
            if DEBUG_LOG: print(f"Iterative Deepening - Current Depth: {depth}")
            score, move = self._minimax(board, depth, float('-inf'), float('inf'), True)
            
            if move:
                best_move = move
                best_score = score
                self.set_best_move(best_move)  # Lưu best move tạm thời
            
            # Update principal variation
            if best_move:
                self.pv_table[board.fen()] = best_move
                
        # Nếu hết thời gian hoặc không tìm được move ở depth cuối, trả về best_move đã lưu
        if self.get_best_move() is not None:
            if DEBUG_LOG: print(f"Minimax selected move: {self.get_best_move()} with eval: {best_score} from FEN: {board.fen()}")
            return self.get_best_move()
        else:
            if DEBUG_LOG: print("No best move found by minimax, selecting first legal move.")
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            else:
                return None
    
    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning."""
        if time.time() - self.start_time > self.time_limit:
            return 0, None
            
        if depth == 0 or board.is_game_over():
            self.nodes_evaluated += 1
            return self.evaluate(board), None
            
        # Check transposition table
        board_hash = hash(board._transposition_key())
        tt_entry = self.transposition_table.get(board_hash, depth, alpha, beta)
        if tt_entry is not None:
            return tt_entry, None # TT typically stores score; best move is reconstructed.
        
        moves = list(board.legal_moves)
        if not moves:
            return 0, None
            
        # Order moves for better pruning
        moves = self._order_moves(board, moves, depth)
        
        best_move = None
        if maximizing_player:
            best_score = float('-inf')
            for move in moves:
                board.push(move)
                score, _ = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
                    
            return best_score, best_move
        else:
            best_score = float('inf')
            for move in moves:
                board.push(move)
                score, _ = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if score < best_score:
                    best_score = score
                    best_move = move
                    
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
                    
            return best_score, best_move
    
    def _order_moves(self, board, moves, depth):
        """Order moves for better alpha-beta pruning."""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Check principal variation
            if board.fen() in self.pv_table and move == self.pv_table[board.fen()]:
                score += 10000
                
            # Check history heuristic
            move_key = (move.from_square, move.to_square)
            if move_key in self.history_table:
                score += self.history_table[move_key] * 2  # Increased history weight
                
            # Check killer moves
            killer_index = min(depth, len(self.killer_moves) - 1)
            if move in self.killer_moves[killer_index]:
                score += 9000
                
            # Check captures with MVV-LVA
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 8000 + (self._get_piece_value(victim) * 10 - self._get_piece_value(attacker))
            
            # Check checks
            if self._is_check_move(board, move):
                score += 7000
                
            # Check promotions
            if move.promotion:
                score += 6000 + self._get_piece_value(chess.Piece(move.promotion, board.turn))
                
            # Check center control
            center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
            if move.to_square in center_squares:
                score += 500
                
            move_scores.append((score, move))
            
        # Sort moves by score
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def _get_piece_value(self, piece):
        """Get the value of a piece."""
        if piece is None:
            return 0
            
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }
        return values[piece.piece_type]
    
    def get_name(self):
        """Get the name of the agent with its parameters."""
        return f"{self.name}(depth={self.max_depth}, time={self.time_limit}s)"

    def quiescence_search(self, board, alpha, beta, q_depth, max_q_depth=3):
        """Quiescence search with depth limit."""
        if q_depth >= max_q_depth:
            return self.evaluate(board)
            
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate(board)
            
        stand_pat = self.evaluate(board)
        
        if stand_pat >= beta:
            return beta
            
        alpha = max(alpha, stand_pat)
        
        # Only consider captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        if not captures:
            return stand_pat
            
        # Order captures using MVV-LVA
        ordered_captures = []
        for move in captures:
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                score = PIECE_VALUES[victim.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]
                ordered_captures.append((score, move))
        
        ordered_captures.sort(key=lambda x: x[0], reverse=True)
        
        for _, move in ordered_captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, q_depth + 1, max_q_depth)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
            
        return alpha

    def _is_check_move(self, board, move):
        board.push(move)
        is_check = board.is_check()
        board.pop()
        return is_check

    def evaluate(self, board):
        """Evaluate the current board position."""
        if board.is_checkmate():
            return -10000 if board.turn == self.color else 10000
            
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # Material score
        material_score = 0
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                material_score += value if piece.color == self.color else -value
                
        # Positional score
        positional_score = 0
        piece_square_tables = {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10, 10, 10, 10, 10,  5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                0,  0,  0,  5,  5,  0,  0,  0
            ],
            chess.QUEEN: [
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5,  5,  5,  5,  0,-10,
                -5,  0,  5,  5,  5,  5,  0, -5,
                0,  0,  5,  5,  5,  5,  0, -5,
                -10,  5,  5,  5,  5,  5,  0,-10,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -20,-10,-10, -5, -5,-10,-10,-20
            ],
            chess.KING: [
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                20, 20,  0,  0,  0,  0, 20, 20,
                20, 30, 10,  0,  0, 10, 30, 20
            ]
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                table = piece_square_tables[piece.piece_type]
                value = table[square if piece.color == chess.WHITE else 63 - square]
                positional_score += value if piece.color == self.color else -value
                
        # Pawn structure score
        pawn_structure_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                # Doubled pawns
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                doubled = False
                for r in range(8):
                    if r != rank:
                        test_square = chess.square(file, r)
                        test_piece = board.piece_at(test_square)
                        if test_piece and test_piece.piece_type == chess.PAWN and test_piece.color == piece.color:
                            doubled = True
                            break
                if doubled:
                    pawn_structure_score -= 20 if piece.color == self.color else 20
                    
                # Isolated pawns
                isolated = True
                for f in [file - 1, file + 1]:
                    if 0 <= f < 8:
                        for r in range(8):
                            test_square = chess.square(f, r)
                            test_piece = board.piece_at(test_square)
                            if test_piece and test_piece.piece_type == chess.PAWN and test_piece.color == piece.color:
                                isolated = False
                                break
                if isolated:
                    pawn_structure_score -= 10 if piece.color == self.color else 10
                    
        # King safety score
        king_safety_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KING:
                # Count pawns in front of king
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                pawn_shield = 0
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        for r in [rank - 1, rank - 2]:
                            if 0 <= r < 8:
                                test_square = chess.square(f, r)
                                test_piece = board.piece_at(test_square)
                                if test_piece and test_piece.piece_type == chess.PAWN and test_piece.color == piece.color:
                                    pawn_shield += 1
                king_safety_score += pawn_shield * 10 if piece.color == self.color else -pawn_shield * 10
                
        # Mobility score
        mobility_score = 0
        for move in board.legal_moves:
            mobility_score += 1 if board.turn == self.color else -1
            
        # Combine all scores with weights
        total_score = (
            material_score * 1.0 +
            positional_score * 0.1 +
            pawn_structure_score * 0.1 +
            king_safety_score * 0.1 +
            mobility_score * 0.05
        )
        
        return total_score if self.color == chess.WHITE else -total_score
        
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure."""
        score = 0
        
        # Doubled pawns
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            
            if white_pawns > 1:
                score -= 30 * (white_pawns - 1)
            if black_pawns > 1:
                score += 30 * (black_pawns - 1)
        
        # Isolated pawns
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    is_isolated = True
                    if file > 0:  # Check left file
                        for r in range(8):
                            if board.piece_at(chess.square(file-1, r)) == piece:
                                is_isolated = False
                                break
                    if file < 7:  # Check right file
                        for r in range(8):
                            if board.piece_at(chess.square(file+1, r)) == piece:
                                is_isolated = False
                                break
                    
                    if is_isolated:
                        if piece.color == chess.WHITE:
                            score -= 20
                        else:
                            score += 20
        
        return score if self.color == chess.WHITE else -score
        
    def _evaluate_king_safety(self, board):
        """Evaluate king safety."""
        score = 0
        
        # Count pawns around king
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                king_rank = chess.square_rank(king_square)
                king_file = chess.square_file(king_square)
                
                # Count pawns in front of king
                pawn_shield = 0
                for file in range(max(0, king_file - 1), min(7, king_file + 2)):
                    for rank in range(max(0, king_rank - 1), min(7, king_rank + 2)):
                        square = chess.square(file, rank)
                        piece = board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            pawn_shield += 1
                
                if color == self.color:
                    score += pawn_shield * 10
                else:
                    score -= pawn_shield * 10
        
        return score 