import chess
import time
from typing import Optional, Dict, Tuple
from src.game.rules import evaluate_position, PIECE_VALUES
from src.agents.minimax_agent import MinimaxAgent

DEBUG_LOG = True # Set to False to disable logs, enable for debugging
# DEBUG_LOG = False # Temporarily disable for cleaner test runs if needed

# Piece-square tables for positional evaluation
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

class AlphaBetaAgent(MinimaxAgent):
    def __init__(self, color, depth=5, time_limit=20.0):
        """Initialize the AlphaBeta agent.
        
        Args:
            color: The color of the agent (chess.WHITE or chess.BLACK)
            depth: The maximum search depth (default: 5)
            time_limit: The maximum time to search in seconds (default: 20.0)
        """
        super().__init__(color)
        self.depth = depth
        self.time_limit = time_limit
        self.pv_table = {}  # Principal variation table
        self.history_table = {}  # History heuristic table
        self.killer_moves = [[None, None] for _ in range(100)]  # Killer moves for each ply
        if DEBUG_LOG: print(f"AlphaBetaAgent initialized. Max Depth: {depth}, Time Limit: {time_limit}")
    
    def get_move(self, board):
        """Get the best move using iterative deepening with alpha-beta pruning.
        
        Args:
            board: The current chess board state
            
        Returns:
            The best move found
        """
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        self.set_best_move(None)  # Reset best move trước khi tìm
        
        # Iterative deepening
        for current_depth in range(1, self.depth + 1):
            if time.time() - start_time > self.time_limit:
                break
                
            print(f"Iterative Deepening - Current Depth: {current_depth}")
            
            # Get all legal moves
            moves = list(board.legal_moves)
            if not moves:
                return None
                
            # Order moves
            ordered_moves = self._order_moves(board, moves, current_depth)
            
            # Search each move
            for move in ordered_moves:
                board.push(move)
                score = -self._minimax(board, current_depth - 1, -beta, -alpha, False, start_time)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    self.set_best_move(best_move)  # Lưu best move tạm thời
                
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
                    
            # Update PV table
            if best_move:
                self.pv_table[board.fen()] = best_move
                
        # Nếu hết thời gian hoặc không tìm được move ở depth cuối, trả về best_move đã lưu
        if self.get_best_move() is not None:
            print(f"AlphaBeta selected move: {self.get_best_move()} with eval: {best_score} from FEN: {board.fen()}")
            return self.get_best_move()
        else:
            print("No best move found by minimax, selecting first legal move.")
            return list(board.legal_moves)[0]
    
    def _minimax(self, board, depth, alpha, beta, maximizing, start_time):
        """Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: The current chess board state
            depth: The remaining search depth
            alpha: The alpha value for pruning
            beta: The beta value for pruning
            maximizing: Whether we're maximizing or minimizing
            start_time: The start time of the search
            
        Returns:
            The evaluation score
        """
        # Check time limit
        if time.time() - start_time > self.time_limit:
            return 0
            
        # Check for terminal states
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 if maximizing else 10000
            return 0
            
        # Check for draw by repetition or insufficient material
        if board.is_repetition() or board.is_insufficient_material():
            return 0
            
        # Check for maximum depth
        if depth == 0:
            return self.evaluate(board)
            
        # Get all legal moves
        moves = list(board.legal_moves)
        if not moves:
            return 0
            
        # Order moves
        ordered_moves = self._order_moves(board, moves, depth)
        
        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, False, start_time)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, True, start_time)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
            
    def _order_moves(self, board, moves, depth):
        """Order moves to improve alpha-beta pruning efficiency.
        
        Args:
            board: The current chess board state
            moves: List of legal moves
            depth: The current search depth
            
        Returns:
            Ordered list of moves
        """
        move_scores = []
        
        for move in moves:
            score = 0
            
            # 1. Principal Variation (PV) moves
            if board.fen() in self.pv_table and move == self.pv_table[board.fen()]:
                score += 10000
            
            # 2. Captures with MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA scoring: victim_value * 10 - attacker_value
                    victim_value = self._get_piece_value(victim)
                    attacker_value = self._get_piece_value(attacker)
                    score += 8000 + (victim_value * 10 - attacker_value)
            
            # 3. Killer moves (non-captures that caused a cutoff in the same position)
            if not board.is_capture(move) and move in self.killer_moves[depth]:
                score += 7000
            
            # 4. History heuristic
            move_key = (move.from_square, move.to_square, move.promotion)
            if move_key in self.history_table:
                score += self.history_table[move_key]
            
            # 5. Checks
            board.push(move)
            if board.is_check():
                score += 6000
                # Bonus for discovered checks
                if not board.is_attacked_by(board.turn, move.from_square):
                    score += 500
            board.pop()
            
            # 6. Promotions
            if move.promotion:
                score += 5000 + self._get_piece_value(chess.Piece(move.promotion, board.turn))
            
            # 7. Center control
            center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
            if move.to_square in center_squares:
                score += 100
            
            # 8. Pawn advancement
            if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                if board.turn == chess.WHITE:
                    score += chess.square_rank(move.to_square) * 10
                else:
                    score += (7 - chess.square_rank(move.to_square)) * 10
            
            # 9. Castling
            if board.is_castling(move):
                score += 3000
            
            # 10. Avoid moving pieces multiple times in opening
            if depth > 2:  # Only in early game
                if board.piece_at(move.from_square):
                    piece = board.piece_at(move.from_square)
                    if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
                        # Check if piece has moved before
                        if board.is_attacked_by(not board.turn, move.from_square):
                            score -= 200  # Penalize moving pieces that are under attack
            
            move_scores.append((move, score))
        
        # Sort moves by score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update history table for non-captures
        for move, score in move_scores:
            if not board.is_capture(move):
                move_key = (move.from_square, move.to_square, move.promotion)
                self.history_table[move_key] = self.history_table.get(move_key, 0) + 2 ** depth
        
        return [move for move, _ in move_scores]
        
    def _get_piece_value(self, piece):
        """Get the value of a piece.
        
        Args:
            piece: The chess piece
            
        Returns:
            The piece value
        """
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
        
    def evaluate(self, board):
        """Evaluate the current board position.
        
        Args:
            board: The current chess board state
            
        Returns:
            The evaluation score
        """
        # Check for terminal states
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 if board.turn == self.color else 10000
            return 0
            
        # Check for draw by repetition or insufficient material
        if board.is_repetition() or board.is_insufficient_material():
            return 0
            
        # Material score
        material_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            value = self._get_piece_value(piece)
            if piece.color == self.color:
                material_score += value
            else:
                material_score -= value
                
        # Positional score
        positional_score = self._evaluate_position(board)
        
        # Pawn structure score
        pawn_score = self._evaluate_pawn_structure(board)
        
        # King safety score
        king_safety_score = self._evaluate_king_safety(board)
        
        # Mobility score
        mobility_score = self._evaluate_mobility(board)
        
        # Combine scores with adjusted weights
        total_score = (
            material_score * 1.2 +  # Increased material weight
            positional_score * 0.4 +  # Increased positional weight
            pawn_score * 0.3 +  # Increased pawn structure weight
            king_safety_score * 0.3 +  # Increased king safety weight
            mobility_score * 0.2  # Increased mobility weight
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

    def _evaluate_mobility(self, board):
        """Evaluate piece mobility and control of the center.
        
        Args:
            board: The current chess board state
            
        Returns:
            The mobility score
        """
        score = 0
        
        # Evaluate mobility for both colors
        for color in [chess.WHITE, chess.BLACK]:
            # Store current turn
            current_turn = board.turn
            board.turn = color
            
            # Count legal moves
            legal_moves = list(board.legal_moves)
            mobility = len(legal_moves)
            
            # Bonus for center control
            center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
            center_control = sum(1 for move in legal_moves if move.to_square in center_squares)
            
            # Bonus for piece development
            developed_pieces = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
                        # Check if piece has moved from starting position
                        if color == chess.WHITE:
                            if square not in [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1]:
                                developed_pieces += 1
                        else:
                            if square not in [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8]:
                                developed_pieces += 1
            
            # Calculate score for this color
            color_score = (
                mobility * 0.1 +  # Basic mobility
                center_control * 0.2 +  # Center control bonus
                developed_pieces * 0.15  # Development bonus
            )
            
            # Add or subtract based on color
            if color == self.color:
                score += color_score
            else:
                score -= color_score
            
            # Restore original turn
            board.turn = current_turn
        
        return score
        
    def _evaluate_position(self, board):
        """Evaluate the current position with piece-square tables and additional factors."""
        if board.is_checkmate():
            return -10000 if board.turn == self.color else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Material and positional evaluation
        material_score = 0
        position_score = 0
        
        # Evaluate each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Material value
            value = self._get_piece_value(piece)
            
            # Positional value
            if piece.color == chess.WHITE:
                if piece.piece_type == chess.PAWN:
                    value += PAWN_TABLE[square]
                elif piece.piece_type == chess.KNIGHT:
                    value += KNIGHT_TABLE[square]
                elif piece.piece_type == chess.BISHOP:
                    value += BISHOP_TABLE[square]
                elif piece.piece_type == chess.ROOK:
                    value += ROOK_TABLE[square]
                elif piece.piece_type == chess.QUEEN:
                    value += QUEEN_TABLE[square]
                elif piece.piece_type == chess.KING:
                    value += KING_TABLE[square]
            else:
                # Mirror tables for black pieces
                if piece.piece_type == chess.PAWN:
                    value += PAWN_TABLE[chess.square_mirror(square)]
                elif piece.piece_type == chess.KNIGHT:
                    value += KNIGHT_TABLE[chess.square_mirror(square)]
                elif piece.piece_type == chess.BISHOP:
                    value += BISHOP_TABLE[chess.square_mirror(square)]
                elif piece.piece_type == chess.ROOK:
                    value += ROOK_TABLE[chess.square_mirror(square)]
                elif piece.piece_type == chess.QUEEN:
                    value += QUEEN_TABLE[chess.square_mirror(square)]
                elif piece.piece_type == chess.KING:
                    value += KING_TABLE[chess.square_mirror(square)]
            
            # Add to material score
            if piece.color == self.color:
                material_score += value
            else:
                material_score -= value
        
        # Mobility evaluation
        mobility_score = 0
        board.turn = self.color
        mobility_score += len(list(board.legal_moves))
        board.turn = not self.color
        mobility_score -= len(list(board.legal_moves))
        
        # Pawn structure evaluation
        pawn_structure_score = self._evaluate_pawn_structure(board)
        
        # King safety evaluation
        king_safety_score = self._evaluate_king_safety(board)
        
        # Combine all factors with adjusted weights
        total_score = (
            material_score + 
            mobility_score * 0.2 + 
            pawn_structure_score * 1.5 + 
            king_safety_score * 1.2
        )
        
        # Penalize for being in check
        if board.is_check():
            if board.turn == self.color:
                total_score -= 50
            else:
                total_score += 50
        
        return total_score 