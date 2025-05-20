import chess
import time
from typing import Optional, Dict, Tuple
from src.game.rules import evaluate_position, PIECE_VALUES
from src.agents.minimax_agent import MinimaxAgent

DEBUG_LOG = True # Set to False to disable logs, enable for debugging
# DEBUG_LOG = False # Temporarily disable for cleaner test runs if needed

class AlphaBetaAgent(MinimaxAgent):
    def __init__(self, color, depth=5, time_limit=5.0):
        """Initialize the Alpha-Beta agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
            depth: Maximum search depth
            time_limit: Maximum time to search in seconds
        """
        super().__init__(color, max_depth=depth, time_limit=time_limit)
        self.name = "AlphaBeta"
        self.pv_table = {}  # Principal Variation table
        self.move_history = {}  # Move history for move ordering
        if DEBUG_LOG: print(f"AlphaBetaAgent initialized. Max Depth: {depth}, Time Limit: {time_limit}")
    
    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        """Principal Variation Search implementation of minimax."""
        if time.time() - self.start_time > self.time_limit:
            return 0, None
            
        # Check transposition table
        board_hash = hash(board._transposition_key())
        stored_eval = self.transposition_table.get(board_hash, depth, alpha, beta)
        if stored_eval is not None:
            return stored_eval, None
        
        if depth == 0 or board.is_game_over():
            self.nodes_evaluated += 1
            eval_score = self.quiescence_search(board, alpha, beta, 0)
            self.transposition_table.store(board_hash, depth, eval_score, 0)
            return eval_score, None
        
        # Null Move Pruning
        R = 3 if depth >= 4 else 2  # Tăng R cho độ sâu lớn hơn
        if depth >= 3 and not board.is_check() and not board.is_variant_end() and not maximizing_player:
            board.push(chess.Move.null())
            try:
                null_eval, _ = self._minimax(board, depth - 1 - R, -beta, -beta + 1, not maximizing_player)
            finally:
                board.pop()
            null_eval = -null_eval
            if null_eval >= beta:
                return beta, None
        
        moves = self._order_moves(board, depth)
        best_move = moves[0] if moves else None
        best_eval = float('-inf') if maximizing_player else float('inf')
        
        # Principal Variation Search
        for i, move in enumerate(moves):
            board.push(move)
            try:
                if i == 0:  # First move - full window search
                    current_eval, _ = self._minimax(board, depth - 1, -beta, -alpha, not maximizing_player)
                else:  # Other moves - zero window search
                    current_eval, _ = self._minimax(board, depth - 1, -(alpha + 1), -alpha, not maximizing_player)
                    if alpha < current_eval < beta:  # Re-search with full window
                        current_eval, _ = self._minimax(board, depth - 1, -beta, -alpha, not maximizing_player)
                current_eval = -current_eval
            finally:
                board.pop()
            
            if maximizing_player:
                if current_eval > best_eval:
                    best_eval = current_eval
                    best_move = move
                    # Update PV table
                    self.pv_table[board_hash] = move
                alpha = max(alpha, best_eval)
            else:
                if current_eval < best_eval:
                    best_eval = current_eval
                    best_move = move
                    # Update PV table
                    self.pv_table[board_hash] = move
                beta = min(beta, best_eval)
            
            if beta <= alpha:
                # Cập nhật history heuristic cho beta cutoff
                move_key = (move.from_square, move.to_square, move.promotion)
                self.move_history[move_key] = self.move_history.get(move_key, 0) + depth * depth
                break
        
        # Store result in transposition table
        tt_flag = 0
        if best_eval <= alpha:
            tt_flag = 2  # UPPERBOUND
        elif best_eval >= beta:
            tt_flag = 1  # LOWERBOUND
        self.transposition_table.store(board_hash, depth, best_eval, tt_flag)
        
        return best_eval, best_move
    
    def _order_moves(self, board, depth=0):
        """Enhanced move ordering with PV table and history heuristic."""
        moves = list(board.legal_moves)
        move_scores = []
        board_hash = hash(board._transposition_key())
        
        # Get PV move if available
        pv_move = self.pv_table.get(board_hash)
        
        # Tạo danh sách các nước đi theo loại
        captures = []
        checks = []
        normal_moves = []
        
        for move in moves:
            # PV move gets highest priority
            if move == pv_move:
                move_scores.append((1000000, move))
                continue
                
            # Phân loại nước đi
            if board.is_capture(move):
                captures.append(move)
            else:
                board.push(move)
                if board.is_check():
                    checks.append(move)
                else:
                    normal_moves.append(move)
                board.pop()

        # Sắp xếp captures theo MVV-LVA
        for move in captures:
            score = 0
            piece = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            
            if board.is_en_passant(move):
                victim_value = PIECE_VALUES[chess.PAWN]
            elif victim:
                victim_value = PIECE_VALUES[victim.piece_type]
            else:
                victim_value = 0
            
            attacker_value = PIECE_VALUES[piece.piece_type]
            # Tăng độ ưu tiên cho captures
            score = 200000 + victim_value * 10 - attacker_value
            
            # Kiểm tra xem nước đi có an toàn không
            board.push(move)
            if board.is_check():
                score -= 50000  # Giảm điểm cho captures không an toàn
            board.pop()
            
            move_scores.append((score, move))

        # Sắp xếp checks
        for move in checks:
            score = 100000
            # Thêm điểm cho checks có thể gây thiệt hại
            board.push(move)
            if board.is_check():
                # Kiểm tra xem có thể bắt quân sau khi chiếu không
                for reply in board.legal_moves:
                    if board.is_capture(reply):
                        score += 50000
                        break
            board.pop()
            move_scores.append((score, move))

        # Sắp xếp normal moves
        for move in normal_moves:
            score = 0
            move_key = (move.from_square, move.to_square, move.promotion)
            piece = board.piece_at(move.from_square)

            # Killer moves
            if move in self.killer_moves[depth]:
                score += 80000

            # History heuristic
            score += self.move_history.get(move_key, 0)

            # Pawn pushes to center/advanced ranks
            if piece.piece_type == chess.PAWN:
                if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                    score += 2000
                if (piece.color == chess.WHITE and chess.square_rank(move.to_square) == 6) or \
                   (piece.color == chess.BLACK and chess.square_rank(move.to_square) == 1):
                    score += 4000

            # Piece development
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                    score += 1600
                if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
                     (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                    score += 1200

            # Castling
            if board.is_castling(move):
                score += 90000

            # Kiểm tra an toàn của nước đi
            board.push(move)
            if board.is_check():
                score -= 30000  # Giảm điểm cho nước đi không an toàn
            board.pop()

            move_scores.append((score, move))
        
        # Sắp xếp tất cả nước đi theo điểm số
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def get_name(self):
        """Get the name of the agent with its parameters."""
        return f"{self.name}(depth={self.max_depth}, time={self.time_limit}s)"

    def evaluate(self, board):
        """Enhanced evaluation function with piece-square tables and mobility."""
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 if board.turn == self.color else 10000
            return 0  # Draw
        
        # Material evaluation with piece-square tables
        material_score = 0
        mobility_score = 0
        
        # Piece-square tables for positional evaluation
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        # Evaluate each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Material value
            value = PIECE_VALUES[piece.piece_type]
            
            # Positional value
            if piece.color == chess.WHITE:
                if piece.piece_type == chess.PAWN:
                    value += pawn_table[square]
                elif piece.piece_type == chess.KNIGHT:
                    value += knight_table[square]
                elif piece.piece_type == chess.BISHOP:
                    value += bishop_table[square]
                elif piece.piece_type == chess.ROOK:
                    value += rook_table[square]
                elif piece.piece_type == chess.QUEEN:
                    value += queen_table[square]
                elif piece.piece_type == chess.KING:
                    value += king_table[square]
            else:
                # Mirror tables for black pieces
                if piece.piece_type == chess.PAWN:
                    value += pawn_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.KNIGHT:
                    value += knight_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.BISHOP:
                    value += bishop_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.ROOK:
                    value += rook_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.QUEEN:
                    value += queen_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.KING:
                    value += king_table[chess.square_mirror(square)]
            
            # Add to material score
            if piece.color == self.color:
                material_score += value
            else:
                material_score -= value
        
        # Mobility evaluation
        board.turn = self.color
        mobility_score += len(list(board.legal_moves))
        board.turn = not self.color
        mobility_score -= len(list(board.legal_moves))
        
        # Pawn structure evaluation
        pawn_structure_score = self._evaluate_pawn_structure(board)
        
        # King safety evaluation
        king_safety_score = self._evaluate_king_safety(board)
        
        # Combine all factors with adjusted weights
        total_score = material_score + mobility_score * 0.2 + pawn_structure_score * 1.5 + king_safety_score * 1.2
        
        # Penalize for being in check
        if board.is_check():
            if board.turn == self.color:
                total_score -= 50
            else:
                total_score += 50
        
        return total_score
        
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
        
        # Find kings
        white_king = None
        black_king = None
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KING:
                if piece.color == chess.WHITE:
                    white_king = square
                else:
                    black_king = square
        
        if white_king is None or black_king is None:
            return 0
        
        # Evaluate pawn shield
        def evaluate_pawn_shield(king_square, color):
            shield_score = 0
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Check pawns in front of king
            for file_offset in [-1, 0, 1]:
                if 0 <= king_file + file_offset <= 7:
                    for rank_offset in [1, 2]:
                        if 0 <= king_rank + rank_offset <= 7:
                            square = chess.square(king_file + file_offset, king_rank + rank_offset)
                            piece = board.piece_at(square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                shield_score += 10
        
        # Evaluate piece attacks near king
        def evaluate_king_attacks(king_square, color):
            attack_score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != color:
                    # Calculate Manhattan distance to king
                    piece_file = chess.square_file(square)
                    piece_rank = chess.square_rank(square)
                    king_file = chess.square_file(king_square)
                    king_rank = chess.square_rank(king_square)
                    distance = abs(piece_file - king_file) + abs(piece_rank - king_rank)
                    
                    if distance <= 2:
                        attack_score += PIECE_VALUES[piece.piece_type] * (3 - distance)
            
            return attack_score
        
        # Evaluate for both kings
        white_shield = evaluate_pawn_shield(white_king, chess.WHITE)
        black_shield = evaluate_pawn_shield(black_king, chess.BLACK)
        white_attacks = evaluate_king_attacks(white_king, chess.WHITE)
        black_attacks = evaluate_king_attacks(black_king, chess.BLACK)
        
        score = (white_shield - black_shield) + (black_attacks - white_attacks)
        return score if self.color == chess.WHITE else -score 