import chess
import time
from typing import Optional, Dict, Tuple
from ..game.rules import evaluate_position, PIECE_VALUES
from .base_agent import BaseAgent

DEBUG_LOG = True # Set to False to disable logs
# DEBUG_LOG = False # Temporarily disable for cleaner test runs if needed, enable for debugging

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
                elif flag == 1 and stored_eval <= alpha:  # LOWERBOUND (Alpha was too low)
                    return alpha # This was previously stored_eval, should be alpha if it's a lower bound fail-low
                elif flag == 2 and stored_eval >= beta:  # UPPERBOUND (Beta was too high)
                    return beta # This was previously stored_eval, should be beta if it's an upper bound fail-high
        return None
    
    def store(self, board_hash, depth, evaluation, flag):
        """Store evaluation for position."""
        if self.entries >= self.max_size and self.max_size > 0: # Check max_size > 0
            # Simple strategy: clear a portion or the whole table
            # For now, let's just not add if full and max_size is set
            # A better strategy would be to replace entries (e.g., based on depth or age)
            # To prevent uncontrolled growth if max_size is restrictive:
            if len(self.table) >= self.max_size:
                 # print(f"TT full ({self.max_size}), not storing. Consider increasing TT size or implementing replacement.")
                 return # Or implement a replacement strategy
        self.table[board_hash] = (depth, evaluation, flag)
        self.entries = len(self.table) # Correctly update entries count

class MinimaxAgent(BaseAgent):
    def __init__(self, color, max_depth=4, time_limit=5.0):
        """Initialize the Minimax agent with alpha-beta pruning.
        
        Args:
            color: chess.WHITE or chess.BLACK
            max_depth: Maximum search depth
            time_limit: Maximum time to search in seconds
        """
        super().__init__(color)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.transposition_table = TranspositionTable()
        self.nodes_evaluated = 0
        self.start_time = 0
        self.killer_moves = [[] for _ in range(self.max_depth + 1)]  # Killer moves for each depth
        self.history_heuristic = {}  # Lưu số lần mỗi nước đi dẫn đến cutoff
        self.move_cache = {}  # Cache for move scores
        self.name = "Minimax"
        if DEBUG_LOG: print(f"MinimaxAgent initialized. Max Depth: {max_depth}, Time Limit: {time_limit}")
    
    def get_move(self, board):
        """Get the best move using iterative deepening minimax with alpha-beta pruning.
        
        Args:
            board: chess.Board object representing the current position
            
        Returns:
            chess.Move: The best move found
        """
        self.nodes_evaluated = 0
        self.start_time = time.time()
        self.move_cache.clear() # Clear move ordering cache for each new get_move call
        if DEBUG_LOG: print(f"\\n--- Minimax Get Move Called --- FEN: {board.fen()}")

        best_move_overall = None
        best_eval_overall = float('-inf')

        for current_depth_iterative in range(1, self.max_depth + 1):
            if DEBUG_LOG: print(f"Iterative Deepening - Current Depth: {current_depth_iterative}")
            # Reset killer moves for each new depth in iterative deepening if they are depth-specific
            # self.killer_moves = [[] for _ in range(current_depth_iterative + 1)] # Or handle max_depth correctly
            
            eval_at_depth, move_at_depth = self._minimax(board, current_depth_iterative, float('-inf'), float('inf'), True)
            
            if time.time() - self.start_time > self.time_limit and current_depth_iterative < self.max_depth :
                if DEBUG_LOG: print(f"Time limit reached during iterative deepening at depth {current_depth_iterative}. Using best move from previous depth.")
                break # Use results from previous depth if time is up

            if move_at_depth is not None: # Check if a move was actually found
                 best_eval_overall = eval_at_depth
                 best_move_overall = move_at_depth
            elif DEBUG_LOG:
                 print(f"No move found at depth {current_depth_iterative}. Eval: {eval_at_depth}")


        final_move = best_move_overall
        if final_move is None and list(board.legal_moves):
            if DEBUG_LOG: print("No best move found by minimax (possibly due to immediate time out or no moves explored), selecting first legal move.")
            final_move = list(board.legal_moves)[0]
        elif final_move is None:
            if DEBUG_LOG: print("No best move found and no legal moves. This should not happen if game is not over.")
            return None # Or raise error

        if DEBUG_LOG: print(f"Minimax selected move: {final_move} with eval: {best_eval_overall} from FEN: {board.fen()}")
        return final_move
    
    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: chess.Board object
            depth: Current search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: Whether the current player is maximizing
            
        Returns:
            tuple: (evaluation, best_move)
        """
        original_alpha = alpha
        original_beta = beta # Needed for TT storing logic for fail-high

        if time.time() - self.start_time > self.time_limit:
            # if DEBUG_LOG: print(f"{current_indent}Time limit in _minimax. Depth {depth}")
            return 0, None
            
        # Check transposition table
        board_hash = hash(board._transposition_key())
        tt_entry = self.transposition_table.get(board_hash, depth, alpha, beta)
        if tt_entry is not None:
            # if DEBUG_LOG: print(f"{current_indent}TT Hit: Depth {depth}, Value {tt_entry}. Returning.")
            # TT should return just the score, move is reconstructed or handled by caller
            return tt_entry, None # TT typically stores score; best move is reconstructed.
        
        if depth == 0 or board.is_game_over():
            self.nodes_evaluated += 1
            eval_score = self.quiescence_search(board, alpha, beta, 0) # depth for q_search starts at 0
            # if DEBUG_LOG: print(f"{current_indent}Leaf node/Game Over. Depth {depth}. QSearch Eval: {eval_score}")
            self.transposition_table.store(board_hash, depth, eval_score, 0) # Flag 0: EXACT
            return eval_score, None
        
        # Null Move Pruning with verification
        R = 3 if depth >= 4 else 2  # Adjust R based on depth
        if depth >= 3 and not board.is_check() and not board.is_variant_end() and not maximizing_player: # NMP is typically for non-maximizing player
            # if DEBUG_LOG: print(f"{current_indent}NMP attempt at depth {depth}")
            board.push(chess.Move.null())
            try:
                # if DEBUG_LOG: print(f"{current_indent}  NMP Push: null, FEN before _minimax: {board.fen()}")
                null_eval, _ = self._minimax(board, depth - 1 - R, -beta, -beta + 1, not maximizing_player)
            finally:
                board.pop()
                # if DEBUG_LOG: print(f"{current_indent}  NMP Pop: null, FEN after pop: {board.fen()}")
            
            null_eval = -null_eval
            if null_eval >= beta:
                # if DEBUG_LOG: print(f"{current_indent}NMP Cutoff: null_eval ({null_eval}) >= beta ({beta}). Depth {depth}.")
                # Verification search (optional, can be intensive)
                # For now, assume standard NMP cutoff is sufficient. If issues persist, can add verification.
                # self.transposition_table.store(board_hash, depth, beta, 1) # Store as LOWERBOUND
                return beta, None # Null move was good enough to cause a cutoff
        
        moves = self._order_moves(board, depth)
        best_move_found = moves[0] if moves else None
        best_eval = float('-inf') if maximizing_player else float('inf')
        
        if maximizing_player:
            current_max_eval = float('-inf')
            for move in moves:
                # assert move in board.legal_moves, f"Illegal move {move} in _minimax (max) from FEN: {board.fen()}"
                board.push(move)
                try:
                    # if DEBUG_LOG: print(f"{current_indent}  Max Push: {move}, FEN before _minimax: {board.fen()}")
                    eval_child, _ = self._minimax(board, depth - 1, alpha, beta, False)
                finally:
                    board.pop()
                    # if DEBUG_LOG: print(f"{current_indent}  Max Pop: {move}, FEN after pop: {board.fen()}")

                if eval_child > current_max_eval:
                    current_max_eval = eval_child
                    best_move_found = move
                alpha = max(alpha, current_max_eval)
                if beta <= alpha:
                    # if DEBUG_LOG: print(f"{current_indent}Max Cutoff: beta ({beta}) <= alpha ({alpha}). Move: {move}. Depth {depth}.")
                    # Store killer move for beta cutoff
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move) # Add to front
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth] = self.killer_moves[depth][:2]
                    move_key = (move.from_square, move.to_square, move.promotion)
                    self.history_heuristic[move_key] = self.history_heuristic.get(move_key, 0) + depth * depth
                    break
            best_eval = current_max_eval
        else: # Minimizing player
            current_min_eval = float('inf')
            for move in moves:
                # assert move in board.legal_moves, f"Illegal move {move} in _minimax (min) from FEN: {board.fen()}"
                board.push(move)
                try:
                    # if DEBUG_LOG: print(f"{current_indent}  Min Push: {move}, FEN before _minimax: {board.fen()}")
                    eval_child, _ = self._minimax(board, depth - 1, alpha, beta, True)
                finally:
                    board.pop()
                    # if DEBUG_LOG: print(f"{current_indent}  Min Pop: {move}, FEN after pop: {board.fen()}")

                if eval_child < current_min_eval:
                    current_min_eval = eval_child
                    best_move_found = move
                beta = min(beta, current_min_eval)
                if beta <= alpha:
                    # if DEBUG_LOG: print(f"{current_indent}Min Cutoff: beta ({beta}) <= alpha ({alpha}). Move: {move}. Depth {depth}.")
                     # Store killer move for alpha cutoff
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move) # Add to front
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth] = self.killer_moves[depth][:2]
                    move_key = (move.from_square, move.to_square, move.promotion)
                    self.history_heuristic[move_key] = self.history_heuristic.get(move_key, 0) + depth * depth
                    break
            best_eval = current_min_eval
        
        # Store in transposition table
        tt_flag = 0 # EXACT
        if best_eval <= original_alpha: # Failed low (didn't improve alpha)
            tt_flag = 2  # UPPERBOUND (actual value is <= best_eval)
        elif best_eval >= original_beta: # Failed high (exceeded beta) - this was 'beta' not 'original_beta'
            tt_flag = 1  # LOWERBOUND (actual value is >= best_eval)
        
        # if DEBUG_LOG: print(f"{current_indent}Storing to TT: Depth {depth}, Eval {best_eval}, Flag {tt_flag}, Move {best_move_found if best_move_found else 'None'}")
        self.transposition_table.store(board_hash, depth, best_eval, tt_flag)
        return best_eval, best_move_found
    
    def _order_moves(self, board, depth=0):
        """Optimized move ordering with cached scores."""
        moves = list(board.legal_moves)
        move_scores = []
        killer_set = set(self.killer_moves[depth]) if depth < len(self.killer_moves) else set()

        for move in moves:
            move_key = (move.from_square, move.to_square, move.promotion)
            if move_key in self.move_cache:
                score = self.move_cache[move_key]
            else:
                score = 0
                piece = board.piece_at(move.from_square)
                if piece is None: # Should not happen for legal moves
                    continue

                # Killer moves
                if move in killer_set:
                    score += 20000 # Highest priority

                # Captures (MVV-LVA)
                if board.is_capture(move):
                    victim = board.piece_at(move.to_square)
                    if board.is_en_passant(move):
                        victim_value = PIECE_VALUES[chess.PAWN]
                    elif victim:
                        victim_value = PIECE_VALUES[victim.piece_type]
                    else:
                        victim_value = 0
                    
                    attacker_value = PIECE_VALUES[piece.piece_type]
                    # Refined MVV-LVA: Prioritize valuable victims, less valuable attackers
                    score += 15000 + victim_value - (attacker_value / 10) 
                
                # Checks - make sure this has high priority but less than killer captures and good captures
                board.push(move)
                if board.is_check():
                    score += 5000
                board.pop() # Important to pop after push

                # History heuristic - apply after specific tactical moves like captures/checks
                score += self.history_heuristic.get(move_key, 0) # Max value could be max_depth^2

                # Pawn pushes to center/advanced ranks
                if piece.piece_type == chess.PAWN:
                    if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                        score += 100
                    if (piece.color == chess.WHITE and chess.square_rank(move.to_square) == 6) or \
                       (piece.color == chess.BLACK and chess.square_rank(move.to_square) == 1):
                        score += 200 # Pawn promotion imminent

                # Piece development
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                        score += 50
                    # Penalty for moving already developed piece back to back rank (heuristic)
                    if (piece.color == chess.WHITE and chess.square_rank(move.from_square) > 0 and chess.square_rank(move.to_square) == 0) or \
                       (piece.color == chess.BLACK and chess.square_rank(move.from_square) < 7 and chess.square_rank(move.to_square) == 7):
                        score -= 30
                    elif (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
                         (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                        score += 30 # Bonus for moving from back rank

                # Castling
                if board.is_castling(move):
                    score += 1500
                
                # Safety check (moving to an unattacked square)
                # This is complex; for now, rely on q-search to handle safety of captures
                # board.push(move)
                # if not board.is_attacked_by(not piece.color, move.to_square):
                #     score += 20
                # board.pop()

                self.move_cache[move_key] = score
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def get_name(self):
        """Get the name of the agent with its parameters."""
        return f"{self.name}(depth={self.max_depth}, time={self.time_limit}s)"

    def quiescence_search(self, board, alpha, beta, q_depth): # Renamed depth to q_depth for clarity
        """Quiescence search with delta pruning."""
        if time.time() - self.start_time > self.time_limit:
            # if DEBUG_LOG: print(f"{current_indent}Time limit in QSearch. Depth {q_depth}")
            return 0 # Return a neutral score on timeout

        self.nodes_evaluated += 1 # Count nodes in q-search too

        if q_depth >= 8: # Max quiescence search depth (can be tuned)
            # if DEBUG_LOG: print(f"{current_indent}QSearch max depth reached. Eval: {evaluate_position(board)}")
            return evaluate_position(board)

        stand_pat_score = evaluate_position(board)
        # if DEBUG_LOG: print(f"{current_indent}QSearch Stand Pat: {stand_pat_score}")

        if stand_pat_score >= beta:
            # if DEBUG_LOG: print(f"{current_indent}QSearch Beta Cutoff (Stand Pat): {stand_pat_score} >= {beta}")
            return beta
        if alpha < stand_pat_score:
            alpha = stand_pat_score

        # Only consider captures and promotions in quiescence search. Potentially checks too.
        # For captures, generate them ordered by MVV-LVA if possible, or just iterate through legal moves and pick captures.
        # For now, iterate through legal moves and filter. A more optimized q-search would generate only captures/promotions.
        
        # Create a list of moves to consider: captures and promotions.
        # Potentially add checks if they are not too noisy.
        q_moves = []
        for move in board.legal_moves: # Consider optimizing to generate only tactical moves
            if board.is_capture(move) or move.promotion is not None: # or self._is_check_move(board, move)
                 q_moves.append(move)
        
        # Simple MVV-LVA for q_moves (optional but good)
        # sorted_q_moves = self._order_q_moves(board, q_moves) # A simpler ordering for q-search

        for move in q_moves: # Iterate through filtered tactical moves
            # Basic Delta Pruning (can be more sophisticated)
            if not board.is_capture(move) and move.promotion is None: # If not a capture or promotion, and we are only considering these
                pass # If we also consider checks, this logic needs adjustment
            elif board.is_capture(move): # Apply delta pruning for captures
                 piece = board.piece_at(move.from_square)
                 captured_piece = board.piece_at(move.to_square)
                 if board.is_en_passant(move):
                     approx_gain = PIECE_VALUES[chess.PAWN]
                 elif captured_piece:
                     approx_gain = PIECE_VALUES[captured_piece.piece_type]
                 else: # Should not happen for a valid capture
                     approx_gain = 0
                 
                 # If stand_pat + gain + futility_margin < alpha, prune
                 futility_margin = 200 # e.g., value of a pawn or two
                 if stand_pat_score + approx_gain + futility_margin < alpha and not board.is_check(): # Don't prune checks this way
                      # if DEBUG_LOG: print(f"{current_indent}  QSearch Delta Prune: Move {move}, Gain {approx_gain}, SP {stand_pat_score}, Alpha {alpha}")
                      continue
            
            # assert move in board.legal_moves, f"Illegal move {move} in QSearch from FEN: {board.fen()}"
            board.push(move)
            try:
                # if DEBUG_LOG: print(f"{current_indent}  QPush: {move}, FEN before q_search: {board.fen()}")
                score = -self.quiescence_search(board, -beta, -alpha, q_depth + 1)
            finally:
                board.pop()
                # if DEBUG_LOG: print(f"{current_indent}  QPop: {move}, FEN after pop: {board.fen()}")
            
            if score >= beta:
                # if DEBUG_LOG: print(f"{current_indent}  QSearch Beta Cutoff: Move {move}, Score {score} >= Beta {beta}")
                return beta
            if score > alpha:
                alpha = score
                # if DEBUG_LOG: print(f"{current_indent}  QSearch Alpha Update: Move {move}, Score {score}, New Alpha {alpha}")

        return alpha

    def _is_check_move(self, board, move):
        board.push(move)
        is_check = board.is_check()
        board.pop()
        return is_check 