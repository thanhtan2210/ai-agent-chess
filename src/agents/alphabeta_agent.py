import chess
import time
from typing import Optional, Dict, Tuple
from ..game.rules import evaluate_position
from .minimax_agent import MinimaxAgent

DEBUG_LOG = True # Set to False to disable logs, enable for debugging
# DEBUG_LOG = False # Temporarily disable for cleaner test runs if needed

class AlphaBetaAgent(MinimaxAgent):
    def __init__(self, color, depth=3, time_limit=5.0):
        """Initialize the Alpha-Beta agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
            depth: Maximum search depth
            time_limit: Maximum time to search in seconds
        """
        super().__init__(color, max_depth=depth, time_limit=time_limit)
        self.name = "AlphaBeta"
        if DEBUG_LOG: print(f"AlphaBetaAgent initialized. Max Depth: {depth}, Time Limit: {time_limit}")
    
    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        """Alpha-beta pruning implementation of minimax.
        
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
        original_beta = beta # For TT storing

        if time.time() - self.start_time > self.time_limit:
            # if DEBUG_LOG: print(f"{current_indent}Time limit in AlphaBeta _minimax. Depth {depth}")
            return 0, None
            
        # Check transposition table
        board_hash = hash(board._transposition_key())
        stored_eval = self.transposition_table.get(board_hash, depth, alpha, beta)
        if stored_eval is not None:
            # if DEBUG_LOG: print(f"{current_indent}AlphaBeta TT Hit: Depth {depth}, Value {stored_eval}. Returning.")
            return stored_eval, None
        
        if depth == 0 or board.is_game_over():
            self.nodes_evaluated += 1
            eval_score = self.quiescence_search(board, alpha, beta, 0)
            # if DEBUG_LOG: print(f"{current_indent}AlphaBeta Leaf/Game Over. Depth {depth}. QSearch Eval: {eval_score}")
            self.transposition_table.store(board_hash, depth, eval_score, 0)
            return eval_score, None
        
        # Null Move Pruning
        R = 2  # Reduction depth for null move
        if depth >= 3 and not board.is_check() and not board.is_variant_end() and not maximizing_player:
            # if DEBUG_LOG: print(f"{current_indent}AlphaBeta NMP attempt at depth {depth}")
            board.push(chess.Move.null())
            try:
                # if DEBUG_LOG: print(f"{current_indent}  AlphaBeta NMP Push: null, FEN before _minimax: {board.fen()}")
                null_eval, _ = self._minimax(board, depth - 1 - R, -beta, -beta + 1, not maximizing_player)
            finally:
                board.pop()
                # if DEBUG_LOG: print(f"{current_indent}  AlphaBeta NMP Pop: null, FEN after pop: {board.fen()}")
            null_eval = -null_eval
            if null_eval >= beta:
                # if DEBUG_LOG: print(f"{current_indent}AlphaBeta NMP Cutoff: null_eval ({null_eval}) >= beta ({beta}). Depth {depth}.")
                # self.transposition_table.store(board_hash, depth, beta, 1) # Store as LOWERBOUND for NMP cutoff
                return beta, None
        
        moves = self._order_moves(board, depth)
        best_move = moves[0] if moves else None
        best_eval = float('-inf') if maximizing_player else float('inf')
        
        # Principal Variation Search
        for i, move in enumerate(moves):
            board.push(move)
            try:
                current_eval = 0
                if i == 0:  # First move - full window search
                    current_eval, _ = self._minimax(board, depth - 1, -original_beta, -alpha, not maximizing_player)
                else:  # Other moves - zero window search
                    current_eval, _ = self._minimax(board, depth - 1, -(alpha + 1), -alpha, not maximizing_player)
                    if alpha < current_eval < beta:  # Re-search with full window
                        current_eval, _ = self._minimax(board, depth - 1, -original_beta, -beta, not maximizing_player)
                current_eval = -current_eval
            finally:
                board.pop()
            
            if maximizing_player:
                if current_eval > best_eval:
                    best_eval = current_eval
                    best_move = move
                alpha = max(alpha, best_eval)
            else:
                if current_eval < best_eval:
                    best_eval = current_eval
                    best_move = move
                beta = min(beta, best_eval)
            
            if beta <= alpha:
                break
        
        # Store result in transposition table (for interior nodes)
        tt_flag = 0
        if best_eval <= original_alpha:
            tt_flag = 2
        elif best_eval >= original_beta:
            tt_flag = 1
        else:
            tt_flag = 0
        self.transposition_table.store(board_hash, depth, best_eval, tt_flag)
        # if DEBUG_LOG: print(f"{current_indent}AlphaBeta Storing TT: Depth {depth}, Eval {best_eval}, Flag {tt_flag}, Move {best_move if best_move else 'None'}")

        return best_eval, best_move
    
    def get_name(self):
        """Get the name of the agent with its parameters."""
        return f"{self.name}(depth={self.max_depth}, time={self.time_limit}s)" 