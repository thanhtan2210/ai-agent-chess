import random
import chess
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def get_move(self, board):
        """Get a random legal move.
        
        Args:
            board: chess.Board object representing the current position
            
        Returns:
            chess.Move: A random legal move
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves) 