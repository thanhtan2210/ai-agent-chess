import chess
import time
from typing import Optional, List, Tuple

class ChessGame:
    """Class to manage the chess game state."""
    
    def __init__(self):
        """Initialize a new chess game."""
        self.board = chess.Board()
        self.start_time = time.time()
        self.move_history: List[Tuple[chess.Move, str]] = []  # List of (move, san) tuples
        
    def make_move(self, move: chess.Move) -> bool:
        """Make a move on the board.
        
        Args:
            move: The move to make
            
        Returns:
            bool: True if move was made successfully, False otherwise
        """
        if move in self.board.legal_moves:
            san = self.board.san(move)
            self.board.push(move)
            self.move_history.append((move, san))
            return True
        return False
        
    def undo_move(self) -> bool:
        """Undo the last move.
        
        Returns:
            bool: True if move was undone successfully, False otherwise
        """
        if len(self.move_history) > 0:
            self.board.pop()
            self.move_history.pop()
            return True
        return False
        
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in the current position.
        
        Returns:
            List[chess.Move]: List of legal moves
        """
        return list(self.board.legal_moves)
        
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.board.is_game_over()
        
    def get_result(self) -> Optional[str]:
        """Get the game result.
        
        Returns:
            Optional[str]: Game result (1-0, 0-1, 1/2-1/2) or None if game is not over
        """
        if self.is_game_over():
            return self.board.outcome().result()
        return None
        
    def get_elapsed_time(self) -> float:
        """Get the elapsed time since game start.
        
        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.start_time
        
    def get_move_history(self) -> List[Tuple[chess.Move, str]]:
        """Get the move history.
        
        Returns:
            List[Tuple[chess.Move, str]]: List of (move, san) tuples
        """
        return self.move_history
        
    def reset(self):
        """Reset the game to initial state."""
        self.board = chess.Board()
        self.start_time = time.time()
        self.move_history = [] 