from abc import ABC, abstractmethod
import chess

class BaseAgent(ABC):
    def __init__(self, color):
        """Initialize the agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
        """
        self.color = color
        self.last_best_move = None  # Lưu best move tạm thời
    
    def set_best_move(self, move):
        self.last_best_move = move
    
    def get_best_move(self):
        return self.last_best_move
    
    @abstractmethod
    def get_move(self, board):
        """Get the next move for the current position.
        
        Args:
            board: chess.Board object representing the current position
            
        Returns:
            chess.Move: The chosen move
        """
        pass
    
    def get_name(self):
        """Get the name of the agent."""
        return self.__class__.__name__ 