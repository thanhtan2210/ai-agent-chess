import random
import chess
from typing import Optional, List
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, color: chess.Color, prefer_captures: bool = True, prefer_checks: bool = True, seed: int = 42):
        """Initialize the random agent.
        
        Args:
            color: The color the agent plays as (chess.WHITE or chess.BLACK)
            prefer_captures: Whether to prefer capturing moves
            prefer_checks: Whether to prefer checking moves
            seed: Random seed for deterministic behavior
        """
        super().__init__(color)
        self.prefer_captures = prefer_captures
        self.prefer_checks = prefer_checks
        self.name = "Random"
        self.seed = seed
        print(f"RandomAgent initialized. Prefer captures: {prefer_captures}, Prefer checks: {prefer_checks}, Seed: {seed}")

    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a random legal move.
        
        Args:
            board: The current chess board
            
        Returns:
            A random legal move, or None if no moves are available
        """
        if board.is_game_over():
            return None
            
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Get preferred moves if enabled
        preferred_moves = self._get_preferred_moves(board, legal_moves)
        
        # If we have preferred moves and preferences are enabled, use them
        if preferred_moves and (self.prefer_captures or self.prefer_checks):
            # Use board state to deterministically select a move
            move_index = hash(board.fen()) % len(preferred_moves)
            move = preferred_moves[move_index]
        else:
            # Use board state to deterministically select a move
            move_index = hash(board.fen()) % len(legal_moves)
            move = legal_moves[move_index]
            
        self.set_best_move(move)
        return move

    def _get_preferred_moves(self, board: chess.Board, legal_moves: List[chess.Move]) -> List[chess.Move]:
        """Get a list of preferred moves based on agent preferences.
        
        Args:
            board: The current chess board
            legal_moves: List of all legal moves
            
        Returns:
            List of preferred moves
        """
        preferred_moves = []
        
        for move in legal_moves:
            is_preferred = False
            
            # Check for captures
            if self.prefer_captures and board.is_capture(move):
                is_preferred = True
                
            # Check for checks
            if self.prefer_checks:
                board.push(move)
                if board.is_check():
                    is_preferred = True
                board.pop()
                
            if is_preferred:
                preferred_moves.append(move)
                
        return preferred_moves

    def get_name(self) -> str:
        """Get the name of the agent with its preferences."""
        preferences = []
        if self.prefer_captures:
            preferences.append("captures")
        if self.prefer_checks:
            preferences.append("checks")
            
        if preferences:
            return f"{self.name} ({', '.join(preferences)})"
        return self.name 