import chess
import numpy as np

class Board:
    def __init__(self):
        """Initialize a new chess board."""
        self.board = chess.Board()
        self.move_history = []
        
    def get_legal_moves(self):
        """Get all legal moves for the current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        """Make a move on the board.
        
        Args:
            move: A chess.Move object or a string in UCI format
            
        Returns:
            bool: True if move was legal and made, False otherwise
        """
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
            
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False
    
    def undo_move(self):
        """Undo the last move made."""
        if self.move_history:
            self.board.pop()
            return self.move_history.pop()
        return None
    
    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_game_result(self):
        """Get the result of the game if it's over.
        
        Returns:
            str: "1-0" for white win, "0-1" for black win, "1/2-1/2" for draw, None if game is not over
        """
        if not self.is_game_over():
            return None
        return self.board.outcome().result()
    
    def get_board_state(self):
        """Get the current board state as a numpy array.
        
        Returns:
            numpy.ndarray: 8x8 array representing the board state
        """
        board_array = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Convert piece to integer value
                value = piece.piece_type
                if piece.color == chess.BLACK:
                    value = -value
                board_array[square // 8][square % 8] = value
        return board_array
    
    def get_fen(self):
        """Get the current position in FEN format."""
        return self.board.fen()
    
    def set_fen(self, fen):
        """Set the board position from a FEN string."""
        self.board = chess.Board(fen)
        self.move_history = []
    
    def __str__(self):
        """Return a string representation of the board."""
        return str(self.board) 