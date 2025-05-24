import chess
import unittest

class TestPromotion(unittest.TestCase):
    def test_basic_promotion(self):
        """Test basic pawn promotion."""
        board = chess.Board("7P/8/8/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("h7h8q")
        self.assertIn(move, board.legal_moves, "Basic promotion should be legal")
        board.push(move)
        piece = board.piece_at(chess.H8)
        self.assertIsNotNone(piece, "Piece should exist after promotion")
        self.assertEqual(piece.piece_type, chess.QUEEN, "Should promote to queen")
        self.assertEqual(piece.color, chess.WHITE, "Promoted piece should be white")

    def test_capture_promotion(self):
        """Test promotion with capture."""
        board = chess.Board("7P/8/8/8/8/8/8/8 w - - 0 1")
        board.set_piece_at(chess.G8, chess.Piece(chess.ROOK, chess.BLACK))
        move = chess.Move.from_uci("h7g8q")
        self.assertIn(move, board.legal_moves, "Capture promotion should be legal")
        board.push(move)
        piece = board.piece_at(chess.G8)
        self.assertIsNotNone(piece, "Piece should exist after capture promotion")
        self.assertEqual(piece.piece_type, chess.QUEEN, "Should promote to queen")
        self.assertEqual(piece.color, chess.WHITE, "Promoted piece should be white")

    def test_all_promotion_pieces(self):
        """Test promotion to all possible pieces."""
        board = chess.Board("7P/8/8/8/8/8/8/8 w - - 0 1")
        promotion_pieces = {
            'q': chess.QUEEN,
            'r': chess.ROOK,
            'b': chess.BISHOP,
            'n': chess.KNIGHT
        }
        
        for piece_code, piece_type in promotion_pieces.items():
            move = chess.Move.from_uci(f"h7h8{piece_code}")
            self.assertIn(move, board.legal_moves, f"Promotion to {piece_code} should be legal")
            board.push(move)
            piece = board.piece_at(chess.H8)
            self.assertIsNotNone(piece, f"Piece should exist after {piece_code} promotion")
            self.assertEqual(piece.piece_type, piece_type, f"Should promote to {piece_code}")
            self.assertEqual(piece.color, chess.WHITE, "Promoted piece should be white")
            board.pop()

    def test_promotion_moves(self):
        """Test that promoted piece has correct moves."""
        board = chess.Board("7P/8/8/8/8/8/8/8 w - - 0 1")
        board.push(chess.Move.from_uci("h7h8q"))
        
        # Queen should be able to move in all directions
        queen_moves = [m for m in board.legal_moves if m.from_square == chess.H8]
        self.assertGreater(len(queen_moves), 0, "Queen should have legal moves")
        
        # Test some specific moves
        test_moves = [
            chess.Move.from_uci("h8h1"),  # Vertical
            chess.Move.from_uci("h8a8"),  # Horizontal
            chess.Move.from_uci("h8a1"),  # Diagonal
        ]
        for move in test_moves:
            self.assertIn(move, queen_moves, f"Queen should be able to move {move.uci()}")

    def test_promotion_check(self):
        """Test promotion that gives check."""
        board = chess.Board("7P/8/8/8/8/8/8/k7 w - - 0 1")
        move = chess.Move.from_uci("h7h8q")
        self.assertIn(move, board.legal_moves, "Promotion with check should be legal")
        board.push(move)
        self.assertTrue(board.is_check(), "Promotion should give check")

if __name__ == "__main__":
    unittest.main() 