import chess

def test_promotion():
    # Tốt trắng ở b7, quân đen ở a8
    board = chess.Board("b7/8/8/8/8/8/8/8 w - - 0 1")
    board.set_piece_at(chess.B7, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.ROOK, chess.BLACK))
    print("Các nước đi hợp lệ:")
    for m in board.legal_moves:
        print(m.uci())
    # Tốt trắng ăn a8 và phong hậu
    move = chess.Move.from_uci("b7a8q")
    assert move in board.legal_moves, "Move promotion ăn quân phải hợp lệ"
    board.push(move)
    piece = board.piece_at(chess.A8)
    print("Sau khi phong tốt ăn quân:", piece)
    assert piece is not None and piece.piece_type == chess.QUEEN and piece.color == chess.WHITE, "Tốt phải biến thành Hậu trắng"
    queen_moves = [m for m in board.legal_moves if m.from_square == chess.A8]
    print("Các nước đi của quân vừa phong:", [board.san(m) for m in queen_moves])
    assert any(m.to_square == chess.A1 for m in queen_moves), "Hậu phải có thể đi dọc đến a1"
    print("Test promotion thành công!")

if __name__ == "__main__":
    test_promotion() 