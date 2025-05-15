import chess
from .pieces import get_piece_value, get_piece_square_value

# Piece values for evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Position tables for piece-square evaluation
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

# Endgame king table (king should be more active in endgame)
KING_ENDGAME_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

# Combine tables into a dictionary
PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}

ENDGAME_TABLES = {
    chess.KING: KING_ENDGAME_TABLE
}

def evaluate_position(board):
    """Evaluate the current position.
    
    Args:
        board: A chess.Board object
        
    Returns:
        int: Score from white's perspective (positive is good for white)
    """
    if board.is_checkmate():
        return -10000 if board.turn == chess.WHITE else 10000
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    is_endgame_phase = is_endgame(board)
    
    # Material and position evaluation
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Get piece value and positional value
        value = PIECE_VALUES[piece.piece_type]
        
        # Use endgame table for king in endgame
        if is_endgame_phase and piece.piece_type == chess.KING:
            position_value = ENDGAME_TABLES[chess.KING][square]
        else:
            position_value = PIECE_TABLES[piece.piece_type][square]
        
        # Add or subtract based on piece color
        if piece.color == chess.WHITE:
            score += value + position_value
        else:
            score -= value + position_value
    
    # Mobility evaluation (number of legal moves)
    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))
    board.turn = chess.WHITE  # Reset turn
    
    # Adjust mobility weight based on game phase
    mobility_weight = 15 if is_endgame_phase else 10
    score += (white_moves - black_moves) * mobility_weight
    
    # Pawn structure evaluation
    score += evaluate_pawn_structure(board)
    
    # King safety evaluation
    score += evaluate_king_safety(board)
    
    # --- TỐI ƯU BỔ SUNG ---
    # 1. Thưởng lớn cho nhập thành
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is not None:
            if color == chess.WHITE:
                if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
                    pass  # Chưa nhập thành
                else:
                    # Đã nhập thành
                    score += 60
            else:
                if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
                    pass
                else:
                    score -= 60
    # 2. Phạt lớn cho vua ở trung tâm khi chưa nhập thành
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is not None:
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            in_center = (king_file in [3,4]) and (king_rank in [0,7])
            if color == chess.WHITE:
                if in_center and (board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE)):
                    score -= 50
            else:
                if in_center and (board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK)):
                    score += 50
    # 3. Thưởng kiểm soát trung tâm
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                score += 20
            else:
                score -= 20
    # 4. Thưởng phát triển quân nhẹ, phạt quân chưa phát triển
    for color in [chess.WHITE, chess.BLACK]:
        minor_pieces = [chess.BISHOP, chess.KNIGHT]
        home_rank = 0 if color == chess.WHITE else 7
        developed = 0
        undeveloped = 0
        for file in range(8):
            square = chess.square(file, home_rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type in minor_pieces and piece.color == color:
                undeveloped += 1
        # Đếm số quân nhẹ đã phát triển
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in minor_pieces and piece.color == color and chess.square_rank(square) != home_rank:
                developed += 1
        if color == chess.WHITE:
            score += 15 * developed
            score -= 10 * undeveloped
        else:
            score -= 15 * developed
            score += 10 * undeveloped
    # 5. Thưởng tốt được bảo vệ, phạt tốt bị tấn công
    for color in [chess.WHITE, chess.BLACK]:
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                attackers = board.attackers(not color, square)
                defenders = board.attackers(color, square)
                if len(defenders) > 0:
                    if color == chess.WHITE:
                        score += 8
                    else:
                        score -= 8
                if len(attackers) > 0:
                    if color == chess.WHITE:
                        score -= 12
                    else:
                        score += 12
    
    return score

def evaluate_king_safety(board):
    """Evaluate king safety for both sides.
    
    Args:
        board: A chess.Board object
        
    Returns:
        int: Score from white's perspective
    """
    score = 0
    
    # Evaluate pawn shield
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is None:  # Skip if king not found
            continue
            
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check pawn shield in front of king
        shield_score = 0
        for file_offset in [-1, 0, 1]:
            shield_file = king_file + file_offset
            if 0 <= shield_file < 8:
                # Look for pawns 1-2 ranks in front of king
                for rank_offset in [1, 2]:
                    shield_rank = king_rank + (1 if color == chess.WHITE else -1) * rank_offset
                    if 0 <= shield_rank < 8:
                        shield_square = chess.square(shield_file, shield_rank)
                        piece = board.piece_at(shield_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            shield_score += 20 if rank_offset == 1 else 10
        
        # Add or subtract based on color
        if color == chess.WHITE:
            score += shield_score
        else:
            score -= shield_score
    
    # Evaluate king tropism (distance of pieces to enemy king)
    for color in [chess.WHITE, chess.BLACK]:
        enemy_king = board.king(not color)
        if enemy_king is None:  # Skip if enemy king not found
            continue
            
        tropism_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Calculate Manhattan distance to enemy king
                distance = abs(chess.square_file(square) - chess.square_file(enemy_king)) + \
                          abs(chess.square_rank(square) - chess.square_rank(enemy_king))
                
                # Weight by piece type
                if piece.piece_type == chess.QUEEN:
                    tropism_score += 14 - distance
                elif piece.piece_type == chess.ROOK:
                    tropism_score += 7 - distance
                elif piece.piece_type == chess.KNIGHT:
                    tropism_score += 5 - distance
                elif piece.piece_type == chess.BISHOP:
                    tropism_score += 4 - distance
        
        # Add or subtract based on color
        if color == chess.WHITE:
            score += tropism_score
        else:
            score -= tropism_score
    
    return score

def evaluate_pawn_structure(board):
    """Evaluate the pawn structure.
    
    Args:
        board: A chess.Board object
        
    Returns:
        int: Score from white's perspective
    """
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
            score -= 35 * (white_pawns - 1)  # Increased penalty
        if black_pawns > 1:
            score += 35 * (black_pawns - 1)
    
    # Isolated and backward pawns
    for file in range(8):
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                is_isolated = True
                is_backward = True
                
                # Check adjacent files for isolated pawns
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        for adj_rank in range(8):
                            adj_square = chess.square(adj_file, adj_rank)
                            adj_piece = board.piece_at(adj_square)
                            if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == piece.color:
                                is_isolated = False
                                break
                
                # Check for backward pawns
                if piece.color == chess.WHITE:
                    # Check if there are friendly pawns behind
                    for back_rank in range(rank):
                        back_square = chess.square(file, back_rank)
                        back_piece = board.piece_at(back_square)
                        if back_piece and back_piece.piece_type == chess.PAWN and back_piece.color == chess.WHITE:
                            is_backward = False
                            break
                else:
                    # Check if there are friendly pawns behind
                    for back_rank in range(rank + 1, 8):
                        back_square = chess.square(file, back_rank)
                        back_piece = board.piece_at(back_square)
                        if back_piece and back_piece.piece_type == chess.PAWN and back_piece.color == chess.BLACK:
                            is_backward = False
                            break
                
                # Apply penalties
                if is_isolated:
                    if piece.color == chess.WHITE:
                        score -= 25  # Increased penalty
                    else:
                        score += 25
                if is_backward:
                    if piece.color == chess.WHITE:
                        score -= 20  # New penalty for backward pawns
                    else:
                        score += 20
    
    return score

def is_endgame(board):
    """Check if the current position is an endgame.
    
    Args:
        board: A chess.Board object
        
    Returns:
        bool: True if the position is an endgame
    """
    # Count pieces
    piece_count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
            piece_count += 1
    
    # Consider it an endgame if there are 6 or fewer pieces (excluding pawns and kings)
    return piece_count <= 6

def get_move_value(board, move):
    """Get the value of a move based on captures and piece development.
    
    Args:
        board: A chess.Board object
        move: A chess.Move object
        
    Returns:
        int: Value of the move
    """
    value = 0
    
    # Capture value
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            value += get_piece_value(captured_piece.piece_type) * 10
    
    # Piece development
    moving_piece = board.piece_at(move.from_square)
    if moving_piece:
        # Encourage center control
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        if move.to_square in center_squares:
            value += 50
        
        # Encourage piece development in opening
        if not is_endgame(board):
            if moving_piece.piece_type == chess.KNIGHT or moving_piece.piece_type == chess.BISHOP:
                if move.from_square in [chess.B1, chess.G1, chess.B8, chess.G8]:
                    value += 30
    
    return value 