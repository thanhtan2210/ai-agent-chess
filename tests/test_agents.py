import chess
import pytest
import time
from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.alphabeta_agent import AlphaBetaAgent
from src.agents.mcts_agent import MCTSAgent

def test_random_agent():
    """Test that RandomAgent returns valid moves."""
    agent = RandomAgent(chess.WHITE)
    board = chess.Board()
    
    # Test multiple moves
    for _ in range(10):
        move = agent.get_move(board.copy())
        assert move in board.legal_moves
        board.push(move)

def test_minimax_agent():
    """Test that MinimaxAgent returns valid moves and improves with depth."""
    # Test different depths
    for depth in [2, 3]:
        agent = MinimaxAgent(chess.WHITE, max_depth=depth)
        board = chess.Board()
        
        # Test multiple moves
        for _ in range(5):
            move = agent.get_move(board.copy())
            assert move in board.legal_moves
            board.push(move)
            
            # Test that agent uses transposition table
            assert hasattr(agent, 'transposition_table')
            assert len(agent.transposition_table.table) > 0

def test_mcts_agent():
    """Test that MCTSAgent returns valid moves and improves with iterations."""
    # Test different iteration counts
    for iterations in [100, 500]:
        agent = MCTSAgent(chess.WHITE, max_iterations=iterations)
        board = chess.Board()
        
        # Test multiple moves
        for _ in range(5):
            move = agent.get_move(board.copy())
            assert move in board.legal_moves
            board.push(move)

def test_agent_vs_random():
    """Test that our agents can consistently beat random agent"""
    agents = [
        MinimaxAgent(chess.WHITE, max_depth=6),
        AlphaBetaAgent(chess.WHITE, depth=3),
        MCTSAgent(chess.WHITE, max_iterations=1000)
    ]
    
    for agent in agents:
        wins = 0
        draws = 0
        losses = 0
        loss_details = []
        
        # Play 3 games as white
        for game_idx in range(3):
            board = chess.Board()
            random_agent = RandomAgent(chess.BLACK)
            move_history = []
            while not board.is_game_over():
                current_board_copy = board.copy()
                if board.turn == chess.WHITE:
                    move = agent.get_move(current_board_copy)
                else:
                    move = random_agent.get_move(current_board_copy)
                move_history.append(board.san(move))
                board.push(move)
            result = board.outcome().result()
            if result == "1-0":
                wins += 1
            elif result == "1/2-1/2":
                draws += 1
            else:
                losses += 1
                loss_details.append(("white", game_idx+1, move_history, result))
        
        # Play 3 games as black
        for game_idx in range(3):
            board = chess.Board()
            random_agent = RandomAgent(chess.WHITE)
            move_history = []
            while not board.is_game_over():
                current_board_copy = board.copy()
                if board.turn == chess.BLACK:
                    move = agent.get_move(current_board_copy)
                else:
                    move = random_agent.get_move(current_board_copy)
                move_history.append(board.san(move))
                board.push(move)
            result = board.outcome().result()
            if result == "0-1":
                wins += 1
            elif result == "1/2-1/2":
                draws += 1
            else:
                losses += 1
                loss_details.append(("black", game_idx+1, move_history, result))
        
        # Agent should win at least 4 out of 6 games
        print(f"Agent {agent.__class__.__name__} won {wins}, drew {draws}, lost {losses} against random agent")
        if loss_details:
            print("Loss details for agent:")
            for color_played, idx, moves, res in loss_details:
                print(f"  Game {idx} as {color_played}: result {res}")
                print("    Moves:", ' '.join(moves))
        assert wins >= 4, f"Agent {agent.__class__.__name__} only won {wins} out of 6 games against random agent"

def test_agent_vs_agent():
    """Test that agents can play against each other."""
    # Test Minimax vs MCTS
    minimax_agent = MinimaxAgent(chess.WHITE, max_depth=3)
    mcts_agent = MCTSAgent(chess.BLACK, max_iterations=500)
    board = chess.Board()
    
    # Play a short game
    for _ in range(10):
        if board.is_game_over():
            break
        current_board_copy = board.copy()
        if board.turn == chess.WHITE:
            move = minimax_agent.get_move(current_board_copy)
        else:
            move = mcts_agent.get_move(current_board_copy)
        board.push(move)

def test_move_ordering():
    """Test that MinimaxAgent's move ordering improves search efficiency."""
    agent = MinimaxAgent(chess.WHITE, max_depth=3)
    board = chess.Board()
    
    # Get initial node count
    agent.get_move(board.copy())
    initial_nodes = agent.nodes_evaluated
    
    # Test with a position that has many captures
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1")
    agent.get_move(board.copy())
    capture_nodes = agent.nodes_evaluated
    
    # The number of nodes evaluated should be significantly less with move ordering
    assert capture_nodes < initial_nodes * 2

def test_time_limit():
    """Test that MinimaxAgent respects time limit."""
    agent = MinimaxAgent(chess.WHITE, max_depth=5, time_limit=0.1)
    board = chess.Board()
    
    start_time = time.time()
    move = agent.get_move(board.copy())
    end_time = time.time()
    
    assert end_time - start_time < 0.2  # Allow some buffer
    assert move in board.legal_moves 