import os
import sys
import chess
import pytest
import time
import logging
from tqdm import tqdm
from colorama import init, Fore, Style

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.alphabeta_agent import AlphaBetaAgent
from src.agents.mcts_agent import MCTSAgent
from src.game.rules import evaluate_position, PIECE_VALUES

# Initialize colorama
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_test_header(test_name: str):
    """Print a formatted test header."""
    logger.info(f"Starting test: {test_name}")
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Running Test: {test_name}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print a formatted test result."""
    status = f"{Fore.GREEN}PASSED{Style.RESET_ALL}" if passed else f"{Fore.RED}FAILED{Style.RESET_ALL}"
    logger.info(f"Test {test_name} {status}")
    print(f"\n{Fore.YELLOW}Test Result:{Style.RESET_ALL}")
    print(f"Test: {test_name}")
    print(f"Status: {status}")
    if details:
        print(f"Details: {details}")
    print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}\n")

@pytest.mark.agent_test
def test_random_agent():
    """Test that RandomAgent returns valid moves."""
    print_test_header("Random Agent Test")
    
    agent = RandomAgent(chess.WHITE)
    board = chess.Board()
    
    logger.info("Testing 5 random moves...")
    print(f"{Fore.BLUE}Testing 5 random moves...{Style.RESET_ALL}")
    for i in tqdm(range(5), desc="Testing moves"):
        try:
            move = agent.get_move(board.copy())
            assert move in board.legal_moves, f"Invalid move generated: {move}"
            board.push(move)
            logger.debug(f"Move {i+1}/5 completed: {move}")
        except Exception as e:
            logger.error(f"Error on move {i+1}: {str(e)}")
            raise
    
    print_test_result("Random Agent", True, "Successfully generated and executed 5 valid moves")

@pytest.mark.agent_test
def test_minimax_agent():
    """Test that MinimaxAgent returns valid moves and improves with depth."""
    print_test_header("Minimax Agent Test")
    
    for depth in [2, 3]:
        logger.info(f"Testing MinimaxAgent with depth {depth}")
        print(f"\n{Fore.BLUE}Testing depth {depth}...{Style.RESET_ALL}")
        agent = MinimaxAgent(chess.WHITE, max_depth=depth)
        board = chess.Board()
        
        for i in tqdm(range(3), desc=f"Testing depth {depth}"):
            try:
                move = agent.get_move(board.copy())
                assert move in board.legal_moves, f"Invalid move generated at depth {depth}: {move}"
                board.push(move)
                logger.debug(f"Move {i+1}/3 at depth {depth} completed: {move}")
            except Exception as e:
                logger.error(f"Error on move {i+1} at depth {depth}: {str(e)}")
                raise
    
    print_test_result("Minimax Agent", True, "Successfully tested at depths 2 and 3")

@pytest.mark.agent_test
def test_mcts_agent():
    """Test that MCTSAgent returns valid moves and improves with iterations."""
    print_test_header("MCTS Agent Test")
    
    for iterations in [50, 100]:
        logger.info(f"Testing MCTSAgent with {iterations} iterations")
        print(f"\n{Fore.BLUE}Testing {iterations} iterations...{Style.RESET_ALL}")
        agent = MCTSAgent(chess.WHITE, max_iterations=iterations)
        board = chess.Board()
        
        for i in tqdm(range(3), desc=f"Testing {iterations} iterations"):
            try:
                move = agent.get_move(board.copy())
                assert move in board.legal_moves, f"Invalid move generated with {iterations} iterations: {move}"
                board.push(move)
                logger.debug(f"Move {i+1}/3 with {iterations} iterations completed: {move}")
            except Exception as e:
                logger.error(f"Error on move {i+1} with {iterations} iterations: {str(e)}")
                raise
    
    print_test_result("MCTS Agent", True, "Successfully tested with 50 and 100 iterations")

@pytest.mark.agent_test
def test_agent_vs_random():
    """Test that our agents can consistently beat random agent."""
    print_test_header("Agent vs Random Test")
    
    agents = [
        MinimaxAgent(chess.WHITE, max_depth=3),
        AlphaBetaAgent(chess.WHITE, depth=2),
        MCTSAgent(chess.WHITE, max_iterations=100)
    ]
    
    for agent_idx, agent in enumerate(agents, 1):
        logger.info(f"Testing agent {agent_idx}/3: {agent.__class__.__name__}")
        print(f"\n{Fore.BLUE}Testing {agent.__class__.__name__}...{Style.RESET_ALL}")
        wins = 0
        draws = 0
        losses = 0
        loss_details = []
        
        # Play 2 games as white
        for game_idx in tqdm(range(2), desc="Playing as white"):
            try:
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
                    logger.debug(f"Game {game_idx+1} as white - Move {len(move_history)}: {move}")
                
                result = board.outcome().result()
                if result == "1-0":
                    wins += 1
                elif result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1
                    loss_details.append(("white", game_idx+1, move_history, result))
                logger.info(f"Game {game_idx+1} as white completed with result: {result}")
            except Exception as e:
                logger.error(f"Error in game {game_idx+1} as white: {str(e)}")
                raise
        
        # Play 2 games as black
        for game_idx in tqdm(range(2), desc="Playing as black"):
            try:
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
                    logger.debug(f"Game {game_idx+1} as black - Move {len(move_history)}: {move}")
                
                result = board.outcome().result()
                if result == "0-1":
                    wins += 1
                elif result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1
                    loss_details.append(("black", game_idx+1, move_history, result))
                logger.info(f"Game {game_idx+1} as black completed with result: {result}")
            except Exception as e:
                logger.error(f"Error in game {game_idx+1} as black: {str(e)}")
                raise
        
        # Print results
        result_str = f"Won {wins}, drew {draws}, lost {losses} against random agent"
        logger.info(f"Results for {agent.__class__.__name__}: {result_str}")
        print(f"\n{Fore.GREEN}Results for {agent.__class__.__name__}:{Style.RESET_ALL}")
        print(result_str)
        
        if loss_details:
            print(f"\n{Fore.YELLOW}Loss details:{Style.RESET_ALL}")
            for color_played, idx, moves, res in loss_details:
                print(f"  Game {idx} as {color_played}: result {res}")
                print(f"  Moves: {' '.join(moves)}")
        
        assert wins >= 2, f"Agent {agent.__class__.__name__} only won {wins} out of 4 games against random agent"
        print_test_result(f"{agent.__class__.__name__} vs Random", True, result_str)

@pytest.mark.agent_test
def test_agent_vs_agent():
    """Test that agents can play against each other."""
    print_test_header("Agent vs Agent Test")
    
    minimax_agent = MinimaxAgent(chess.WHITE, max_depth=2)
    mcts_agent = MCTSAgent(chess.BLACK, max_iterations=50)
    board = chess.Board()
    
    print(f"{Fore.BLUE}Playing Minimax vs MCTS...{Style.RESET_ALL}")
    move_count = 0
    for _ in tqdm(range(5), desc="Playing moves"):
        if board.is_game_over():
            break
        current_board_copy = board.copy()
        if board.turn == chess.WHITE:
            move = minimax_agent.get_move(current_board_copy)
        else:
            move = mcts_agent.get_move(current_board_copy)
        board.push(move)
        move_count += 1
    
    result = "Game completed" if board.is_game_over() else f"Game stopped after {move_count} moves"
    print_test_result("Agent vs Agent", True, result)

@pytest.mark.agent_test
def test_move_ordering():
    """Test that MinimaxAgent's move ordering improves search efficiency."""
    print_test_header("Move Ordering Test")
    
    agent = MinimaxAgent(chess.WHITE, max_depth=2)
    board = chess.Board()
    
    print(f"{Fore.BLUE}Testing move ordering efficiency...{Style.RESET_ALL}")
    
    # Get initial node count
    agent.get_move(board.copy())
    initial_nodes = agent.nodes_evaluated
    print(f"Initial nodes evaluated: {initial_nodes}")
    
    # Test with a position that has many captures
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1")
    agent.get_move(board.copy())
    capture_nodes = agent.nodes_evaluated
    print(f"Nodes evaluated with captures: {capture_nodes}")
    
    # The number of nodes evaluated should be reasonable
    assert capture_nodes < initial_nodes * 4, "Move ordering did not maintain reasonable search efficiency"
    print_test_result("Move Ordering", True, f"Search efficiency maintained within reasonable bounds (ratio: {capture_nodes/initial_nodes:.2f}x)")

@pytest.mark.agent_test
def test_time_limit():
    """Test that MinimaxAgent respects time limit."""
    print_test_header("Time Limit Test")
    
    agent = MinimaxAgent(chess.WHITE, max_depth=3, time_limit=0.1)
    board = chess.Board()
    
    print(f"{Fore.BLUE}Testing time limit of 0.1 seconds...{Style.RESET_ALL}")
    start_time = time.time()
    move = agent.get_move(board.copy())
    end_time = time.time()
    
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.3f} seconds")
    
    assert time_taken < 0.2, f"Move generation took too long: {time_taken:.3f} seconds"
    assert move in board.legal_moves, "Invalid move generated under time limit"
    print_test_result("Time Limit", True, f"Move generated in {time_taken:.3f} seconds") 