# src/compare_agents.py
import chess
import time
from typing import Dict, Optional
from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.deep_learning_agent import DeepLearningAgent
import torch

# Elo parameters
INITIAL_ELO = 500
K_FACTOR = 32

def play_game(agent1, agent2, max_moves=100, show_moves=False):
    """Play a game between two agents.
    
    Args:
        agent1: First agent (plays as White)
        agent2: Second agent (plays as Black)
        max_moves: Maximum number of moves before draw
        show_moves: Whether to print moves during the game
        
    Returns:
        str: Game result ("1-0", "0-1", or "1/2-1/2")
    """
    board = chess.Board()
    moves = 0
    start_time = time.time()
    
    while not board.is_game_over() and moves < max_moves:
        if board.turn == chess.WHITE:
            move = agent1.get_move(board)
        else:
            move = agent2.get_move(board)
        
        board.push(move)
        moves += 1
    
    game_time = time.time() - start_time
    result = board.outcome().result() if board.is_game_over() else "1/2-1/2"
    print(f"    Game finished in {game_time:.2f}s, result: {result}")
    return result

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, score_b, k=K_FACTOR):
    exp_a = expected_score(rating_a, rating_b)
    exp_b = expected_score(rating_b, rating_a)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * (score_b - exp_b)
    return new_a, new_b

def compare_agents(num_games=2, show_moves=False):
    """Compare different chess agents against each other.
    
    Args:
        num_games: Number of games to play between each pair of agents
        show_moves: Whether to show moves during games
    """
    agent_classes = {
        "Random": lambda: RandomAgent(chess.WHITE, prefer_captures=True, prefer_checks=True),
        "Minimax (d=2)": lambda: MinimaxAgent(chess.WHITE, max_depth=2, time_limit=2.0),
        "Minimax (d=3)": lambda: MinimaxAgent(chess.WHITE, max_depth=3, time_limit=2.0),
        "MCTS (1s)": lambda: MCTSAgent(chess.WHITE, max_time=1.0, exploration_weight=1.41),
        "Deep Learning": lambda: DeepLearningAgent(chess.WHITE, model_path='models/chess_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu')
    }
    agent_names = list(agent_classes.keys())
    elos = {name: INITIAL_ELO for name in agent_names}
    results = {f"{a} vs {b}": {"1-0": 0, "0-1": 0, "1/2-1/2": 0} for a in agent_names for b in agent_names if a != b}
    print(f"Starting Elo tournament with {len(agent_names)} agents, {num_games} games per matchup")
    for i, name1 in enumerate(agent_names):
        for j, name2 in enumerate(agent_names):
            if name1 == name2:
                continue
            for game_idx in range(num_games):
                # Alternate colors
                if game_idx % 2 == 0:
                    agent1 = agent_classes[name1]()
                    agent2 = agent_classes[name2]()
                else:
                    agent1 = agent_classes[name2]()
                    agent2 = agent_classes[name1]()
                    name1, name2 = name2, name1  # Swap for Elo update
                result = play_game(agent1, agent2, show_moves=show_moves)
                # Elo update
                if result == "1-0":
                    score1, score2 = 1, 0
                elif result == "0-1":
                    score1, score2 = 0, 1
                else:
                    score1 = score2 = 0.5
                elos[name1], elos[name2] = update_elo(elos[name1], elos[name2], score1, score2)
                # Record result for reporting
                results[f"{name1} vs {name2}"][result] += 1
    print("\nTournament Results (Win/Draw/Loss per matchup):")
    for matchup, scores in results.items():
        print(f"{matchup}: W: {scores['1-0']}, D: {scores['1/2-1/2']}, L: {scores['0-1']}")
    print("\nFinal Elo Ratings:")
    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    for name, elo in sorted_elos:
        print(f"{name}: {elo:.1f}")

if __name__ == "__main__":
    compare_agents(num_games=2, show_moves=False)