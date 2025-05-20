# src/compare_agents.py
import chess
from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.alphabeta_agent import AlphaBetaAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.deep_learning_agent import DeepLearningAgent

def play_game(agent1, agent2, max_moves=100):
    board = chess.Board()
    moves = 0
    
    while not board.is_game_over() and moves < max_moves:
        if board.turn == chess.WHITE:
            move = agent1.get_move(board)
        else:
            move = agent2.get_move(board)
        board.push(move)
        moves += 1
    
    return board.outcome().result() if board.is_game_over() else "1/2-1/2"

def compare_agents(num_games=10):
    agents = {
        "Random": RandomAgent(chess.WHITE),
        "Minimax (d=2)": MinimaxAgent(chess.WHITE, max_depth=2),
        "Minimax (d=3)": MinimaxAgent(chess.WHITE, max_depth=3),
        "AlphaBeta (d=2)": AlphaBetaAgent(chess.WHITE, depth=2),
        "AlphaBeta (d=3)": AlphaBetaAgent(chess.WHITE, depth=3),
        "MCTS (100)": MCTSAgent(chess.WHITE, max_iterations=100),
        "MCTS (1000)": MCTSAgent(chess.WHITE, max_iterations=1000),
        "Deep Learning": DeepLearningAgent(chess.WHITE, model_path="models/chess_model.pth")
    }
    
    results = {}
    for name1, agent1 in agents.items():
        for name2, agent2 in agents.items():
            if name1 != name2:
                key = f"{name1} vs {name2}"
                if key not in results:
                    results[key] = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
                
                for _ in range(num_games):
                    result = play_game(agent1, agent2)
                    results[key][result] += 1
    
    return results

if __name__ == "__main__":
    results = compare_agents()
    for matchup, scores in results.items():
        print(f"\n{matchup}:")
        print(f"White wins: {scores['1-0']}")
        print(f"Black wins: {scores['0-1']}")
        print(f"Draws: {scores['1/2-1/2']}")