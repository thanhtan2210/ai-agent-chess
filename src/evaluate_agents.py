import chess
import time
from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.alphabeta_agent import AlphaBetaAgent
from src.agents.mcts_agent import MCTSAgent
from multiprocessing import Pool, cpu_count

def evaluate_single_game(args):
    """Evaluate a single game between two agents"""
    agent, opponent, color = args
    board = chess.Board()
    start_time = time.time()
    
    while not board.is_game_over():
        if board.turn == color:
            move = agent.get_move(board)
        else:
            move = opponent.get_move(board)
        board.push(move)
    
    game_time = time.time() - start_time
    result = board.outcome().result()
    
    if color == chess.WHITE:
        if result == "1-0":
            return (1, 0, 0, game_time)  # win
        elif result == "1/2-1/2":
            return (0, 1, 0, game_time)  # draw
        else:
            return (0, 0, 1, game_time)  # loss
    else:
        if result == "0-1":
            return (1, 0, 0, game_time)  # win
        elif result == "1/2-1/2":
            return (0, 1, 0, game_time)  # draw
        else:
            return (0, 0, 1, game_time)  # loss

def evaluate_agent_strength(agent, num_games=20, opponent=None):
    """Evaluate agent strength against a given opponent or random agent using parallel processing"""
    if opponent is None:
        opponent = RandomAgent(chess.WHITE)
    
    # Create game arguments for parallel processing
    game_args = []
    for _ in range(num_games // 2):
        game_args.append((agent, opponent, chess.WHITE))
        game_args.append((agent, opponent, chess.BLACK))
    
    # Use parallel processing to evaluate games
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(evaluate_single_game, game_args)
    
    # Aggregate results
    wins = sum(r[0] for r in results)
    draws = sum(r[1] for r in results)
    losses = sum(r[2] for r in results)
    total_time = sum(r[3] for r in results)
    
    # Calculate rating (0-10 scale)
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    
    # Base rating on win rate and draw rate
    rating = (win_rate * 10) + (draw_rate * 5)
    
    # Adjust rating based on average move time
    avg_time = total_time / num_games
    if avg_time < 0.1:  # Very fast moves
        rating *= 0.8
    elif avg_time > 2.0:  # Very slow moves
        rating *= 1.2
    
    return {
        'rating': round(rating, 1),
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': round(win_rate * 100, 1),
        'avg_time': round(avg_time, 2)
    }

def main():
    # Create agents with different parameters
    agents = [
        ('Random Agent', RandomAgent(chess.WHITE)),
        ('Minimax (depth=2)', MinimaxAgent(chess.WHITE, max_depth=2)),
        ('Minimax (depth=3)', MinimaxAgent(chess.WHITE, max_depth=3)),
        ('AlphaBeta (depth=2)', AlphaBetaAgent(chess.WHITE, depth=2)),
        ('AlphaBeta (depth=3)', AlphaBetaAgent(chess.WHITE, depth=3)),
        ('MCTS (100 sims)', MCTSAgent(chess.WHITE, max_iterations=100)),
        ('MCTS (1000 sims)', MCTSAgent(chess.WHITE, max_iterations=1000))
    ]
    
    print("\nAgent Strength Evaluation")
    print("=" * 50)
    
    # Evaluate each agent
    for name, agent in agents:
        print(f"\nEvaluating {name}...")
        results = evaluate_agent_strength(agent)
        
        print(f"Rating: {results['rating']}/10")
        print(f"Record: {results['wins']}W {results['draws']}D {results['losses']}L")
        print(f"Win Rate: {results['win_rate']}%")
        print(f"Average Time per Game: {results['avg_time']}s")
        
        # Additional analysis
        if results['rating'] >= 8:
            print("Strength: Expert")
        elif results['rating'] >= 6:
            print("Strength: Advanced")
        elif results['rating'] >= 4:
            print("Strength: Intermediate")
        else:
            print("Strength: Beginner")
    
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main() 