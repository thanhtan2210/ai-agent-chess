import chess
import time
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import sys
from contextlib import contextmanager
import torch

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.deep_learning_agent import DeepLearningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SHOW_MOVES = True  # Toggle để bật/tắt hiển thị nước đi

# Elo parameters
INITIAL_ELO = 500
K_FACTOR = 32

# Add constant at the top of the file, after imports
MAX_MOVES_PER_GAME = 200

@dataclass
class GameResult:
    wins: int
    draws: int 
    losses: int
    time: float
    moves: int

@dataclass
class AgentEvaluation:
    rating: float
    win_rate: float
    avg_time: float
    avg_moves: float
    strength: str
    results: GameResult

# Tạo context manager để tạm thời tắt output
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class AgentEvaluator:
    def __init__(self, num_games: int = 20, parallel: bool = True, show_moves: bool = False):
        self.num_games = num_games
        self.parallel = parallel
        self.show_moves = show_moves
        self.strengths = {
            8: "Expert",
            6: "Advanced", 
            4: "Intermediate",
            0: "Beginner"
        }

    def evaluate_single_game(self, params: Tuple[BaseAgent, BaseAgent, chess.Color]) -> GameResult:
        agent, opponent, color = params
        board = chess.Board()
        start_time = time.time()
        moves = 0

        try:
            with suppress_output():
                while not board.is_game_over() and moves < MAX_MOVES_PER_GAME:  # Maximum 200 moves
                    current_agent = agent if board.turn == color else opponent
                    move = current_agent.get_move(board)
                    board.push(move)
                    moves += 1
            
            game_time = time.time() - start_time
            result = board.outcome().result() if board.is_game_over() else "1/2-1/2"
            print(f"    Game finished in {game_time:.2f}s, result: {result}")

            if color == chess.WHITE:
                return GameResult(
                    wins = 1 if result == "1-0" else 0,
                    draws = 1 if result == "1/2-1/2" else 0,
                    losses = 1 if result == "0-1" else 0,
                    time = game_time,
                    moves = moves
                )
            else:
                return GameResult(
                    wins = 1 if result == "0-1" else 0,
                    draws = 1 if result == "1/2-1/2" else 0,
                    losses = 1 if result == "1-0" else 0,
                    time = game_time,
                    moves = moves
                )

        except Exception as e:
            logging.error(f"Error in game evaluation: {str(e)}")
            return GameResult(0, 0, 0, 0, 0)

    def evaluate_agent(self, agent: BaseAgent, opponent: Optional[BaseAgent] = None) -> AgentEvaluation:
        if opponent is None:
            opponent = RandomAgent(chess.WHITE, prefer_captures=True, prefer_checks=True)

        game_args = []
        for _ in range(self.num_games // 2):
            game_args.extend([
                (agent, opponent, chess.WHITE),
                (agent, opponent, chess.BLACK)
            ])

        results = []
        elo_agent = 500
        elo_random = 500
        K = 32
        
        total_games = len(game_args)
        for game_idx, args in enumerate(game_args):
            progress = (game_idx + 1) / total_games * 100
            print(f"\rProgress: {progress:.1f}% ({game_idx + 1}/{total_games})", end="", flush=True)
            
            r_agent, r_random = elo_agent, elo_random
            agent_inst, opponent_inst, color = args
            board = chess.Board()
            moves = 0
            start_time = time.time()
            
            try:
                while not board.is_game_over() and moves < MAX_MOVES_PER_GAME:
                    current_agent = agent_inst if board.turn == color else opponent_inst
                    move = current_agent.get_move(board)
                    board.push(move)
                    moves += 1
                        
                game_time = time.time() - start_time
                result = board.outcome().result() if board.is_game_over() else "1/2-1/2"
                
                if color == chess.WHITE:
                    if result == "1-0":
                        score_agent, score_random = 1, 0
                    elif result == "0-1":
                        score_agent, score_random = 0, 1
                    else:
                        score_agent = score_random = 0.5
                else:
                    if result == "0-1":
                        score_agent, score_random = 1, 0
                    elif result == "1-0":
                        score_agent, score_random = 0, 1
                    else:
                        score_agent = score_random = 0.5
                        
                # Update Elo
                exp_agent = 1 / (1 + 10 ** ((r_random - r_agent) / 400))
                exp_random = 1 / (1 + 10 ** ((r_agent - r_random) / 400))
                elo_agent = r_agent + K * (score_agent - exp_agent)
                elo_random = r_random + K * (score_random - exp_random)
                
                # Save results
                results.append(GameResult(
                    wins = 1 if score_agent == 1 else 0,
                    draws = 1 if score_agent == 0.5 else 0,
                    losses = 1 if score_agent == 0 else 0,
                    time = game_time,
                    moves = moves
                ))
                
            except KeyboardInterrupt:
                print("\nGame interrupted!")
                break
            except Exception as e:
                print(f"\nError in game: {str(e)}")
                continue
            
        print()  # New line after progress bar
        
        if not results:  # If no games completed successfully
            return AgentEvaluation(
                rating=500.0,
                win_rate=0.0,
                avg_time=0.0,
                avg_moves=0.0,
                strength="Unrated",
                results=GameResult(0, 0, 0, 0, 0)
            )
        
        total = GameResult(
            sum(r.wins for r in results),
            sum(r.draws for r in results),
            sum(r.losses for r in results),
            sum(r.time for r in results),
            sum(r.moves for r in results)
        )
        
        win_rate = total.wins / len(results)  # Use actual number of completed games
        avg_time = total.time / len(results)
        avg_moves = total.moves / len(results)
        rating = elo_agent
        
        return AgentEvaluation(
            rating=round(rating, 1),
            win_rate=round(win_rate * 100, 1),
            avg_time=round(avg_time, 2),
            avg_moves=round(avg_moves, 1),
            strength=self._get_strength(rating),
            results=total
        )

    def _calculate_rating(self, win_rate: float, draw_rate: float, avg_time: float, avg_moves: float) -> float:
        base_rating = (win_rate * 10) + (draw_rate * 5)
        
        # Time penalty/bonus
        time_multiplier = 1.0
        if avg_time < 0.1:
            time_multiplier = 0.8  # Penalty for too fast moves
        elif avg_time > 2.0:
            time_multiplier = 1.2  # Bonus for thoughtful moves
            
        # Move length bonus
        move_bonus = min(avg_moves / 50, 1.0)  # Bonus for longer games
        
        return base_rating * time_multiplier * (1 + move_bonus)

    def _get_strength(self, rating: float) -> str:
        if rating >= 2500:
            return "Grandmaster (GM)"
        elif rating >= 2400:
            return "International Master (IM)"
        elif rating >= 2300:
            return "FIDE Master (FM)"
        elif rating >= 2200:
            return "Candidate Master (CM)"
        elif rating >= 2000:
            return "Expert"
        elif rating >= 1800:
            return "Class A"
        elif rating >= 1600:
            return "Class B"
        elif rating >= 1400:
            return "Class C"
        elif rating >= 1200:
            return "Class D"
        elif rating >= 1000:
            return "Intermediate"
        elif rating >= 900:
            return "Advanced Novice"
        elif rating >= 800:
            return "Novice"
        elif rating >= 700:
            return "Advanced Beginner"
        elif rating >= 600:
            return "Beginner"
        elif rating >= 500:
            return "Novice Learner"
        elif rating >= 400:
            return "Learner"
        elif rating >= 300:
            return "Newcomer"
        elif rating >= 200:
            return "Starter"
        elif rating >= 100:
            return "Absolute Beginner"
        else:
            return "Unrated/New"

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, score_b, k=K_FACTOR):
    exp_a = expected_score(rating_a, rating_b)
    exp_b = expected_score(rating_b, rating_a)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * (score_b - exp_b)
    return new_a, new_b

def main():
    print("Evaluating chess agents...\n")
    # Reduce number of games for testing
    evaluator = AgentEvaluator(num_games=4)  # Changed from 10 to 4
    # Tắt logging cho thư viện chess
    logging.getLogger('chess').setLevel(logging.WARNING)
    
    # Test only a subset of agents first
    agents = [
        ('Random', RandomAgent(chess.WHITE, prefer_captures=True, prefer_checks=True)),
        ('Minimax-2', MinimaxAgent(chess.WHITE, max_depth=2, time_limit=5.0)),
        ('MCTS-5s', MCTSAgent(chess.WHITE, max_time=5.0, exploration_weight=1.41)),
    ]
    
    results: Dict[str, AgentEvaluation] = {}
    for name, agent in agents:
        print(f"\nTesting {name}...", end=" ", flush=True)
        try:
            eval_result = evaluator.evaluate_agent(agent)
            results[name] = eval_result
            print("Done!")
            print(f"Rating: {eval_result.rating:.1f}({eval_result.strength})")
            print(f"Win rate: {eval_result.win_rate:.1f}%")
            print(f"Average time per move: {eval_result.avg_time:.2f}s")
            print(f"Average moves per game: {eval_result.avg_moves:.1f}")
            print(f"  Tổng kết với Random agent: Thắng: {eval_result.results.wins}, Hòa: {eval_result.results.draws}, Thua: {eval_result.results.losses}")
            print()
        except Exception as e:
            print(f"Failed! ({str(e)})")
            logging.error(f"Error evaluating {name}: {str(e)}", exc_info=True)
    # --- Bổ sung đấu vòng tròn Elo tổng hợp ---
    print("\nĐấu vòng tròn Elo tổng hợp giữa tất cả các agent...")
    agent_classes = [
        ("Random", lambda: RandomAgent(chess.WHITE, prefer_captures=True, prefer_checks=True)),
        ("Minimax-2", lambda: MinimaxAgent(chess.WHITE, max_depth=2, time_limit=5.0)),
        ("Minimax-3", lambda: MinimaxAgent(chess.WHITE, max_depth=3, time_limit=10.0)),
        ("MCTS-5s", lambda: MCTSAgent(chess.WHITE, max_time=5.0, exploration_weight=1.41)),
        ("MCTS-10s", lambda: MCTSAgent(chess.WHITE, max_time=10.0, exploration_weight=1.41)),
        ("Deep Learning", lambda: DeepLearningAgent(
            chess.WHITE,
            model_path='models/chess_model.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ))
    ]
    agent_names = [name for name, _ in agent_classes]
    elos = {name: INITIAL_ELO for name in agent_names}
    num_games = 4  # Số ván mỗi cặp (2 trắng, 2 đen)
    for i, (name1, agent_fn1) in enumerate(agent_classes):
        for j, (name2, agent_fn2) in enumerate(agent_classes):
            if i >= j:
                continue
            for game_idx in range(num_games):
                # Alternate colors
                if game_idx % 2 == 0:
                    agent1 = agent_fn1()
                    agent2 = agent_fn2()
                    n1, n2 = name1, name2
                else:
                    agent1 = agent_fn2()
                    agent2 = agent_fn1()
                    n1, n2 = name2, name1
                board = chess.Board()
                moves = 0
                while not board.is_game_over() and moves < 200:
                    if board.turn == chess.WHITE:
                        move = agent1.get_move(board)
                    else:
                        move = agent2.get_move(board)
                    board.push(move)
                    moves += 1
                result = board.outcome().result() if board.is_game_over() else "1/2-1/2"
                if result == "1-0":
                    score1, score2 = 1, 0
                elif result == "0-1":
                    score1, score2 = 0, 1
                else:
                    score1 = score2 = 0.5
                elos[n1], elos[n2] = update_elo(elos[n1], elos[n2], score1, score2)
    print("\nBảng Elo tổng hợp sau đấu vòng tròn:")
    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    for name, elo in sorted_elos:
        print(f"{name}: {elo:.1f}")

if __name__ == "__main__":
    main()