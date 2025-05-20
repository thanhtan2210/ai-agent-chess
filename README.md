# AI Chess Agent

A chess game implementation with multiple AI agents using different search algorithms and machine learning approaches.

## Features

- Complete chess game implementation with GUI using Pygame
- Multiple AI agents with different playing strengths
- Support for all chess rules including special moves
- Game features: move history, chess clock, undo/redo
- Agent evaluation system
- Deep learning based agent

## AI Agents

The project implements several search algorithms and approaches for the chess AI:

### 1. Random Agent
- Makes completely random legal moves
- Used as a baseline for evaluating other agents
- Rating: 1-2/10

### 2. Minimax Agent
- Implements the minimax algorithm with configurable depth
- Uses a simple piece-value evaluation function
- Performance scales with search depth
- Rating: 2-4/10 depending on depth

### 3. Alpha-Beta Agent
- Implements minimax with alpha-beta pruning
- More efficient than basic minimax
- Can search deeper in the same time
- Rating: 3-5/10 depending on depth

### 4. Monte Carlo Tree Search Agent
- Implements MCTS with UCB1 selection
- Number of simulations configurable
- Good balance of exploration and exploitation
- Rating: 4-6/10 depending on simulation count

### 5. Deep Learning Agent
- Uses neural networks for move evaluation
- Trained on chess game data
- Can learn from experience
- Rating: 5-7/10 depending on training

## Search Algorithms

### Minimax
- Complete search of game tree to fixed depth
- Alternates between maximizing and minimizing players
- Uses evaluation function at leaf nodes
- Time complexity: O(b^d) where b is branching factor, d is depth

### Alpha-Beta Pruning
- Optimization of minimax
- Prunes branches that cannot affect final decision
- Reduces time complexity to O(b^(d/2)) in best case
- Maintains same optimality as minimax

### Monte Carlo Tree Search
- Four phases: Selection, Expansion, Simulation, Backpropagation
- Uses UCB1 formula for tree policy
- Asymptotically converges to minimax
- Can handle large branching factors well
- Time complexity depends on number of simulations

### Deep Learning
- Neural network based evaluation
- Can learn complex patterns and strategies
- Requires training data and computational resources
- Can improve over time with more training

## Evaluation

The project includes an evaluation system that:
- Rates agents on a 0-10 scale
- Measures win rates against random agent
- Considers move time in rating
- Classifies agents as Beginner/Intermediate/Advanced/Expert

## Requirements

- Python 3.8+
- pygame
- python-chess
- numpy
- tensorflow (for deep learning agent)

## Usage

1. Run the main game:
```bash
python -m src.main
```

2. Train the deep learning model:
```bash
python -m src.train_chess_model
```

3. Evaluate agents:
```bash
python -m src.evaluate_agents
```

4. Compare agents:
```bash
python -m src.compare_agents
```

5. Run tests:
```bash
python -m pytest tests/
```

## Project Structure

```
src/
├── agents/           # AI agent implementations
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── minimax_agent.py
│   ├── alpha_beta_agent.py
│   ├── mcts_agent.py
│   └── deep_learning_agent.py
├── game/            # Core game logic
├── ui/              # Pygame interface
├── utils/           # Helper functions
├── main.py          # Main game entry
├── train_chess_model.py # Deep learning model training
├── evaluate_agents.py # Agent evaluation
└── compare_agents.py # Agent comparison
tests/
├── test_agents.py   # Agent tests
└── test_promotion.py # Special move tests
```

## Future Improvements

- Implement more sophisticated evaluation functions
- Add opening book support
- Optimize search algorithms further
- Add more agent types (e.g., Negamax, Iterative Deepening)
- Improve UI and add more features
- Enhance deep learning model with more training data
- Add reinforcement learning capabilities