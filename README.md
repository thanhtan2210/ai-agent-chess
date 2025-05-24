# AI Chess Agent

A modern chess game implementation with multiple AI agents using different search algorithms and machine learning approaches. The project features a chess.com-like user interface and a comprehensive evaluation system for comparing agent strengths.

## Features

### Game Features
- Modern, minimal UI inspired by chess.com
- Complete chess rules implementation including special moves
- Interactive game controls (New Game, Undo, Resign)
- Move history and captured pieces display
- Flexible game setup (Player vs AI, AI vs AI)
- Agent selection for both sides
- Multiple difficulty levels

### AI Agents
1. **Random Agent** (Rating: ~500 Elo)
   - Makes random legal moves
   - Used as baseline for evaluation
   - Configurable preferences for captures and checks

2. **Minimax Agent** (Rating: 500-600 Elo)
   - Configurable search depth (2-3)
   - Time-limited search
   - Material-based evaluation

3. **MCTS Agent** (Rating: 550-650 Elo)
   - Monte Carlo Tree Search implementation
   - Configurable time limit (5s/10s)
   - UCB1 selection with exploration weight
   - Efficient board state management

4. **Deep Learning Agent** (Rating: 600-700 Elo)
   - Neural network based evaluation
   - Pre-trained model support
   - CPU/GPU acceleration

### Evaluation System
- True Elo rating system
- Round-robin tournaments
- Win rate analysis
- Move time tracking
- Strength classification:
  - 2500+ Elo: Grandmaster
  - 2000+ Elo: Expert
  - 1600+ Elo: Advanced
  - 1200+ Elo: Intermediate
  - 800+ Elo: Beginner
  - Below 800: Novice

## Requirements

- Python 3.8+
- Dependencies:
  ```
  pygame
  python-chess
  numpy
  torch (for deep learning agent)
  ```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-agent-chess.git
cd ai-agent-chess
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Play Chess
```bash
python -m src.main
```

### Evaluate Agents
```bash
python -m src.evaluate_agents
```

### Compare Agents
```bash
python -m src.compare_agents
```

### Train Deep Learning Model
```bash
python -m src.train_chess_model
```

## Project Structure

```
src/
├── agents/           # AI agent implementations
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── minimax_agent.py
│   ├── mcts_agent.py
│   └── deep_learning_agent.py
├── game/            # Core game logic
├── ui/              # Pygame interface
│   ├── game_ui.py
│   ├── board_ui.py
│   └── setup_dialog.py
├── utils/           # Helper functions
├── main.py          # Main game entry
├── evaluate_agents.py
└── compare_agents.py
```

## Development

### Adding New Agents
1. Create new agent class inheriting from `BaseAgent`
2. Implement required methods:
   - `get_move(board)`
   - `evaluate_position(board)`
3. Add agent to evaluation suite

### UI Customization
- Modify `src/ui/config.py` for colors and dimensions
- Edit `src/ui/game_ui.py` for game interface
- Update `src/ui/setup_dialog.py` for game setup

## Future Improvements

- [ ] Add opening book support
- [ ] Implement more sophisticated evaluation functions
- [ ] Add reinforcement learning capabilities
- [ ] Support for saving/loading games
- [ ] Network play support
- [ ] Enhanced UI animations
- [ ] More agent types (Negamax, Iterative Deepening)
- [ ] Improved deep learning model

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.