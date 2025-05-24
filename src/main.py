"""Main entry point for the chess game."""

import os
import sys
import chess
import pygame
from src.game.chess_game import ChessGame
from src.agents import (
    RandomAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    MCTSAgent,
    DeepLearningAgent
)
from src.ui import GameUI, COLORS, FONTS, FONT_SIZES
from src.ui.setup_dialog import select_difficulty_and_color

def main():
    """Main function to run the chess game."""
    # Select difficulty, color, ai_vs_ai, agent names
    depth, color, ai_vs_ai, agent_white_name, agent_black_name = select_difficulty_and_color()
    
    # Create game instance
    game = ChessGame()
    
    # Helper to get agent instance by name
    def get_agent(agent_name, color, depth):
        if agent_name == "MinimaxAgent":
            return MinimaxAgent(color, max_depth=depth)
        elif agent_name == "AlphaBetaAgent":
            return AlphaBetaAgent(color, depth=depth)
        elif agent_name == "MCTSAgent":
            return MCTSAgent(color, max_time=5.0)
        elif agent_name == "DeepLearningAgent":
            return DeepLearningAgent(color)
        elif agent_name == "RandomAgent":
            return RandomAgent(color)
        else:
            return None
    # Create agents based on selection
    if ai_vs_ai:
        white_agent = get_agent(agent_white_name, chess.WHITE, depth)
        black_agent = get_agent(agent_black_name, chess.BLACK, depth)
    elif color == 0:  # Player chose White
        white_agent = None  # Human player
        black_agent = get_agent(agent_black_name, chess.BLACK, depth)
    else:  # Player chose Black
        white_agent = get_agent(agent_white_name, chess.WHITE, depth)
        black_agent = None  # Human player
    # Create UI and run game
    ui = GameUI(game, white_agent, black_agent)
    ui.run()

if __name__ == "__main__":
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    main()