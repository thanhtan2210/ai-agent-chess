import chess
import math
import random
import time
from typing import Optional, Dict, List, Tuple
from src.game.rules import evaluate_position, PIECE_VALUES
from src.agents.base_agent import BaseAgent
from multiprocessing import Pool, cpu_count

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        self.move_scores = {}  # Cache for move scores
    
    def ucb1(self, exploration=1.41):
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def add_child(self, move):
        """Add a child node for the given move."""
        child_board = self.board.copy()
        child_board.push(move)
        child = MCTSNode(child_board, self, move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """Update node statistics."""
        self.visits += 1
        self.wins += result

def simulate_game(board):
    """Simulate a random game from the given position."""
    board = board.copy()
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if not moves:
            break
        move = random.choice(moves)
        board.push(move)
    
    if board.is_checkmate():
        return 1 if board.turn == chess.WHITE else 0
    return 0.5

class MCTSAgent(BaseAgent):
    def __init__(self, color, max_iterations=1000, exploration=1.41):
        """Initialize the MCTS agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
            max_iterations: Maximum number of MCTS iterations
            exploration: Exploration constant for UCB1
        """
        super().__init__(color)
        self.max_iterations = max_iterations
        self.exploration = exploration
        self.move_history = {}  # For move ordering
    
    def get_move(self, board):
        """Get the best move using MCTS.
        
        Args:
            board: chess.Board object representing the current position
            
        Returns:
            chess.Move: The best move found
        """
        root = MCTSNode(board)
        
        # Use parallel processing for simulations
        with Pool(processes=cpu_count()) as pool:
            for _ in range(self.max_iterations):
                node = self._select(root)
                if not node.board.is_game_over():
                    node = self._expand(node)
                # Run simulations in parallel
                results = pool.map(simulate_game, [node.board] * cpu_count())
                for result in results:
                    self._backpropagate(node, result)
        
        # Choose the best move based on visit count
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def _select(self, node):
        """Select a node to expand using UCB1."""
        while not node.untried_moves and node.children:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        return node
    
    def _expand(self, node):
        """Expand the tree by adding a new child node."""
        if node.untried_moves:
            # Use move ordering for expansion
            move = self._select_best_move(node)
            return node.add_child(move)
        return node
    
    def _select_best_move(self, node):
        """Select the best move to expand using move ordering."""
        moves = node.untried_moves
        if not moves:
            return None
            
        move_scores = []
        for move in moves:
            score = 0
            
            # Captures
            if node.board.is_capture(move):
                victim = node.board.piece_at(move.to_square)
                attacker = node.board.piece_at(move.from_square)
                if victim and attacker:
                    # Use piece values directly instead of evaluate_position
                    victim_value = PIECE_VALUES[victim.piece_type]
                    attacker_value = PIECE_VALUES[attacker.piece_type]
                    score += 1000 + victim_value - attacker_value/10
            
            # History heuristic
            move_key = (move.from_square, move.to_square, move.promotion)
            score += self.move_history.get(move_key, 0)
            
            # Checks
            node.board.push(move)
            if node.board.is_check():
                score += 800
            node.board.pop()
            
            move_scores.append((score, move))
        
        # Sort moves by score and select randomly from top 3
        move_scores.sort(key=lambda x: x[0], reverse=True)
        top_moves = [move for _, move in move_scores[:3]]
        return random.choice(top_moves)
    
    def _simulate(self, node):
        """Simulate a random game from the given node."""
        board = node.board.copy()
        while not board.is_game_over():
            moves = list(board.legal_moves)
            if not moves:
                break
            move = random.choice(moves)
            board.push(move)
        
        # Return 1 for win, 0.5 for draw, 0 for loss
        if board.is_checkmate():
            return 0 if board.turn == self.color else 1
        return 0.5
    
    def _backpropagate(self, node, result):
        """Backpropagate the simulation result up the tree."""
        while node:
            node.update(result)
            node = node.parent
            if node:
                result = 1 - result  # Invert result for opponent
    
    def get_name(self):
        """Get the name of the agent with its parameters."""
        return f"{super().get_name()}(iterations={self.max_iterations})" 