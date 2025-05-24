import chess
import math
import time
import random
from typing import Optional, List, Dict
from src.agents.base_agent import BaseAgent
from src.game.rules import evaluate_position

class MCTSNode:
    def __init__(self, board: chess.Board, move: Optional[chess.Move] = None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        
    def ucb1(self, exploration_weight: float) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MCTSAgent(BaseAgent):
    def __init__(self, color: chess.Color, max_time: float = 5.0, exploration_weight: float = 1.41):
        """Initialize the MCTS agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
            max_time: Maximum time to search in seconds
            exploration_weight: Weight for exploration in UCB1 formula
        """
        super().__init__(color)
        self.max_time = max_time
        self.exploration_weight = exploration_weight
        self.name = "MCTS"
        print(f"MCTSAgent initialized. Max Time: {max_time}s, Exploration Weight: {exploration_weight}")
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move using MCTS."""
        start_time = time.time()
        root = MCTSNode(board)
        
        # Run MCTS until time limit
        while time.time() - start_time < self.max_time:
            node = self._select(root)
            if not node.board.is_game_over():
                result = self._rollout(node.board)
            else:
                result = self._evaluate_position(node.board)
            self._backpropagate(node, result)
            
            # Check if we've exceeded time limit
            if time.time() - start_time >= self.max_time:
                break
        
        # Select best move based on visit count
        if not root.children:
            return next(iter(board.legal_moves))
            
        best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return best_move
        
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1."""
        while node.children and not node.board.is_game_over():
            # If node is fully expanded, select best child
            if len(node.children) == len(list(node.board.legal_moves)):
                node = max(node.children.values(), key=lambda n: n.ucb1(self.exploration_weight))
            else:
                # Otherwise, expand the node
                return self._expand(node)
        return node
        
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding a child."""
        # Get all legal moves
        legal_moves = list(node.board.legal_moves)
        
        # Find moves that haven't been tried yet
        tried_moves = set(node.children.keys())
        untried_moves = [move for move in legal_moves if move not in tried_moves]
        
        if not untried_moves:
            return node
            
        # Select a random untried move
        move = random.choice(untried_moves)
        
        # Create new board state
        new_board = node.board.copy(stack=False)  # Don't copy move stack
        new_board.push(move)
        
        # Create new node
        child = MCTSNode(new_board, move, node)
        node.children[move] = child
        
        return child
        
    def _rollout(self, board: chess.Board) -> float:
        """Perform a rollout from the given position."""
        if board.is_game_over():
            if board.is_checkmate():
                return -1.0 if board.turn == self.color else 1.0
            return 0.0
            
        # Use simple evaluation for rollout
        return self._evaluate_position(board)
        
    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate the result up the tree."""
        while node is not None:
            node.visits += 1
            node.value += result
            result = -result  # Negamax backpropagation
            node = node.parent
            
    def _evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a position using material count and piece-square tables."""
        if board.is_checkmate():
            return -1.0 if board.turn == self.color else 1.0
            
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
            
        # Material values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.2,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_values[piece.piece_type]
                if piece.color == self.color:
                    score += value
                else:
                    score -= value
                    
        # Normalize score to [-1, 1]
        return max(min(score / 39, 1.0), -1.0)  # 39 is max possible material difference
        
    def get_name(self) -> str:
        """Get the name of the agent."""
        return f"MCTS (t={self.max_time:.1f}s, c={self.exploration_weight:.2f})" 