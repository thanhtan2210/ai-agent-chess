import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from src.agents.base_agent import BaseAgent
from src.game.board import Board

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 8x8x12 (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DeepLearningAgent(BaseAgent):
    def __init__(self, color=chess.WHITE, model_path=None, batch_size=32):
        super().__init__(color)
        self.model = ChessNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.batch_size = batch_size
        self.move_cache = {}  # Cache for move evaluations
    
    def board_to_tensor(self, board):
        """Convert a chess board to a tensor representation.
        
        Args:
            board: chess.Board object
            
        Returns:
            torch.Tensor: 8x8x12 tensor representing the board state
        """
        tensor = torch.zeros((8, 8, 12))
        
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)  # Convert to chess square index
                piece = board.piece_at(square)
                if piece:
                    # Calculate the channel index based on piece type and color
                    piece_type = piece.piece_type - 1  # 0-5 for piece types
                    color_offset = 0 if piece.color == chess.WHITE else 6
                    channel = piece_type + color_offset
                    tensor[row, col, channel] = 1
                
        return tensor
    
    def evaluate_positions_batch(self, boards):
        """Evaluate multiple positions in a batch."""
        tensors = [self.board_to_tensor(board) for board in boards]
        # Reshape tensors to [batch_size, channels, height, width] format
        tensors = [t.permute(2, 0, 1).unsqueeze(0) for t in tensors]
        batch = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            evaluations = self.model(batch)
            # Ensure we return a list of scores
            return evaluations.squeeze().tolist() if evaluations.numel() > 1 else [evaluations.item()]
    
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Process moves in batches
        best_score = float('-inf')
        best_move = None
        
        for i in range(0, len(legal_moves), self.batch_size):
            batch_moves = legal_moves[i:i + self.batch_size]
            batch_boards = []
            
            for move in batch_moves:
                # Check cache first
                move_key = (move.from_square, move.to_square, move.promotion)
                if move_key in self.move_cache:
                    score = self.move_cache[move_key]
                    if score > best_score:
                        best_score = score
                        best_move = move
                    continue
                
                # Make move and evaluate
                board.push(move)
                batch_boards.append(board.copy())
                board.pop()
            
            if batch_boards:
                scores = self.evaluate_positions_batch(batch_boards)
                
                for move, score in zip(batch_moves, scores):
                    move_key = (move.from_square, move.to_square, move.promotion)
                    self.move_cache[move_key] = score
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
        
        return best_move
    
    def get_name(self):
        return f"{super().get_name()}(batch_size={self.batch_size})" 