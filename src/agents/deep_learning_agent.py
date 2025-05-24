import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from src.agents.base_agent import BaseAgent
from src.game.board import Board
from typing import Optional, Dict, Tuple, List
from src.game.rules import evaluate_position

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 12 piece types * 64 squares = 768 features
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class DeepLearningAgent(BaseAgent):
    def __init__(self, color, model_path: Optional[str] = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the Deep Learning agent.
        
        Args:
            color: chess.WHITE or chess.BLACK
            model_path: Path to pretrained model weights (optional)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__(color)
        self.device = device
        self.model = ChessNet().to(device)
        self.name = "DeepLearning"
        
        if model_path:
            self.load_model(model_path)
        else:
            print("No model path provided. Using untrained model.")
            
        self.model.eval()  # Set to evaluation mode
        print(f"DeepLearningAgent initialized on {device}")
        
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor representation."""
        # Create 12 channels (6 piece types for each color)
        tensor = torch.zeros(12, 8, 8, device=self.device)
        
        # Map pieces to channels
        piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        # Fill tensor with piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                channel = piece_to_channel[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                tensor[channel, rank, file] = 1
                
        return tensor.unsqueeze(0)  # Add batch dimension
        
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using neural network."""
        with torch.no_grad():
            tensor = self.board_to_tensor(board)
            evaluation = self.model(tensor).item()
            return evaluation if board.turn == self.color else -evaluation
            
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move using neural network evaluation."""
        best_score = float('-inf')
        best_move = None
        
        # Evaluate all legal moves
        for move in board.legal_moves:
            board.push(move)
            score = self.evaluate_position(board)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                
        if best_move is None:
            # Fallback to first legal move if no evaluation
            best_move = next(iter(board.legal_moves))
            
        self.set_best_move(best_move)
        return best_move
        
    def save_model(self, path: str):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model weights from file."""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def get_name(self):
        """Get the name of the agent with its parameters."""
        return f"{self.name}(device={self.device})"
        
    def train_step(self, board: chess.Board, target_score: float, learning_rate: float = 0.001):
        """Perform one training step on a position.
        
        Args:
            board: Chess board position
            target_score: Target evaluation score (-1 to 1)
            learning_rate: Learning rate for optimization
        """
        self.model.train()  # Set to training mode
        
        # Convert board to tensor
        input_tensor = self.board_to_tensor(board)
        target_tensor = torch.tensor([target_score], device=self.device)
        
        # Forward pass
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        
        output = self.model(input_tensor)
        loss = F.mse_loss(output, target_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item() 