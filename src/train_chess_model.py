import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChessDataset(Dataset):
    def __init__(self, positions: List[torch.Tensor], evaluations: List[float]):
        self.positions = positions
        self.evaluations = evaluations
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]

class ChessModel(nn.Module):
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

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert chess board to tensor representation."""
    tensor = torch.zeros(12, 8, 8)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_idx = piece_map[piece.piece_type]
            if piece.color == chess.WHITE:
                tensor[piece_idx, rank, file] = 1
            else:
                tensor[piece_idx + 6, rank, file] = 1
    
    return tensor

def evaluate_position_heuristic(board: chess.Board) -> float:
    """Evaluate position using material count and piece-square tables."""
    if board.is_checkmate():
        return -100 if board.turn else 100
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
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
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    # Add mobility bonus
    mobility = len(list(board.legal_moves))
    if board.turn == chess.WHITE:
        score += mobility * 0.1
    else:
        score -= mobility * 0.1
    
    return score

def generate_game_data() -> Tuple[List[torch.Tensor], List[float]]:
    """Generate training data from a single game."""
    board = chess.Board()
    positions = []
    evaluations = []
    
    while not board.is_game_over():
        # Add current position
        positions.append(board_to_tensor(board))
        evaluations.append(evaluate_position_heuristic(board))
        
        # Make a random move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    
    return positions, evaluations

def generate_training_data(num_games: int = 1000) -> Tuple[List[torch.Tensor], List[float]]:
    """Generate training data from multiple games in parallel."""
    all_positions = []
    all_evaluations = []
    
    logger.info(f"Starting data generation with {num_games} games...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(generate_game_data) for _ in range(num_games)]
        for future in tqdm(futures, desc="Generating games", unit="game"):
            positions, evaluations = future.result()
            all_positions.extend(positions)
            all_evaluations.extend(evaluations)
    
    end_time = time.time()
    logger.info(f"Data generation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Generated {len(all_positions)} positions")
    
    return all_positions, all_evaluations

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """Train the model with early stopping and learning rate scheduling."""
    logger.info(f"Starting training on {device}")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for batch_idx, (positions, evaluations) in enumerate(pbar):
            positions = positions.to(device)
            evaluations = evaluations.to(device)
            
            optimizer.zero_grad()
            outputs = model(positions)
            loss = criterion(outputs.squeeze(), evaluations)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/batch_count:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"New best model saved with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

def main():
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        
        # Generate training data
        logger.info("Generating training data...")
        positions, evaluations = generate_training_data(num_games=1000)
        
        # Convert to tensors
        positions = torch.stack(positions)
        evaluations = torch.tensor(evaluations, dtype=torch.float32)
        
        # Create dataset and dataloader
        dataset = ChessDataset(positions, evaluations)
        train_loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = ChessModel()
        
        # Train model
        logger.info("Starting model training...")
        train_model(model, train_loader)
        
        # Save final model
        torch.save(model.state_dict(), output_dir / "chess_model.pth")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 