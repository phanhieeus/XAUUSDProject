import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models.time_aware_transformer import TimeAwareTransformer
from data.dataset import XAUUSDDataset
from config import *
import os
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, timestamps, labels) in enumerate(train_loader):
        features, timestamps, labels = features.to(device), timestamps.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features, timestamps)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Add regularization for time decay
        time_decay_reg = 0
        for block in model.transformer_blocks:
            time_decay_reg += torch.mean(block.attention.time_decay)
        loss += TIME_DECAY_REG * time_decay_reg
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, timestamps, labels in val_loader:
            features, timestamps, labels = features.to(device), timestamps.to(device), labels.to(device)
            outputs = model(features, timestamps)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    # Thiết lập device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Tạo thư mục checkpoints nếu chưa tồn tại
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_dataset = XAUUSDDataset(TRAIN_PATH)
    val_dataset = XAUUSDDataset(VAL_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Khởi tạo model
    logger.info("Initializing model...")
    model = TimeAwareTransformer(
        input_dim=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Training loop
    logger.info("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/model.pth')
    logger.info("Training completed and model saved")

if __name__ == "__main__":
    main() 