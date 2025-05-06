import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report
import os

from models.time_aware_transformer import TimeAwareTransformer
from utils.data_utils import get_data_loaders
from config import Config

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    def __init__(self, patience=5, min_delta=0):
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: TimeAwareTransformer model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for features, timestamps, labels in pbar:
        # Move data to device
        features = features.to(device)  # [batch_size, seq_len, input_dim]
        timestamps = timestamps.to(device)  # [batch_size, seq_len, 1]
        labels = labels.to(device)  # [batch_size]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, timestamps)  # [batch_size, num_classes]
        loss = criterion(outputs, labels)
        
        # Add time decay regularization
        time_decay_reg = Config.TIME_DECAY_REG * torch.abs(model.time_aware_attention.time_decay)
        loss = loss + time_decay_reg
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)  # [batch_size]
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss/len(train_loader), correct/total

def evaluate(model, data_loader, criterion, device, desc="Evaluating"):
    """
    Evaluate model on a dataset
    
    Args:
        model: TimeAwareTransformer model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        desc: Description for progress bar
    
    Returns:
        tuple: (average loss, accuracy, classification report)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for features, timestamps, labels in pbar:
            # Move data to device
            features = features.to(device)  # [batch_size, seq_len, input_dim]
            timestamps = timestamps.to(device)  # [batch_size, seq_len, 1]
            labels = labels.to(device)  # [batch_size]
            
            # Forward pass
            outputs = model(features, timestamps)  # [batch_size, num_classes]
            loss = criterion(outputs, labels)
            
            # Update statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)  # [batch_size]
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss/len(data_loader):.4f}'})
    
    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=['HOLD', 'BUY', 'SELL'],
        digits=4
    )
    
    return total_loss/len(data_loader), accuracy, report

def main():
    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if processed data exists
    if not all(os.path.exists(path) for path in [Config.TRAIN_PATH, Config.DEV_PATH, Config.TEST_PATH]):
        print("Processed data files not found. Please run preprocessing first:")
        print("python utils/preprocess_data.py")
        return
    
    # Get data loaders
    print("Loading data...")
    train_loader, dev_loader, test_loader = get_data_loaders(
        Config.TRAIN_PATH,
        Config.DEV_PATH,
        Config.TEST_PATH,
        Config.SEQUENCE_LENGTH,
        Config.BATCH_SIZE
    )
    
    # Initialize model
    print("Initializing model...")
    model = TimeAwareTransformer(
        input_dim=Config.INPUT_DIM,  # 2 for Close and Volume
        embedding_dim=Config.EMBEDDING_DIM,
        num_heads=Config.NUM_HEADS,
        num_classes=Config.NUM_CLASSES,  # 3 for HOLD, BUY, SELL
        dropout=Config.DROPOUT
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Training loop
    print("\nStarting training...")
    best_dev_accuracy = 0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on dev
        dev_loss, dev_accuracy, dev_report = evaluate(
            model, dev_loader, criterion, device, desc="Validating"
        )
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")
        print("\nDev Classification Report:")
        print(dev_report)
        
        # Save best model
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"Saved best model with dev accuracy: {best_dev_accuracy:.4f}")
        
        # Check early stopping
        early_stopping(dev_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_loss, test_accuracy, test_report = evaluate(
        model, test_loader, criterion, device, desc="Testing"
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:")
    print(test_report)

if __name__ == "__main__":
    main() 