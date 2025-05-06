# Data paths
TRAIN_PATH = "data/processed/train_processed.csv"
VAL_PATH = "data/processed/dev_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"
SEQUENCE_LENGTH = 128

# Model parameters
INPUT_DIM = 2  # Close and Volume
EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_CLASSES = 3  # HOLD, BUY, SELL
NUM_LAYERS = 6  # Number of transformer blocks
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
TIME_DECAY_REG = 0.01  # Regularization for time decay parameter

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5

# Device
DEVICE = "cuda"  # or "cpu"

class Config:
    # Data parameters
    TRAIN_PATH = TRAIN_PATH
    DEV_PATH = VAL_PATH
    TEST_PATH = TEST_PATH
    SEQUENCE_LENGTH = SEQUENCE_LENGTH
    
    # Model parameters
    INPUT_DIM = INPUT_DIM
    EMBEDDING_DIM = EMBEDDING_DIM
    NUM_HEADS = NUM_HEADS
    NUM_CLASSES = NUM_CLASSES
    DROPOUT = DROPOUT
    NUM_LAYERS = NUM_LAYERS
    
    # Training parameters
    BATCH_SIZE = BATCH_SIZE
    NUM_EPOCHS = NUM_EPOCHS
    LEARNING_RATE = LEARNING_RATE
    WEIGHT_DECAY = WEIGHT_DECAY
    
    # Time decay parameters
    TIME_DECAY_REG = TIME_DECAY_REG
    
    # Device
    DEVICE = DEVICE
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE 