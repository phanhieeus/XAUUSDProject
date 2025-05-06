class Config:
    # Data parameters
    TRAIN_PATH = "data/processed/train_processed.csv"
    DEV_PATH = "data/processed/dev_processed.csv"
    TEST_PATH = "data/processed/test_processed.csv"
    SEQUENCE_LENGTH = 128
    
    # Model parameters
    INPUT_DIM = 2  # Close, Volume
    EMBEDDING_DIM = 64
    NUM_HEADS = 8
    NUM_CLASSES = 3  # buy/hold/sell
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    
    # Time decay parameters
    TIME_DECAY_REG = 0.01
    
    # Device
    DEVICE = "cuda"  # or "cpu" 