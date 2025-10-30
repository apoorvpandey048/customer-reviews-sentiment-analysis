"""
Configuration file for Amazon Review Analysis project.
Contains hyperparameters and settings for the multi-task learning pipeline.
"""

# Model hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 512
NUM_EPOCHS = 5
DROPOUT_RATE = 0.1

# Model configuration
MODEL_NAME = 'bert-base-uncased'
NUM_SENTIMENT_CLASSES = 3  # Positive, Neutral, Negative
NUM_HELPFULNESS_CLASSES = 2  # Helpful, Not Helpful

# Data configuration
CATEGORIES = [
    'Electronics',
    'Books', 
    'Home_and_Kitchen',
    'Beauty_and_Personal_Care'
]

SAMPLE_SIZE_PER_CATEGORY = 100000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Text preprocessing
MIN_WORDS = 10
MAX_TOKENS = 512
HELPFULNESS_THRESHOLD = 0.6

# File paths
DATA_DIR = 'data'
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
VISUALIZATIONS_DIR = 'visualizations'
EDA_DIR = 'visualizations/eda'

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging
LOG_LEVEL = 'INFO'
SAVE_MODELS = True
SAVE_RESULTS = True
