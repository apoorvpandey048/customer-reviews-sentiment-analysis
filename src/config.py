"""
Configuration file for Amazon Review Analysis project.
Contains hyperparameters and settings for the multi-task learning pipeline.

Course: CSE3712 Big Data Analytics
Project: Multi-Task Learning for Amazon Reviews Analysis
"""

import os
import torch
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
VISUALIZATIONS_DIR = PROJECT_ROOT / 'visualizations'
EDA_DIR = VISUALIZATIONS_DIR / 'eda'
MODELING_DIR = VISUALIZATIONS_DIR / 'modeling'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
DOCS_DIR = PROJECT_ROOT / 'docs'
TESTS_DIR = PROJECT_ROOT / 'tests'

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, 
                  EDA_DIR, MODELING_DIR, NOTEBOOKS_DIR, DOCS_DIR, TESTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset categories to analyze
CATEGORIES = [
    'Electronics',
    'Books', 
    'Home_and_Kitchen',
    'Beauty_and_Personal_Care'
]

# HuggingFace dataset configuration
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
SAMPLE_SIZE_PER_CATEGORY = 250000  # 250K per category = 1M total

# Data splits
TRAIN_SPLIT = 0.70  # 70% for training
VAL_SPLIT = 0.15    # 15% for validation
TEST_SPLIT = 0.15   # 15% for testing

# Data quality filters
MIN_REVIEW_LENGTH = 20  # Minimum characters in review text
MAX_REVIEW_LENGTH = 5000  # Maximum characters (filter spam)
MIN_WORDS = 5  # Minimum words in review
REMOVE_DUPLICATES = True
FILTER_NON_ENGLISH = True

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Base model selection
MODEL_NAME = 'distilbert-base-uncased'  # Efficient transformer model
# Alternatives: 'bert-base-uncased', 'roberta-base', 'albert-base-v2'

# Model architecture
HIDDEN_DIM = 768  # DistilBERT hidden dimension
DROPOUT_RATE = 0.1
NUM_TRANSFORMER_LAYERS = 6  # DistilBERT has 6 layers

# Task-specific configurations
NUM_SENTIMENT_CLASSES = 3  # Positive (4-5 stars), Neutral (3 stars), Negative (1-2 stars)
NUM_ASPECT_LABELS = 10  # Number of product aspects to extract
HELPFULNESS_REGRESSION = True  # Predict helpfulness as regression task

# Aspect categories for multi-label classification
ASPECT_CATEGORIES = [
    'quality',
    'price',
    'shipping',
    'packaging',
    'durability',
    'appearance',
    'functionality',
    'ease_of_use',
    'customer_service',
    'value_for_money'
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Hyperparameters
LEARNING_RATE = 2e-5  # Standard for fine-tuning transformers
WEIGHT_DECAY = 0.01  # L2 regularization
BATCH_SIZE = 32  # Adjust based on GPU memory
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Learning rate scheduler
WARMUP_RATIO = 0.1  # 10% of training steps for warmup
LR_SCHEDULER = 'linear'  # Options: 'linear', 'cosine', 'constant'

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001

# Multi-task loss weights
SENTIMENT_LOSS_WEIGHT = 1.0
HELPFULNESS_LOSS_WEIGHT = 0.5
ASPECT_LOSS_WEIGHT = 0.3

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

# Tokenization
MAX_LENGTH = 256  # Maximum sequence length (tokens)
PADDING = 'max_length'
TRUNCATION = True

# Text cleaning
LOWERCASE = True
REMOVE_URLS = True
REMOVE_HTML = True
REMOVE_SPECIAL_CHARS = False  # Keep some punctuation for sentiment
EXPAND_CONTRACTIONS = True

# Stopwords
REMOVE_STOPWORDS = False  # Keep stopwords for sentiment analysis
CUSTOM_STOPWORDS = []

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Rating to sentiment mapping
SENTIMENT_MAPPING = {
    1: 'negative',
    2: 'negative',
    3: 'neutral',
    4: 'positive',
    5: 'positive'
}

# Sentiment to label ID
SENTIMENT_LABELS = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

# Helpfulness calculation
HELPFULNESS_THRESHOLD = 0.6  # Reviews with >60% helpful votes
MIN_VOTES_FOR_HELPFULNESS = 5  # Minimum votes to consider helpfulness

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to compute
SENTIMENT_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
HELPFULNESS_METRICS = ['mse', 'rmse', 'mae', 'r2']
ASPECT_METRICS = ['accuracy', 'f1_micro', 'f1_macro', 'hamming_loss']

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatic device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4  # DataLoader workers (for CPU preprocessing)
PIN_MEMORY = True if DEVICE == 'cuda' else False

# Mixed precision training (for GPU)
USE_AMP = True if DEVICE == 'cuda' else False

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================

# Logging configuration
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = RESULTS_DIR / 'training.log'
LOG_TO_CONSOLE = True
LOG_TO_FILE = True

# Checkpointing
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = MODELS_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_BEST_ONLY = True
CHECKPOINT_METRIC = 'val_loss'  # Metric to monitor for best model
CHECKPOINT_MODE = 'min'  # 'min' or 'max'

# Model saving
SAVE_FINAL_MODEL = True
MODEL_SAVE_PATH = MODELS_DIR / 'multitask_model_best.pt'
CONFIG_SAVE_PATH = MODELS_DIR / 'config.json'

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

# Random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Deterministic behavior (may reduce performance)
DETERMINISTIC = False
if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Experiment name
EXPERIMENT_NAME = 'amazon_reviews_multitask'
EXPERIMENT_VERSION = 'v1.0'

# Results storage
RESULTS_FILE = RESULTS_DIR / 'metrics.json'
TRAINING_HISTORY_FILE = RESULTS_DIR / 'training_history.csv'
PREDICTIONS_FILE = RESULTS_DIR / 'predictions.csv'

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plotting style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_FORMAT = 'png'

# Color schemes
COLOR_PALETTE = 'husl'
SENTIMENT_COLORS = {
    'negative': '#e74c3c',  # Red
    'neutral': '#95a5a6',   # Gray
    'positive': '#2ecc71'   # Green
}

# ============================================================================
# BIG DATA CONCEPTS (for academic documentation)
# ============================================================================

# MapReduce-inspired processing
CHUNK_SIZE = 10000  # Process data in chunks (simulates distributed processing)
PARALLEL_PROCESSING = True
NUM_PROCESSES = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

# Scalability considerations
MEMORY_EFFICIENT_MODE = True  # Use streaming and chunked processing
CACHE_PREPROCESSED_DATA = True  # Cache tokenized data to disk

# ============================================================================
# DEBUGGING & DEVELOPMENT
# ============================================================================

# Development mode (uses small subset of data)
DEBUG_MODE = False
DEBUG_SAMPLES = 1000

# Verbose output
VERBOSE = True

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    assert TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == 1.0, "Data splits must sum to 1.0"
    assert BATCH_SIZE > 0, "Batch size must be positive"
    assert NUM_EPOCHS > 0, "Number of epochs must be positive"
    assert LEARNING_RATE > 0, "Learning rate must be positive"
    assert MAX_LENGTH > 0, "Max length must be positive"
    assert len(CATEGORIES) > 0, "At least one category must be specified"
    assert NUM_SENTIMENT_CLASSES == 3, "Sentiment classes must be 3 (positive, neutral, negative)"
    assert SENTIMENT_LOSS_WEIGHT >= 0 and HELPFULNESS_LOSS_WEIGHT >= 0 and ASPECT_LOSS_WEIGHT >= 0, \
        "Loss weights must be non-negative"
    print("✓ Configuration validated successfully")

# Run validation
if __name__ == "__main__":
    validate_config()
    print(f"\nProject Configuration Summary:")
    print(f"  Device: {DEVICE}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Categories: {len(CATEGORIES)}")
    print(f"  Sample Size: {SAMPLE_SIZE_PER_CATEGORY:,} per category")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Max Epochs: {NUM_EPOCHS}")
    print(f"  Max Sequence Length: {MAX_LENGTH}")
    print(f"  Experiment: {EXPERIMENT_NAME} {EXPERIMENT_VERSION}")
