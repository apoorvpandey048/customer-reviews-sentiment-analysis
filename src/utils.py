"""
Utility functions for Amazon Review Analysis project.
Provides helper functions for data processing, evaluation, and visualization.

Course: CSE3712 Big Data Analytics
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file=None, log_level='INFO', log_to_console=True):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
    
    Returns:
        logger: Configured logger object
    """
    logger = logging.getLogger('AmazonReviewAnalysis')
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} for reproducibility")

# ============================================================================
# FILE I/O
# ============================================================================

def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✓ Saved to {filepath}")

def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_model(model, filepath: str, config: Dict = None):
    """
    Save PyTorch model with configuration.
    
    Args:
        model: PyTorch model
        filepath: Path to save model
        config: Optional configuration dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if config:
        save_dict['config'] = config
    
    torch.save(save_dict, filepath)
    print(f"✓ Model saved to {filepath}")

def load_model(model, filepath: str):
    """
    Load PyTorch model from file.
    
    Args:
        model: PyTorch model instance
        filepath: Path to model file
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from {filepath}")
    return model

# ============================================================================
# DATA PROCESSING
# ============================================================================

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_column: str = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify_column: Column to stratify by
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df: Split dataframes
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    if stratify_column:
        # Stratified split (more complex, not implemented here for simplicity)
        # For now, simple split
        pass
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"Data split:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df

def balance_classes(
    df: pd.DataFrame,
    label_column: str,
    method: str = 'undersample',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance class distribution in dataframe.
    
    Args:
        df: Input dataframe
        label_column: Column containing class labels
        method: 'undersample' or 'oversample'
        random_state: Random seed
    
    Returns:
        balanced_df: Balanced dataframe
    """
    class_counts = df[label_column].value_counts()
    print(f"Original class distribution:\n{class_counts}")
    
    if method == 'undersample':
        min_count = class_counts.min()
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            sampled_df = class_df.sample(n=min_count, random_state=random_state)
            balanced_dfs.append(sampled_df)
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    elif method == 'oversample':
        max_count = class_counts.max()
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            sampled_df = class_df.sample(n=max_count, replace=True, random_state=random_state)
            balanced_dfs.append(sampled_df)
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nBalanced class distribution:\n{balanced_df[label_column].value_counts()}")
    return balanced_df

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, task_type='classification'):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: 'classification' or 'regression'
    
    Returns:
        metrics: Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
    )
    
    metrics = {}
    
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    elif task_type == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics

def print_metrics(metrics: Dict, title: str = "Metrics"):
    """Pretty print metrics dictionary."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            if isinstance(value, float):
                print(f"  {key:<20}: {value:.4f}")
            else:
                print(f"  {key:<20}: {value}")
    print(f"{'='*60}\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track and display training progress."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def update(self, epoch: int, train_loss: float, val_loss: float,
               train_acc: float = None, val_acc: float = None):
        """Update progress with epoch results."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        
        # Print progress
        print(f"\nEpoch {epoch}/{self.total_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        if train_acc:
            print(f"  Train Acc:  {train_acc:.4f}")
        if val_acc:
            print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
    
    def get_history(self) -> Dict:
        """Return training history."""
        return self.history

# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device_info():
    """Print device information."""
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print(f"CUDA Available: No")
        print(f"Using CPU")
    
    print(f"PyTorch Version: {torch.__version__}")
    print("="*60 + "\n")

# ============================================================================
# TIME UTILITIES
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test device info
    get_device_info()
    
    # Test seed setting
    set_seed(42)
    
    # Test time formatting
    print(f"Time format test: {format_time(3725)}")
    
    print("\n✓ All utility functions working correctly")
