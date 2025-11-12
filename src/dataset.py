"""
PyTorch Dataset class for Amazon Reviews Multi-Task Learning

This module provides a custom Dataset class for loading and preprocessing
Amazon reviews data for multi-task learning (sentiment, rating, aspects).

Author: Apoorv Pandey
Course: CSE3712 - Big Data Analytics
Date: November 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import DistilBertTokenizer
from typing import Dict, Tuple, Optional
import numpy as np


class ReviewDataset(Dataset):
    """
    PyTorch Dataset for Amazon reviews with multi-task targets.
    
    Handles:
    - Text tokenization using DistilBERT tokenizer
    - Sentiment labels (classification)
    - Rating values (regression)
    - Product aspects (multi-label classification)
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        max_length: int = 128,
        text_column: str = 'cleaned_text',
        aspect_columns: Optional[list] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            dataframe: Pandas DataFrame with review data
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization (default: 128)
            text_column: Column name containing review text (default: 'cleaned_text')
            aspect_columns: List of aspect column names (auto-detected if None)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Auto-detect aspect columns if not provided
        if aspect_columns is None:
            self.aspect_columns = [col for col in dataframe.columns if col.startswith('aspect_')]
        else:
            self.aspect_columns = aspect_columns
        
        # Validate required columns
        required_cols = [text_column, 'sentiment_label', 'rating']
        for col in required_cols:
            if col not in dataframe.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - input_ids: Token IDs (torch.LongTensor)
                - attention_mask: Attention mask (torch.LongTensor)
                - sentiment_label: Sentiment class (torch.LongTensor)
                - rating: Rating value (torch.FloatTensor)
                - aspects: Binary aspect labels (torch.FloatTensor)
        """
        # Get row data
        row = self.dataframe.iloc[idx]
        text = str(row[self.text_column])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Extract aspects as binary vector
        aspects = torch.tensor(
            [row[col] for col in self.aspect_columns],
            dtype=torch.float32
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_label': torch.tensor(row['sentiment_label'], dtype=torch.long),
            'rating': torch.tensor(row['rating'], dtype=torch.float32),
            'aspects': aspects
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced sentiment data.
        
        Returns:
            Tensor of weights for each sentiment class
        """
        sentiment_counts = self.dataframe['sentiment_label'].value_counts().sort_index()
        total_samples = len(self.dataframe)
        
        # Inverse frequency weighting
        weights = total_samples / (len(sentiment_counts) * sentiment_counts.values)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_aspect_statistics(self) -> Dict[str, float]:
        """
        Get statistics about aspect label distribution.
        
        Returns:
            Dictionary with aspect frequencies and coverage
        """
        aspect_data = self.dataframe[self.aspect_columns]
        
        stats = {
            'total_aspects': len(self.aspect_columns),
            'avg_aspects_per_review': aspect_data.sum(axis=1).mean(),
            'aspect_frequencies': aspect_data.sum().to_dict(),
            'aspect_coverage': (aspect_data.sum() / len(self.dataframe)).to_dict()
        }
        
        return stats


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: DistilBertTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for DataLoader (default: 16)
        max_length: Maximum sequence length (default: 128)
        num_workers: Number of workers for DataLoader (default: 0)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Create datasets
    train_dataset = ReviewDataset(train_df, tokenizer, max_length)
    val_dataset = ReviewDataset(val_df, tokenizer, max_length)
    test_dataset = ReviewDataset(test_df, tokenizer, max_length)
    
    # Get class weights from training data
    class_weights = train_dataset.get_class_weights()
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, class_weights


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching (if needed for variable-length sequences).
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary of tensors
    """
    # Stack all tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sentiment_label = torch.stack([item['sentiment_label'] for item in batch])
    rating = torch.stack([item['rating'] for item in batch])
    aspects = torch.stack([item['aspects'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'sentiment_label': sentiment_label,
        'rating': rating,
        'aspects': aspects
    }


if __name__ == "__main__":
    """Test dataset loading and DataLoader creation."""
    
    print("Testing Review Dataset and DataLoader...")
    print("="*60)
    
    # Load sample data
    from pathlib import Path
    
    DATA_DIR = Path("../data/processed")
    train_path = DATA_DIR / "train.parquet"
    
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        print(f"✓ Loaded training data: {len(train_df)} samples")
        
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print(f"✓ Tokenizer loaded")
        
        # Create dataset
        dataset = ReviewDataset(train_df, tokenizer, max_length=128)
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"\n✓ Sample data:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Sentiment label: {sample['sentiment_label'].item()}")
        print(f"  Rating: {sample['rating'].item()}")
        print(f"  Aspects shape: {sample['aspects'].shape}")
        print(f"  Aspects sum: {sample['aspects'].sum().item()}")
        
        # Test DataLoader
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch = next(iter(train_loader))
        
        print(f"\n✓ DataLoader batch:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Sentiment labels shape: {batch['sentiment_label'].shape}")
        print(f"  Ratings shape: {batch['rating'].shape}")
        print(f"  Aspects shape: {batch['aspects'].shape}")
        
        # Get class weights
        class_weights = dataset.get_class_weights()
        print(f"\n✓ Class weights (for imbalanced data):")
        print(f"  Negative (0): {class_weights[0]:.4f}")
        print(f"  Neutral (1): {class_weights[1]:.4f}")
        print(f"  Positive (2): {class_weights[2]:.4f}")
        
        # Get aspect statistics
        aspect_stats = dataset.get_aspect_statistics()
        print(f"\n✓ Aspect statistics:")
        print(f"  Total aspects: {aspect_stats['total_aspects']}")
        print(f"  Avg aspects per review: {aspect_stats['avg_aspects_per_review']:.2f}")
        print(f"  Top 3 aspects:")
        sorted_aspects = sorted(
            aspect_stats['aspect_frequencies'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for aspect, freq in sorted_aspects[:3]:
            coverage = aspect_stats['aspect_coverage'][aspect]
            print(f"    {aspect}: {freq} ({coverage:.1%})")
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
    else:
        print(f"✗ Training data not found at {train_path}")
        print("  Run scripts/preprocess_data.py first")
