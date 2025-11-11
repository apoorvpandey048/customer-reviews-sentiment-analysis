"""
Text preprocessing and feature engineering for Amazon Reviews.

Course: CSE3712 Big Data Analytics
Author: [Your Name]

This module handles:
- Text cleaning and normalization
- Sentiment label generation
- Helpfulness score calculation
- Aspect keyword extraction
- Feature engineering for multi-task learning
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import textstat

# Transformers for tokenization
from transformers import DistilBertTokenizer

# Import configuration
try:
    from .config import (
        SENTIMENT_MAPPING, SENTIMENT_LABELS, MIN_WORDS,
        MAX_LENGTH, MODEL_NAME, LOWERCASE, REMOVE_URLS,
        REMOVE_HTML, EXPAND_CONTRACTIONS, REMOVE_STOPWORDS,
        HELPFULNESS_THRESHOLD, MIN_VOTES_FOR_HELPFULNESS,
        ASPECT_CATEGORIES, PROCESSED_DATA_DIR, RANDOM_SEED
    )
    from .utils import setup_logging, train_val_test_split
except ImportError:
    from config import (
        SENTIMENT_MAPPING, SENTIMENT_LABELS, MIN_WORDS,
        MAX_LENGTH, MODEL_NAME, LOWERCASE, REMOVE_URLS,
        REMOVE_HTML, EXPAND_CONTRACTIONS, REMOVE_STOPWORDS,
        HELPFULNESS_THRESHOLD, MIN_VOTES_FOR_HELPFULNESS,
        ASPECT_CATEGORIES, PROCESSED_DATA_DIR, RANDOM_SEED
    )
    from utils import setup_logging, train_val_test_split

# Setup logging
logger = setup_logging(log_level='INFO', log_to_console=True)

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    Handles text preprocessing for Amazon reviews.
    
    Features:
    - Text cleaning and normalization
    - Contraction expansion
    - Stopword removal (optional)
    - Lemmatization
    - Special character handling
    """
    
    def __init__(self):
        """Initialize text preprocessor with required resources."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        logger.info("TextPreprocessor initialized")
    
    def clean_text(
        self,
        text: str,
        lowercase: bool = LOWERCASE,
        remove_urls: bool = REMOVE_URLS,
        remove_html: bool = REMOVE_HTML,
        expand_contractions: bool = EXPAND_CONTRACTIONS
    ) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_html: Remove HTML tags
            expand_contractions: Expand contractions (don't -> do not)
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        if remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Expand contractions
        if expand_contractions:
            try:
                text = contractions.fix(text)
            except Exception:
                pass  # If contraction expansion fails, continue
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        return text
    
    def remove_special_characters(
        self,
        text: str,
        keep_punctuation: bool = True
    ) -> str:
        """
        Remove special characters while optionally keeping punctuation.
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep basic punctuation
        
        Returns:
            Cleaned text
        """
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:\'-]'
        else:
            # Keep only letters, numbers, and spaces
            pattern = r'[^a-zA-Z0-9\s]'
        
        text = re.sub(pattern, '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
        
        Returns:
            Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize words in text.
        
        Args:
            text: Input text
        
        Returns:
            Lemmatized text
        """
        try:
            tokens = word_tokenize(text)
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized)
        except Exception:
            return text
    
    def preprocess(
        self,
        text: str,
        remove_stopwords: bool = REMOVE_STOPWORDS,
        lemmatize: bool = False
    ) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize
        
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove special characters (keep some punctuation for sentiment)
        text = self.remove_special_characters(text, keep_punctuation=True)
        
        # Optional: Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Optional: Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text


class FeatureEngineer:
    """
    Feature engineering for multi-task learning.
    
    Generates:
    - Sentiment labels from ratings
    - Helpfulness scores
    - Aspect presence indicators
    - Text statistics
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.aspect_keywords = self._create_aspect_keywords()
        logger.info("FeatureEngineer initialized")
    
    def _create_aspect_keywords(self) -> Dict[str, List[str]]:
        """
        Create keyword lists for each product aspect.
        
        Returns:
            Dictionary mapping aspects to keywords
        """
        keywords = {
            'quality': ['quality', 'durable', 'sturdy', 'well-made', 'cheap', 'flimsy'],
            'price': ['price', 'expensive', 'affordable', 'value', 'worth', 'cost'],
            'shipping': ['shipping', 'delivery', 'arrived', 'package', 'fast', 'slow'],
            'packaging': ['packaging', 'box', 'wrapped', 'damaged', 'protected'],
            'durability': ['durable', 'lasted', 'broke', 'sturdy', 'fragile'],
            'appearance': ['looks', 'color', 'design', 'style', 'beautiful', 'ugly'],
            'functionality': ['works', 'functional', 'easy', 'difficult', 'useful'],
            'ease_of_use': ['easy', 'simple', 'complicated', 'intuitive', 'user-friendly'],
            'customer_service': ['service', 'support', 'help', 'response', 'replacement'],
            'value_for_money': ['value', 'worth', 'money', 'price', 'quality']
        }
        return keywords
    
    def create_sentiment_label(self, rating: int) -> int:
        """
        Convert rating to sentiment label.
        
        Args:
            rating: Star rating (1-5)
        
        Returns:
            Sentiment label (0=negative, 1=neutral, 2=positive)
        """
        sentiment_name = SENTIMENT_MAPPING.get(rating, 'neutral')
        return SENTIMENT_LABELS[sentiment_name]
    
    def calculate_helpfulness_score(
        self,
        helpful_votes: int,
        total_votes: int
    ) -> float:
        """
        Calculate helpfulness score as a ratio.
        
        Args:
            helpful_votes: Number of helpful votes
            total_votes: Total number of votes
        
        Returns:
            Helpfulness score [0, 1]
        """
        if total_votes < MIN_VOTES_FOR_HELPFULNESS:
            return 0.0  # Not enough votes to determine helpfulness
        
        return helpful_votes / total_votes
    
    def extract_aspects(self, text: str) -> np.ndarray:
        """
        Extract aspect presence as multi-hot encoded vector.
        
        Args:
            text: Review text
        
        Returns:
            Binary array indicating aspect presence
        """
        text_lower = text.lower()
        aspects = np.zeros(len(ASPECT_CATEGORIES), dtype=np.float32)
        
        for i, aspect in enumerate(ASPECT_CATEGORIES):
            keywords = self.aspect_keywords.get(aspect, [])
            for keyword in keywords:
                if keyword in text_lower:
                    aspects[i] = 1.0
                    break
        
        return aspects
    
    def calculate_text_statistics(self, text: str) -> Dict:
        """
        Calculate text statistics and readability metrics.
        
        Args:
            text: Review text
        
        Returns:
            Dictionary with text statistics
        """
        stats = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        }
        
        # Readability scores (requires sufficient text)
        try:
            if stats['word_count'] > 10:
                stats['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                stats['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            else:
                stats['flesch_reading_ease'] = 0
                stats['flesch_kincaid_grade'] = 0
        except Exception:
            stats['flesch_reading_ease'] = 0
            stats['flesch_kincaid_grade'] = 0
        
        return stats


class ReviewTokenizer:
    """
    Tokenizer for DistilBERT model.
    
    Handles tokenization and encoding for the model input.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Pretrained model name
            max_length: Maximum sequence length
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info(f"Tokenizer initialized: {model_name}")
        logger.info(f"Max length: {max_length}")
    
    def tokenize(
        self,
        texts: List[str],
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = None
    ) -> Dict:
        """
        Tokenize batch of texts.
        
        Args:
            texts: List of text strings
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format ('pt', 'tf', None)
        
        Returns:
            Dictionary with input_ids, attention_mask
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        return encodings


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    rating_column: str = 'rating',
    helpful_votes_column: str = 'helpful_vote',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Preprocess entire dataframe with all transformations.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        rating_column: Name of rating column
        helpful_votes_column: Name of helpful votes column
        verbose: Print progress
    
    Returns:
        Preprocessed DataFrame
    """
    if verbose:
        logger.info("="*60)
        logger.info("PREPROCESSING REVIEWS")
        logger.info("="*60)
        logger.info(f"Input shape: {df.shape}")
    
    df = df.copy()
    
    # Initialize processors
    text_processor = TextPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # 1. Clean text
    if verbose:
        logger.info("Step 1: Cleaning text...")
    df['cleaned_text'] = df[text_column].apply(text_processor.preprocess)
    
    # 2. Filter by word count
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= MIN_WORDS]
    if verbose:
        logger.info(f"  Filtered reviews with < {MIN_WORDS} words")
        logger.info(f"  Remaining: {len(df):,} reviews")
    
    # 3. Create sentiment labels
    if verbose:
        logger.info("Step 2: Creating sentiment labels...")
    df['sentiment_label'] = df[rating_column].apply(feature_engineer.create_sentiment_label)
    
    # 4. Calculate helpfulness scores
    if verbose:
        logger.info("Step 3: Calculating helpfulness scores...")
    
    # Assume total_votes = helpful_votes + 10 (synthetic)
    df['total_votes'] = df[helpful_votes_column] + np.random.randint(1, 10, size=len(df))
    df['helpfulness_score'] = df.apply(
        lambda row: feature_engineer.calculate_helpfulness_score(
            row[helpful_votes_column],
            row['total_votes']
        ),
        axis=1
    )
    
    # 5. Extract aspects
    if verbose:
        logger.info("Step 4: Extracting product aspects...")
    aspects_list = df['cleaned_text'].apply(feature_engineer.extract_aspects)
    
    # Create aspect columns
    for i, aspect in enumerate(ASPECT_CATEGORIES):
        df[f'aspect_{aspect}'] = aspects_list.apply(lambda x: x[i])
    
    # 6. Calculate text statistics
    if verbose:
        logger.info("Step 5: Calculating text statistics...")
    text_stats = df['cleaned_text'].apply(feature_engineer.calculate_text_statistics)
    text_stats_df = pd.DataFrame(text_stats.tolist())
    
    # Drop word_count from text_stats_df as we already have it
    text_stats_df = text_stats_df.drop(columns=['word_count'], errors='ignore')
    df = pd.concat([df, text_stats_df], axis=1)
    
    # 7. Select and reorder columns
    feature_cols = [
        'category', 'rating', 'sentiment_label',
        'cleaned_text', 'word_count', 'char_count',
        'helpfulness_score', 'verified_purchase'
    ]
    
    # Add aspect columns
    aspect_cols = [f'aspect_{aspect}' for aspect in ASPECT_CATEGORIES]
    feature_cols.extend(aspect_cols)
    
    # Add text statistics
    stat_cols = ['flesch_reading_ease', 'flesch_kincaid_grade']
    feature_cols.extend(stat_cols)
    
    # Keep only relevant columns
    available_cols = [col for col in feature_cols if col in df.columns]
    df = df[available_cols]
    
    if verbose:
        logger.info(f"\n✓ Preprocessing complete!")
        logger.info(f"Output shape: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        logger.info("="*60 + "\n")
    
    return df


def create_train_val_test_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train, validation, and test splits.
    
    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        save_splits: Whether to save splits to disk
    
    Returns:
        train_df, val_df, test_df
    """
    logger.info("Creating data splits...")
    
    train_df, val_df, test_df = train_val_test_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_column='sentiment_label',
        random_state=RANDOM_SEED
    )
    
    if save_splits:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        train_path = PROCESSED_DATA_DIR / 'train.parquet'
        val_path = PROCESSED_DATA_DIR / 'val.parquet'
        test_path = PROCESSED_DATA_DIR / 'test.parquet'
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        logger.info(f"✓ Saved train set to {train_path}")
        logger.info(f"✓ Saved val set to {val_path}")
        logger.info(f"✓ Saved test set to {test_path}")
    
    return train_df, val_df, test_df


# Command-line interface
if __name__ == "__main__":
    import argparse
    from data_loader import download_and_prepare_data
    
    parser = argparse.ArgumentParser(description='Preprocess Amazon Reviews data')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input parquet file (if None, will download)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AMAZON REVIEWS PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Load or download data
    if args.input:
        logger.info(f"Loading data from {args.input}...")
        df = pd.read_parquet(args.input)
    else:
        logger.info("Downloading data...")
        df = download_and_prepare_data()
    
    # Preprocess
    df_processed = preprocess_dataframe(df)
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_splits(
        df_processed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        save_splits=True
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Train set: {len(train_df):,} reviews")
    print(f"Val set:   {len(val_df):,} reviews")
    print(f"Test set:  {len(test_df):,} reviews")
    print(f"\nData saved to: {PROCESSED_DATA_DIR}")
    print("="*60 + "\n")
