"""
Data preprocessing pipeline for Amazon review analysis.
Handles text cleaning, labeling, feature engineering, and tokenization.
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import contractions
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    MIN_WORDS, MAX_TOKENS, MAX_LENGTH, HELPFULNESS_THRESHOLD,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    PROCESSED_DATA_DIR, MODEL_NAME
)

class TextPreprocessor:
    """Handles text cleaning and preprocessing operations."""
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        
    def clean_text(self, text):
        """
        Clean review text by removing unwanted elements and normalizing.
        
        Args:
            text: Raw review text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_valid_review(self, text, min_words=MIN_WORDS):
        """
        Check if review meets minimum word requirement.
        
        Args:
            text: Review text
            min_words: Minimum number of words required
            
        Returns:
            bool: True if valid, False otherwise
        """
        if pd.isna(text) or text == '':
            return False
        
        word_count = len(text.split())
        return word_count >= min_words
    
    def tokenize_text(self, text, max_length=MAX_LENGTH):
        """
        Tokenize text using BERT tokenizer.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            
        Returns:
            dict: Tokenized output with input_ids and attention_mask
        """
        if pd.isna(text):
            text = ''
        
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten()
        }

class LabelProcessor:
    """Handles label creation and processing."""
    
    def __init__(self):
        pass
    
    def calculate_helpfulness_ratio(self, helpful_vote, total_vote):
        """
        Calculate helpfulness ratio.
        
        Args:
            helpful_vote: Number of helpful votes
            total_vote: Total number of votes
            
        Returns:
            float: Helpfulness ratio
        """
        if pd.isna(helpful_vote) or pd.isna(total_vote):
            return 0.0
        
        helpful_vote = int(helpful_vote) if not pd.isna(helpful_vote) else 0
        total_vote = int(total_vote) if not pd.isna(total_vote) else 1
        
        if total_vote == 0:
            return 0.0
        
        return helpful_vote / total_vote
    
    def create_helpfulness_label(self, helpfulness_ratio, threshold=HELPFULNESS_THRESHOLD):
        """
        Create binary helpfulness label.
        
        Args:
            helpfulness_ratio: Calculated helpfulness ratio
            threshold: Threshold for helpfulness classification
            
        Returns:
            str: 'helpful' or 'not_helpful'
        """
        return 'helpful' if helpfulness_ratio > threshold else 'not_helpful'
    
    def create_sentiment_label(self, rating):
        """
        Create sentiment label from rating.
        
        Args:
            rating: Star rating (1-5)
            
        Returns:
            str: 'Positive', 'Neutral', or 'Negative'
        """
        if pd.isna(rating):
            return 'Neutral'
        
        rating = int(rating)
        
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:  # rating >= 4
            return 'Positive'

class FeatureEngineer:
    """Handles feature engineering operations."""
    
    def __init__(self):
        pass
    
    def add_metadata_features(self, df):
        """
        Add metadata features to dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        df = df.copy()
        
        # Convert verified_purchase to boolean
        df['verified_purchase'] = df['verified_purchase'].map({'Y': True, 'N': False})
        df['verified_purchase'] = df['verified_purchase'].fillna(False)
        
        # Calculate review length (word count)
        df['review_length'] = df['review_text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Calculate review age (days from review date to dataset collection date)
        dataset_collection_date = datetime(2023, 12, 31)  # Approximate dataset collection date
        
        def calculate_age(review_date):
            if pd.isna(review_date):
                return None
            try:
                # Handle different date formats
                if isinstance(review_date, str):
                    # Try different date formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                        try:
                            date_obj = datetime.strptime(review_date.split()[0], fmt)
                            break
                        except:
                            continue
                    else:
                        return None
                else:
                    date_obj = pd.to_datetime(review_date)
                
                return (dataset_collection_date - date_obj).days
            except:
                return None
        
        df['review_age'] = df['review_date'].apply(calculate_age)
        df['review_age'] = df['review_age'].fillna(df['review_age'].median())
        
        return df

def load_raw_data():
    """
    Load raw data from parquet files.
    
    Returns:
        pd.DataFrame: Raw review data
    """
    raw_filename = os.path.join('data/raw', 'amazon_reviews_raw.parquet')
    
    if not os.path.exists(raw_filename):
        raise FileNotFoundError(f"Raw data file not found: {raw_filename}")
    
    print(f"Loading raw data from {raw_filename}...")
    df = pd.read_parquet(raw_filename)
    print(f"Loaded {len(df):,} reviews")
    
    return df

def preprocess_reviews():
    """
    Main preprocessing pipeline.
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("=" * 60)
    print("AMAZON REVIEWS PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Initialize processors
    text_processor = TextPreprocessor()
    label_processor = LabelProcessor()
    feature_engineer = FeatureEngineer()
    
    # Load raw data
    df = load_raw_data()
    
    print(f"Starting with {len(df):,} reviews")
    
    # Step 1: Clean review text
    print("\n1. Cleaning review text...")
    df['review_text'] = df['review_text'].apply(text_processor.clean_text)
    
    # Step 2: Filter valid reviews
    print("2. Filtering valid reviews...")
    initial_count = len(df)
    df = df[df['review_text'].apply(text_processor.is_valid_review)]
    print(f"   Removed {initial_count - len(df):,} reviews with insufficient words")
    
    # Step 3: Remove duplicates
    print("3. Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    print(f"   Removed {initial_count - len(df):,} duplicate reviews")
    
    # Step 4: Create labels
    print("4. Creating labels...")
    
    # Helpfulness labels
    df['helpfulness_ratio'] = df.apply(
        lambda row: label_processor.calculate_helpfulness_ratio(
            row['helpful_vote'], row['total_votes']
        ), axis=1
    )
    df['helpfulness_label'] = df['helpfulness_ratio'].apply(
        label_processor.create_helpfulness_label
    )
    
    # Sentiment labels
    df['sentiment_label'] = df['rating'].apply(
        label_processor.create_sentiment_label
    )
    
    # Step 5: Add metadata features
    print("5. Adding metadata features...")
    df = feature_engineer.add_metadata_features(df)
    
    # Step 6: Tokenization
    print("6. Tokenizing reviews...")
    tokenized_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        tokenized = text_processor.tokenize_text(row['review_text'])
        tokenized_data.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        })
    
    # Add tokenized data to dataframe
    tokenized_df = pd.DataFrame(tokenized_data)
    df = pd.concat([df.reset_index(drop=True), tokenized_df], axis=1)
    
    # Step 7: Create train/val/test split
    print("7. Creating train/validation/test splits...")
    
    # Stratified split by category and sentiment
    stratify_cols = df[['category', 'sentiment_label']].apply(
        lambda x: f"{x['category']}_{x['sentiment_label']}", axis=1
    )
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=stratify_cols,
        random_state=RANDOM_SEED
    )
    
    # Second split: val vs test
    val_test_stratify = temp_df[['category', 'sentiment_label']].apply(
        lambda x: f"{x['category']}_{x['sentiment_label']}", axis=1
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT),
        stratify=val_test_stratify,
        random_state=RANDOM_SEED
    )
    
    print(f"   Train set: {len(train_df):,} reviews")
    print(f"   Validation set: {len(val_df):,} reviews")
    print(f"   Test set: {len(test_df):,} reviews")
    
    # Step 8: Save processed data
    print("8. Saving processed data...")
    
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save splits
    train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'val.parquet'), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'test.parquet'), index=False)
    
    # Save full processed dataset
    df.to_parquet(os.path.join(PROCESSED_DATA_DIR, 'processed_full.parquet'), index=False)
    
    print(f"Processed data saved to {PROCESSED_DATA_DIR}")
    
    # Display final statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final dataset size: {len(df):,} reviews")
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    print("\nSentiment distribution:")
    print(df['sentiment_label'].value_counts())
    print("\nHelpfulness distribution:")
    print(df['helpfulness_label'].value_counts())
    print(f"\nAverage review length: {df['review_length'].mean():.1f} words")
    print(f"Average helpfulness ratio: {df['helpfulness_ratio'].mean():.3f}")
    
    return train_df, val_df, test_df

def load_processed_data():
    """
    Load previously processed data.
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.parquet')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val.parquet')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.parquet')
    
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        print("Processed data not found. Running preprocessing pipeline...")
        return preprocess_reviews()
    
    print("Loading previously processed data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Loaded processed data:")
    print(f"  Train: {len(train_df):,} reviews")
    print(f"  Validation: {len(val_df):,} reviews")
    print(f"  Test: {len(test_df):,} reviews")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Run the preprocessing pipeline
    train_df, val_df, test_df = preprocess_reviews()
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(train_df[['category', 'rating', 'sentiment_label', 'helpfulness_label', 'review_length']].head())
