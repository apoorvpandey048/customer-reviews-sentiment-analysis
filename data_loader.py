"""
Data loader for Amazon Reviews 2023 dataset.
Downloads and processes the dataset from HuggingFace with streaming support.
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from config import (
    CATEGORIES, 
    SAMPLE_SIZE_PER_CATEGORY, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR
)

def download_amazon_reviews():
    """
    Download Amazon Reviews 2023 dataset from HuggingFace.
    Uses streaming to handle large files efficiently.
    """
    print("Downloading Amazon Reviews 2023 dataset...")
    
    # Load dataset with streaming for memory efficiency
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", 
        streaming=True,
        split="train"
    )
    
    print("Dataset loaded successfully with streaming enabled.")
    return dataset

def filter_categories(dataset, categories=None):
    """
    Filter dataset to include only specified categories.
    
    Args:
        dataset: HuggingFace dataset object
        categories: List of category names to include
        
    Returns:
        Filtered dataset iterator
    """
    if categories is None:
        categories = CATEGORIES
    
    print(f"Filtering dataset for categories: {categories}")
    
    # Filter for specified categories
    filtered_data = []
    category_counts = {cat: 0 for cat in categories}
    
    for item in tqdm(dataset, desc="Filtering categories"):
        if item.get('category') in categories:
            filtered_data.append(item)
            category_counts[item['category']] += 1
            
            # Stop when we have enough samples for each category
            if all(count >= SAMPLE_SIZE_PER_CATEGORY for count in category_counts.values()):
                break
    
    print(f"Category distribution after filtering:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count:,} reviews")
    
    return filtered_data

def sample_reviews(data, sample_size_per_category=None):
    """
    Sample reviews from each category to create balanced dataset.
    
    Args:
        data: List of review dictionaries
        sample_size_per_category: Number of reviews per category
        
    Returns:
        Sampled data as list of dictionaries
    """
    if sample_size_per_category is None:
        sample_size_per_category = SAMPLE_SIZE_PER_CATEGORY
    
    print(f"Sampling {sample_size_per_category:,} reviews per category...")
    
    # Group by category
    category_data = {}
    for item in data:
        category = item.get('category')
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(item)
    
    # Sample from each category
    sampled_data = []
    for category, items in category_data.items():
        if len(items) > sample_size_per_category:
            # Random sampling
            import random
            random.seed(42)  # For reproducibility
            sampled_items = random.sample(items, sample_size_per_category)
        else:
            sampled_items = items
        
        sampled_data.extend(sampled_items)
        print(f"  {category}: {len(sampled_items):,} reviews sampled")
    
    return sampled_data

def save_as_parquet(data, filename):
    """
    Save data as parquet file for efficient storage and loading.
    
    Args:
        data: List of dictionaries to save
        filename: Output filename
    """
    print(f"Saving {len(data):,} reviews to {filename}...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save as parquet
    df.to_parquet(filename, index=False)
    print(f"Data saved successfully to {filename}")
    
    return df

def load_amazon_reviews():
    """
    Main function to download, filter, and save Amazon Reviews dataset.
    
    Returns:
        pandas.DataFrame: Processed dataset
    """
    print("=" * 60)
    print("AMAZON REVIEWS 2023 DATA LOADER")
    print("=" * 60)
    
    # Step 1: Download dataset
    dataset = download_amazon_reviews()
    
    # Step 2: Filter for specified categories
    filtered_data = filter_categories(dataset)
    
    # Step 3: Sample reviews
    sampled_data = sample_reviews(filtered_data)
    
    # Step 4: Save raw data
    raw_filename = os.path.join(RAW_DATA_DIR, "amazon_reviews_raw.parquet")
    raw_df = save_as_parquet(sampled_data, raw_filename)
    
    # Step 5: Save metadata
    metadata_filename = os.path.join(RAW_DATA_DIR, "amazon_reviews_metadata.parquet")
    
    # Extract metadata columns
    metadata_columns = [
        'review_id', 'product_id', 'reviewer_id', 'category', 
        'product_title', 'review_title', 'verified_purchase',
        'review_date', 'rating', 'helpful_vote', 'total_votes'
    ]
    
    metadata_data = []
    for item in sampled_data:
        metadata_item = {col: item.get(col) for col in metadata_columns}
        metadata_data.append(metadata_item)
    
    metadata_df = save_as_parquet(metadata_data, metadata_filename)
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)
    print(f"Total reviews loaded: {len(sampled_data):,}")
    print(f"Raw data saved to: {raw_filename}")
    print(f"Metadata saved to: {metadata_filename}")
    print(f"Categories included: {CATEGORIES}")
    
    return raw_df, metadata_df

def load_processed_data():
    """
    Load previously processed data from parquet files.
    
    Returns:
        tuple: (raw_df, metadata_df)
    """
    raw_filename = os.path.join(RAW_DATA_DIR, "amazon_reviews_raw.parquet")
    metadata_filename = os.path.join(RAW_DATA_DIR, "amazon_reviews_metadata.parquet")
    
    if not os.path.exists(raw_filename) or not os.path.exists(metadata_filename):
        print("Processed data not found. Running data loading pipeline...")
        return load_amazon_reviews()
    
    print("Loading previously processed data...")
    raw_df = pd.read_parquet(raw_filename)
    metadata_df = pd.read_parquet(metadata_filename)
    
    print(f"Loaded {len(raw_df):,} reviews from processed files.")
    return raw_df, metadata_df

if __name__ == "__main__":
    # Run the data loading pipeline
    raw_df, metadata_df = load_amazon_reviews()
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Shape: {raw_df.shape}")
    print(f"Columns: {list(raw_df.columns)}")
    print("\nCategory distribution:")
    print(raw_df['category'].value_counts())
    print("\nRating distribution:")
    print(raw_df['rating'].value_counts().sort_index())
