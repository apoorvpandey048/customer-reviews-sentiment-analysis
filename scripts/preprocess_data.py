"""
Preprocess Amazon Reviews data.

Course: CSE3712 Big Data Analytics
Script: Data preprocessing pipeline

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --input data/raw/amazon_reviews_raw.parquet
    python scripts/preprocess_data.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_dataframe, create_train_val_test_splits
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
import pandas as pd
import argparse


def main():
    """Main function for preprocessing script."""
    parser = argparse.ArgumentParser(
        description='Preprocess Amazon Reviews dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess with default settings
  python scripts/preprocess_data.py
  
  # Specify input file
  python scripts/preprocess_data.py --input data/raw/custom_data.parquet
  
  # Custom split ratios
  python scripts/preprocess_data.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
  
  # Skip saving (for testing)
  python scripts/preprocess_data.py --no-save
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input parquet file (default: data/raw/amazon_reviews_raw.parquet)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed/)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save processed data (for testing)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Use only a sample of data (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("❌ Error: Split ratios must sum to 1.0")
        sys.exit(1)
    
    # Determine input file
    if args.input is None:
        input_file = RAW_DATA_DIR / 'amazon_reviews_raw.parquet'
    else:
        input_file = Path(args.input)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("\nPlease run data download first:")
        print("  python scripts/download_data.py")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*70)
    print("AMAZON REVIEWS PREPROCESSING")
    print("="*70)
    print(f"Input file:    {input_file}")
    print(f"Output dir:    {PROCESSED_DATA_DIR}")
    print(f"Train ratio:   {args.train_ratio:.1%}")
    print(f"Val ratio:     {args.val_ratio:.1%}")
    print(f"Test ratio:    {args.test_ratio:.1%}")
    print(f"Random seed:   {RANDOM_SEED}")
    print("="*70 + "\n")
    
    try:
        # Load data
        print(f"Loading data from {input_file}...")
        df = pd.read_parquet(input_file)
        print(f"✓ Loaded {len(df):,} reviews\n")
        
        # Sample if requested
        if args.sample and args.sample < len(df):
            print(f"Sampling {args.sample:,} reviews for testing...")
            df = df.sample(n=args.sample, random_state=RANDOM_SEED)
            print(f"✓ Using {len(df):,} reviews\n")
        
        # Preprocess
        print("Starting preprocessing pipeline...\n")
        df_processed = preprocess_dataframe(
            df,
            text_column='text',
            rating_column='rating',
            helpful_votes_column='helpful_vote',
            verbose=True
        )
        
        # Display sample
        print("\nSample of preprocessed data:")
        print("-" * 70)
        sample_cols = ['category', 'rating', 'sentiment_label', 'helpfulness_score', 'word_count']
        available_sample_cols = [col for col in sample_cols if col in df_processed.columns]
        print(df_processed[available_sample_cols].head())
        print("-" * 70 + "\n")
        
        # Create splits
        print("Creating train/val/test splits...\n")
        train_df, val_df, test_df = create_train_val_test_splits(
            df_processed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            save_splits=not args.no_save
        )
        
        # Print summary
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"Train set:     {len(train_df):>10,} reviews ({args.train_ratio:.1%})")
        print(f"Val set:       {len(val_df):>10,} reviews ({args.val_ratio:.1%})")
        print(f"Test set:      {len(test_df):>10,} reviews ({args.test_ratio:.1%})")
        print(f"Total:         {len(df_processed):>10,} reviews")
        
        if not args.no_save:
            print(f"\n✓ Data saved to: {PROCESSED_DATA_DIR}")
            print("\nFiles created:")
            print("  - train.parquet")
            print("  - val.parquet")
            print("  - test.parquet")
        
        print("\nDataset features:")
        print(f"  Columns: {len(df_processed.columns)}")
        print(f"  Categories: {df_processed['category'].nunique()}")
        print(f"  Sentiment classes: {df_processed['sentiment_label'].nunique()}")
        
        print("\nNext steps:")
        print("  1. Explore data: jupyter notebook notebooks/eda_analysis.ipynb")
        print("  2. Train model: python scripts/train.py")
        print("="*70 + "\n")
        
        # Generate preprocessing report
        if not args.no_save:
            report_path = PROCESSED_DATA_DIR / 'preprocessing_report.txt'
            with open(report_path, 'w') as f:
                f.write("PREPROCESSING REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Date: {pd.Timestamp.now()}\n")
                f.write(f"Input file: {input_file}\n")
                f.write(f"Random seed: {RANDOM_SEED}\n\n")
                f.write("Data Splits:\n")
                f.write(f"  Train: {len(train_df):,} ({args.train_ratio:.1%})\n")
                f.write(f"  Val:   {len(val_df):,} ({args.val_ratio:.1%})\n")
                f.write(f"  Test:  {len(test_df):,} ({args.test_ratio:.1%})\n\n")
                f.write("Sentiment Distribution:\n")
                f.write(df_processed['sentiment_label'].value_counts().to_string())
                f.write("\n\nCategory Distribution:\n")
                f.write(df_processed['category'].value_counts().to_string())
            print(f"✓ Report saved to: {report_path}\n")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
