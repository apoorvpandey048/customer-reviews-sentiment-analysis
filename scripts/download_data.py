"""
Download Amazon Reviews 2023 dataset.

Course: CSE3712 Big Data Analytics
Script: Automated data acquisition

Usage:
    python scripts/download_data.py --categories all --samples 250000
    python scripts/download_data.py --categories Electronics Books --samples 100000
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import download_and_prepare_data
from src.config import CATEGORIES, SAMPLE_SIZE_PER_CATEGORY
import argparse


def main():
    """Main function for data download script."""
    parser = argparse.ArgumentParser(
        description='Download Amazon Reviews 2023 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all categories with default sample size
  python scripts/download_data.py
  
  # Download specific categories
  python scripts/download_data.py --categories Electronics Books --samples 100000
  
  # Force re-download
  python scripts/download_data.py --force
  
  # Download small sample for testing
  python scripts/download_data.py --samples 1000
        """
    )
    
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        choices=CATEGORIES + ['all'],
        help=f'Product categories to download. Options: {", ".join(CATEGORIES)} or "all"'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLE_SIZE_PER_CATEGORY,
        help=f'Number of samples per category (default: {SAMPLE_SIZE_PER_CATEGORY:,})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if data exists'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/raw/)'
    )
    
    args = parser.parse_args()
    
    # Handle 'all' categories option
    if args.categories and 'all' in args.categories:
        args.categories = CATEGORIES
    elif args.categories is None:
        args.categories = CATEGORIES
    
    # Print configuration
    print("\n" + "="*70)
    print("AMAZON REVIEWS DATA DOWNLOAD")
    print("="*70)
    print(f"Categories:     {', '.join(args.categories)}")
    print(f"Samples/category: {args.samples:,}")
    print(f"Force download:  {args.force}")
    print(f"Total reviews:   ~{args.samples * len(args.categories):,}")
    print("="*70 + "\n")
    
    # Confirm for large downloads
    if args.samples * len(args.categories) > 500000 and not args.force:
        response = input("This will download a large dataset. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    # Download data
    try:
        df = download_and_prepare_data(
            categories=args.categories,
            sample_size=args.samples,
            force_download=args.force
        )
        
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"✓ Total reviews: {len(df):,}")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Categories: {df['category'].nunique()}")
        print(f"✓ Data location: data/raw/")
        print("\nNext steps:")
        print("  1. Run: python scripts/preprocess_data.py")
        print("  2. Or open: notebooks/eda_analysis.ipynb")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during download: {str(e)}")
        print("Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
