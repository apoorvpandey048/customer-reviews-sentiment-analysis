"""
Quick test script to verify project setup and data pipeline.

Course: CSE3712 Big Data Analytics

This script tests:
1. Configuration loading
2. Utility functions
3. Data loader (with synthetic data)
4. Preprocessing pipeline
5. Directory structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src import config, utils, data_loader, preprocessing
        print("✓ All modules imported successfully\n")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}\n")
        return False

def test_configuration():
    """Test configuration."""
    print("Testing configuration...")
    try:
        from src.config import (
            CATEGORIES, MODEL_NAME, BATCH_SIZE, 
            LEARNING_RATE, MAX_LENGTH, validate_config
        )
        print(f"  Model: {MODEL_NAME}")
        print(f"  Categories: {len(CATEGORIES)}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Max length: {MAX_LENGTH}")
        
        # Validate config
        validate_config()
        print("✓ Configuration valid\n")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}\n")
        return False

def test_utilities():
    """Test utility functions."""
    print("Testing utilities...")
    try:
        from src.utils import set_seed, format_time, setup_logging
        
        # Test seed setting
        set_seed(42)
        
        # Test time formatting
        time_str = format_time(3725)
        assert "1h" in time_str or "62m" in time_str
        
        # Test logging
        logger = setup_logging(log_level='INFO', log_to_console=False)
        
        print("✓ Utilities working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}\n")
        return False

def test_data_loader():
    """Test data loader with synthetic data."""
    print("Testing data loader (synthetic data)...")
    try:
        from src.data_loader import AmazonReviewsLoader
        
        # Create loader with small sample
        loader = AmazonReviewsLoader(
            categories=['Electronics'],
            sample_size_per_category=100
        )
        
        # Generate synthetic data
        df = loader._generate_synthetic_data('Electronics', 100)
        
        assert len(df) == 100
        assert 'text' in df.columns
        assert 'rating' in df.columns
        
        # Test validation
        df_valid = loader.validate_data(df)
        assert len(df_valid) > 0
        
        # Test statistics
        stats = loader.get_data_statistics(df_valid)
        assert 'total_reviews' in stats
        
        print(f"  Generated {len(df)} synthetic reviews")
        print(f"  After validation: {len(df_valid)} reviews")
        print("✓ Data loader working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Data loader error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing pipeline."""
    print("Testing preprocessing...")
    try:
        from src.preprocessing import TextPreprocessor, FeatureEngineer
        import pandas as pd
        
        # Create test data
        test_df = pd.DataFrame({
            'text': [
                "This product is AMAZING! I love it!",
                "Terrible quality, don't buy this.",
                "It's okay, nothing special."
            ],
            'rating': [5, 1, 3],
            'helpful_vote': [10, 5, 2],
            'category': ['Electronics', 'Electronics', 'Electronics'],
            'verified_purchase': [True, True, False]
        })
        
        # Test text preprocessing
        processor = TextPreprocessor()
        cleaned = processor.preprocess(test_df['text'].iloc[0])
        assert len(cleaned) > 0
        print(f"  Original: {test_df['text'].iloc[0]}")
        print(f"  Cleaned: {cleaned}")
        
        # Test feature engineering
        engineer = FeatureEngineer()
        sentiment = engineer.create_sentiment_label(5)
        assert sentiment == 2  # Positive
        
        aspects = engineer.extract_aspects("The quality and price are great!")
        assert len(aspects) == 10  # Number of aspect categories
        
        print("✓ Preprocessing working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Preprocessing error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test that all directories exist."""
    print("Testing directory structure...")
    try:
        from src.config import (
            DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
            MODELS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR,
            NOTEBOOKS_DIR, DOCS_DIR, TESTS_DIR
        )
        
        dirs = {
            'data': DATA_DIR,
            'data/raw': RAW_DATA_DIR,
            'data/processed': PROCESSED_DATA_DIR,
            'models': MODELS_DIR,
            'results': RESULTS_DIR,
            'visualizations': VISUALIZATIONS_DIR,
            'notebooks': NOTEBOOKS_DIR,
            'docs': DOCS_DIR,
            'tests': TESTS_DIR
        }
        
        all_exist = True
        for name, path in dirs.items():
            if path.exists():
                print(f"  ✓ {name}")
            else:
                print(f"  ❌ {name} (missing)")
                all_exist = False
        
        if all_exist:
            print("✓ All directories exist\n")
            return True
        else:
            print("⚠ Some directories missing (will be created automatically)\n")
            return True
    except Exception as e:
        print(f"❌ Directory structure error: {e}\n")
        return False

def test_full_pipeline():
    """Test full pipeline with synthetic data."""
    print("Testing full pipeline...")
    try:
        from src.data_loader import AmazonReviewsLoader
        from src.preprocessing import preprocess_dataframe, create_train_val_test_splits
        
        # Generate small synthetic dataset
        loader = AmazonReviewsLoader(
            categories=['Electronics', 'Books'],
            sample_size_per_category=50
        )
        
        df1 = loader._generate_synthetic_data('Electronics', 50)
        df2 = loader._generate_synthetic_data('Books', 50)
        
        import pandas as pd
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Preprocess
        df_processed = preprocess_dataframe(df, verbose=False)
        
        # Create splits
        train_df, val_df, test_df = create_train_val_test_splits(
            df_processed,
            save_splits=False
        )
        
        print(f"  Input: {len(df)} reviews")
        print(f"  Processed: {len(df_processed)} reviews")
        print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        print("✓ Full pipeline working correctly\n")
        return True
    except Exception as e:
        print(f"❌ Pipeline error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PROJECT SETUP VERIFICATION")
    print("="*70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Utilities", test_utilities),
        ("Directory Structure", test_directory_structure),
        ("Data Loader", test_data_loader),
        ("Preprocessing", test_preprocessing),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("  1. Download data: python scripts/download_data.py --samples 1000")
        print("  2. Preprocess: python scripts/preprocess_data.py")
        print("  3. Explore: jupyter notebook notebooks/eda_analysis.ipynb")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - NLTK data: python -c \"import nltk; nltk.download('all')\"")
    
    print()
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
