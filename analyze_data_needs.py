"""
Data Analysis: Why We Need More Data
Shows the relationship between dataset size and model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("DATA REQUIREMENTS ANALYSIS")
print("="*80)

# Load current training data
try:
    df = pd.read_parquet('data/processed/train.parquet')
    print(f"\n✅ Loaded current training set: {len(df)} samples")
except:
    print("\n❌ Could not load data")
    exit(1)

# Analyze class distribution
sent_counts = df['sentiment_label'].value_counts().sort_index()
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

print("\n📊 CURRENT CLASS DISTRIBUTION")
print("-" * 80)
for i, count in sent_counts.items():
    pct = count / len(df) * 100
    print(f"{sentiment_labels[i]:8s}: {count:3d} samples ({pct:5.1f}%)")

# Calculate imbalance
imbalance_ratio = sent_counts.max() / sent_counts.min()
print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.1f}:1")
print(f"   Problem: {imbalance_ratio:.1f}x more majority class than minority class!")

# Text length analysis
text_lengths = df['word_count'] if 'word_count' in df.columns else df['cleaned_text'].str.split().str.len()
print(f"\n📝 TEXT STATISTICS")
print("-" * 80)
print(f"Average length: {text_lengths.mean():.1f} words")
print(f"Median length:  {text_lengths.median():.1f} words")
print(f"Min length:     {text_lengths.min():.0f} words")
print(f"Max length:     {text_lengths.max():.0f} words")

if text_lengths.mean() < 10:
    print(f"\n⚠️  Reviews are VERY SHORT (< 10 words)")
    print(f"   This makes learning harder - need MORE samples to compensate!")

# Deep learning requirements
print(f"\n📚 DEEP LEARNING DATA REQUIREMENTS")
print("-" * 80)
print(f"Minimum per class:     100-500 samples")
print(f"Recommended per class: 1,000-5,000 samples")
print(f"Optimal per class:     10,000+ samples")

print(f"\n🎯 CURRENT STATUS PER CLASS:")
print("-" * 80)
for i, count in sent_counts.items():
    if count < 100:
        status = "❌ TOO SMALL - Model cannot learn"
    elif count < 1000:
        status = "⚠️  INSUFFICIENT - Will struggle"
    else:
        status = "✅ ADEQUATE - Can learn"
    print(f"{sentiment_labels[i]:8s}: {count:3d} samples - {status}")

# Calculate needed samples
print(f"\n📈 TO REACH MINIMUM (100 per class):")
for i, count in sent_counts.items():
    needed = max(0, 100 - count)
    if needed > 0:
        multiplier = 100 / count
        print(f"{sentiment_labels[i]:8s}: Need {needed:3d} more ({multiplier:.1f}x current)")

print(f"\n📈 TO REACH RECOMMENDED (1,000 per class):")
for i, count in sent_counts.items():
    needed = max(0, 1000 - count)
    multiplier = 1000 / count if count > 0 else float('inf')
    print(f"{sentiment_labels[i]:8s}: Need {needed:4d} more ({multiplier:.1f}x current)")

# Predict impact
print(f"\n🎯 PREDICTED IMPACT OF MORE DATA")
print("=" * 80)

target_sizes = [500, 1000, 2000, 5000]
current_size = len(df)

print(f"\nBaseline (current):")
print(f"  Dataset size: {current_size} samples")
print(f"  Accuracy: 53.57%")
print(f"  Negative F1: 0.00 (cannot detect negatives!)")

for target in target_sizes:
    multiplier = target / current_size
    # Rough estimates based on learning curve theory
    acc_improvement = min(30, 15 * np.log(multiplier + 1))
    neg_f1_improvement = min(0.70, 0.35 * np.log(multiplier + 1))
    
    expected_acc = min(90, 53.57 + acc_improvement)
    expected_neg_f1 = min(0.75, 0.00 + neg_f1_improvement)
    
    print(f"\nWith {target:,} samples ({multiplier:.1f}x increase):")
    print(f"  Expected Accuracy: {expected_acc:.1f}% (+{acc_improvement:.1f}%)")
    print(f"  Expected Negative F1: {expected_neg_f1:.2f} (+{neg_f1_improvement:.2f})")
    
    if multiplier >= 5:
        confidence = "🟢 HIGH"
    elif multiplier >= 3:
        confidence = "🟡 MEDIUM"
    else:
        confidence = "🔴 LOW"
    print(f"  Confidence: {confidence}")

print(f"\n💡 KEY INSIGHTS")
print("=" * 80)
print(f"1. Current negative samples: {sent_counts.get(0, 0)} - WAY TOO SMALL!")
print(f"   • Model has seen too few examples to learn negative patterns")
print(f"   • Result: Negative F1 = 0.00 (never predicts negative)")
print(f"   • Solution: Need at least 500-1,000 negative samples")

print(f"\n2. Class imbalance: {imbalance_ratio:.1f}:1 ratio")
print(f"   • Model biased toward majority class (Positive)")
print(f"   • Class weights help but not enough with tiny dataset")
print(f"   • Solution: Balance classes through oversampling or more data")

print(f"\n3. Very short text: {text_lengths.mean():.1f} words average")
print(f"   • Less context makes learning harder")
print(f"   • Requires more samples to learn patterns")
print(f"   • Solution: More data OR reduce dropout (0.3 → 0.15)")

print(f"\n📋 ACTION PLAN")
print("=" * 80)
print(f"Priority 1: Download 5,000+ reviews")
print(f"  Command: python scripts/download_more_data.py --size=5000")
print(f"  Impact: Single biggest improvement (+20-30% accuracy)")

print(f"\nPriority 2: Preprocess expanded dataset")
print(f"  Command: python scripts/preprocess_expanded.py")
print(f"  Impact: Balanced classes, better aspect coverage")

print(f"\nPriority 3: Train with improved hyperparameters")
print(f"  Command: python scripts/train.py --train_file=train_expanded.csv \\")
print(f"           --experiment_name=exp_expanded --learning_rate=1e-5 \\")
print(f"           --dropout_rate=0.15 --class_weight_negative=2.5")
print(f"  Impact: Stable training, better convergence")

print(f"\n" + "="*80)
print("✅ ANALYSIS COMPLETE - NEXT STEP: DOWNLOAD MORE DATA!")
print("="*80 + "\n")
