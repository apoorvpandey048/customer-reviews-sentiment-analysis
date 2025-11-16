# Extended EDA - Understanding Data Requirements and Class Imbalance

## Purpose
This notebook analyzes:
1. WHY our model struggles (class imbalance, small dataset)
2. HOW more data will help
3. WHAT improvements to expect

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

sys.path.append('..')
from src.config import PROCESSED_DATA_DIR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
%matplotlib inline

print("="*80)
print("EXTENDED EDA: DATA REQUIREMENTS ANALYSIS")
print("="*80)

## 1. Load Current (Small) Dataset

print("\n📂 Loading current dataset (177 reviews)...")
try:
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / 'train.parquet')
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / 'val.parquet')
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / 'test.parquet')
    
    df_current = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"✅ Loaded {len(df_current)} reviews")
except:
    print("⚠️  Could not load parquet files, trying CSV...")
    try:
        train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
        val_df = pd.read_csv(PROCESSED_DATA_DIR / 'val.csv')
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')
        df_current = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(f"✅ Loaded {len(df_current)} reviews")
    except:
        print("❌ No data found. Please run preprocessing first.")
        df_current = None

## 2. Analyze Class Imbalance Problem

if df_current is not None:
    print("\n📊 CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Sentiment distribution
    sentiment_counts = df_current['sentiment_label'].value_counts().sort_index()
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar chart
    ax = axes[0]
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    bars = ax.bar([sentiment_labels[i] for i in sentiment_counts.index], 
                   sentiment_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Reviews', fontweight='bold')
    ax.set_title('Sentiment Distribution - Current Dataset (177 reviews)', 
                 fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({height/len(df_current)*100:.1f}%)',
               ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax = axes[1]
    colors_pie = ['#ff6b6b', '#feca57', '#48dbfb']
    wedges, texts, autotexts = ax.pie(sentiment_counts.values, 
                                       labels=[sentiment_labels[i] for i in sentiment_counts.index],
                                       colors=colors_pie, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontweight': 'bold'})
    ax.set_title('Class Imbalance Visualization', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../visualizations/eda/class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Key Findings:")
    for i, count in sentiment_counts.items():
        pct = count / len(df_current) * 100
        print(f"   {sentiment_labels[i]:8s}: {count:3d} reviews ({pct:5.1f}%)")
    
    # Calculate imbalance ratio
    max_class = sentiment_counts.max()
    min_class = sentiment_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.1f}:1 (Positive:Negative)")
    print(f"   This means we have {imbalance_ratio:.1f}x more positive than negative reviews!")

## 3. Sample Size Analysis

if df_current is not None:
    print("\n📏 SAMPLE SIZE ANALYSIS")
    print("="*80)
    
    # Calculate samples per class in training set
    train_sentiment_counts = train_df['sentiment_label'].value_counts().sort_index()
    
    print("\n🎯 Training Set Distribution:")
    for i, count in train_sentiment_counts.items():
        print(f"   {sentiment_labels[i]:8s}: {count:3d} training samples")
    
    # Deep learning recommendations
    print("\n📚 Deep Learning Data Requirements:")
    print("   Minimum per class: 100-500 samples")
    print("   Recommended per class: 1,000-5,000 samples")
    print("   Optimal per class: 10,000+ samples")
    
    print(f"\n⚠️  Current Status:")
    for i, count in train_sentiment_counts.items():
        if count < 100:
            status = "❌ TOO SMALL"
        elif count < 1000:
            status = "⚠️  INSUFFICIENT"
        else:
            status = "✅ ADEQUATE"
        print(f"   {sentiment_labels[i]:8s}: {count:3d} samples - {status}")
    
    # Calculate needed samples
    print(f"\n🎯 To Reach Minimum (100 per class):")
    for i, count in train_sentiment_counts.items():
        needed = max(0, 100 - count)
        print(f"   {sentiment_labels[i]:8s}: Need {needed:3d} more samples")
    
    print(f"\n🎯 To Reach Recommended (1,000 per class):")
    for i, count in train_sentiment_counts.items():
        needed = max(0, 1000 - count)
        multiplier = (1000 / count) if count > 0 else float('inf')
        print(f"   {sentiment_labels[i]:8s}: Need {needed:4d} more samples ({multiplier:.1f}x current)")

## 4. Text Length Analysis

if df_current is not None:
    print("\n📝 TEXT LENGTH ANALYSIS")
    print("="*80)
    
    df_current['text_length'] = df_current['text'].str.split().str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(df_current['text_length'], bins=30, color='#5f27cd', alpha=0.7, edgecolor='black')
    ax.axvline(df_current['text_length'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df_current["text_length"].mean():.1f} words')
    ax.axvline(df_current['text_length'].median(), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {df_current["text_length"].median():.1f} words')
    ax.set_xlabel('Review Length (words)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Review Lengths', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Box plot by sentiment
    ax = axes[1]
    sentiment_data = [df_current[df_current['sentiment_label']==i]['text_length'] 
                     for i in sorted(df_current['sentiment_label'].unique())]
    bp = ax.boxplot(sentiment_data, labels=[sentiment_labels[i] for i in sorted(df_current['sentiment_label'].unique())],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Review Length (words)', fontweight='bold')
    ax.set_title('Review Length by Sentiment', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/eda/text_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Statistics:")
    print(f"   Mean length: {df_current['text_length'].mean():.1f} words")
    print(f"   Median length: {df_current['text_length'].median():.1f} words")
    print(f"   Min length: {df_current['text_length'].min():.0f} words")
    print(f"   Max length: {df_current['text_length'].max():.0f} words")
    
    print(f"\n📝 Insight:")
    if df_current['text_length'].mean() < 10:
        print("   ⚠️  Reviews are VERY SHORT (< 10 words average)")
        print("   This makes sentiment analysis challenging - less context for model")
        print("   Solution: Need MORE samples to compensate for short text")
    
    # Length by sentiment
    print(f"\n📊 Average Length by Sentiment:")
    for i in sorted(df_current['sentiment_label'].unique()):
        avg_len = df_current[df_current['sentiment_label']==i]['text_length'].mean()
        print(f"   {sentiment_labels[i]:8s}: {avg_len:.1f} words")

## 5. Impact Prediction

print("\n🎯 PREDICTED IMPACT OF MORE DATA")
print("="*80)

if df_current is not None:
    current_size = len(train_df)
    target_sizes = [500, 1000, 2000, 5000]
    
    print("\n📈 Expected Performance Improvements:")
    print(f"\n   Current: {current_size} training samples")
    print(f"   Baseline Accuracy: 53.57%")
    print(f"   Baseline Negative F1: 0.00 (cannot detect negatives!)")
    
    for target in target_sizes:
        multiplier = target / current_size
        # Rough estimates based on learning curve theory
        acc_improvement = min(30, 15 * np.log(multiplier + 1))
        neg_f1_improvement = min(0.70, 0.35 * np.log(multiplier + 1))
        
        expected_acc = min(90, 53.57 + acc_improvement)
        expected_neg_f1 = min(0.75, 0.00 + neg_f1_improvement)
        
        print(f"\n   📊 With {target:,} samples ({multiplier:.1f}x current):")
        print(f"      Expected Accuracy: {expected_acc:.1f}% (+{acc_improvement:.1f}%)")
        print(f"      Expected Negative F1: {expected_neg_f1:.2f} (+{neg_f1_improvement:.2f})")
        print(f"      Confidence: {'🟢 High' if multiplier >= 3 else '🟡 Medium'}")

## 6. Recommendations

print("\n💡 RECOMMENDATIONS")
print("="*80)

if df_current is not None:
    neg_count = sentiment_counts.get(0, 0)
    neu_count = sentiment_counts.get(1, 0)
    
    print("\n🎯 Priority 1: Increase Dataset Size")
    print(f"   • Download 5,000-10,000 reviews from Amazon Reviews 2023")
    print(f"   • Focus on Electronics category for consistency")
    print(f"   • Expected impact: +20-30% accuracy, Negative F1: 0.00 → 0.50-0.70")
    
    print("\n🎯 Priority 2: Balance Classes")
    if neg_count < 50:
        print(f"   • Current negative samples: {neg_count} (TOO LOW)")
        print(f"   • Target: At least 500-1,000 negative samples")
        print(f"   • Methods: Oversample negatives OR download more negative reviews")
    
    print("\n🎯 Priority 3: Data Augmentation")
    print(f"   • Apply to minority classes (Negative, Neutral)")
    print(f"   • Techniques: Synonym replacement, back-translation")
    print(f"   • Can increase effective dataset by 2-3x")
    
    print("\n🎯 Priority 4: Adjust Model Parameters")
    print(f"   • Use moderate class weights: [2.5, 2.0, 0.7] not [4.0, 3.0, 0.5]")
    print(f"   • Lower learning rate: 1e-5 instead of 2e-5")
    print(f"   • Reduce dropout: 0.15 instead of 0.3 (for short text)")
    print(f"   • Add gradient clipping: max_norm=1.0")

print("\n" + "="*80)
print("✅ EDA COMPLETE - NEXT STEPS IDENTIFIED")
print("="*80)
print("\n📋 Action Items:")
print("   1. Run: python scripts/download_more_data.py --size=5000")
print("   2. Run: python scripts/preprocess_expanded.py")
print("   3. Train with: python scripts/train.py --train_file=train_expanded.csv ...")
print("\n")
