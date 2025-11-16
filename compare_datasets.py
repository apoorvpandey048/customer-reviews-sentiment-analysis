"""
Quick comparison of original vs expanded datasets to determine if new EDA is needed
"""
import pandas as pd

print("\n" + "="*80)
print("DATASET COMPARISON: ORIGINAL VS EXPANDED")
print("="*80)

# Load both datasets
df_old = pd.read_parquet('data/processed/train.parquet')
df_new = pd.read_csv('data/processed/train_expanded.csv')

print("\n📊 SIZE COMPARISON:")
print(f"  Original dataset: {len(df_old):,} samples")
print(f"  Expanded dataset: {len(df_new):,} samples")
print(f"  Increase: {len(df_new)/len(df_old):.1f}x")

print("\n📝 TEXT LENGTH:")
old_avg = df_old['cleaned_text'].str.split().str.len().mean()
new_avg = df_new['cleaned_text'].str.split().str.len().mean()
print(f"  Original: {old_avg:.1f} words average")
print(f"  Expanded: {new_avg:.1f} words average")
print(f"  Increase: {new_avg/old_avg:.1f}x")

print("\n🏷️  SENTIMENT DISTRIBUTION:")
print("\nOriginal:")
old_dist = df_old['sentiment_label'].value_counts().sort_index()
for label, count in old_dist.items():
    pct = count / len(df_old) * 100
    print(f"  Class {label}: {count:>4} ({pct:>5.1f}%)")

print("\nExpanded:")
new_dist = df_new['sentiment_label'].value_counts().sort_index()
for label, count in new_dist.items():
    pct = count / len(df_new) * 100
    print(f"  Class {label}: {count:>4} ({pct:>5.1f}%)")

print("\n⚖️  CLASS BALANCE:")
old_imbalance = old_dist.max() / old_dist.min()
new_imbalance = new_dist.max() / new_dist.min() if len(new_dist) > 1 else 1.0
print(f"  Original imbalance ratio: {old_imbalance:.1f}:1")
print(f"  Expanded imbalance ratio: {new_imbalance:.1f}:1")

print("\n📋 COLUMNS:")
print(f"  Original: {len(df_old.columns)} columns")
print(f"  Expanded: {len(df_new.columns)} columns")

# Check for new columns
old_cols = set(df_old.columns)
new_cols = set(df_new.columns)
added_cols = new_cols - old_cols
removed_cols = old_cols - new_cols

if added_cols:
    print(f"\n  ✅ Added columns: {', '.join(added_cols)}")
if removed_cols:
    print(f"  ❌ Removed columns: {', '.join(removed_cols)}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Determine if EDA is needed
major_changes = []
if len(df_new) / len(df_old) > 5:
    major_changes.append(f"Dataset size increased {len(df_new)/len(df_old):.1f}x")
if new_avg / old_avg > 2:
    major_changes.append(f"Average text length increased {new_avg/old_avg:.1f}x")
if abs(new_imbalance - old_imbalance) > 2:
    major_changes.append(f"Class balance changed significantly")
if added_cols or removed_cols:
    major_changes.append("Column structure changed")

if major_changes:
    print("\n⚠️  YES, NEW EDA IS RECOMMENDED because:")
    for i, change in enumerate(major_changes, 1):
        print(f"  {i}. {change}")
    
    print("\n📊 Suggested EDA analyses:")
    print("  • Distribution plots (ratings, sentiment, text length)")
    print("  • Word clouds for positive/negative reviews")
    print("  • Correlation analysis (if rating available)")
    print("  • Aspect coverage analysis")
    print("  • Comparison with original dataset")
else:
    print("\n✅ No major changes detected. Existing EDA is sufficient.")

print("\n" + "="*80 + "\n")
