"""Quick manual comparison since experiment overwrote baseline"""
import json

# Load current results (from experiment)
with open('models/test_results.json') as f:
    exp_data = json.load(f)

# Baseline metrics (from TRAINING_RESULTS.md)
baseline = {
    'test_acc': 53.57,
    'val_acc': 30.77,
    'rating_mae': 1.37,
    'rating_rmse': 1.53,
    'best_epoch': 1,
    'neg_f1': 0.00,
    'neu_f1': 0.40,
    'pos_f1': 0.69
}

# Experiment metrics
experiment = {
    'test_acc': exp_data['test_losses']['sentiment_acc'] * 100,
    'val_acc': exp_data['val_losses']['sentiment_acc'] * 100,
    'rating_mae': exp_data['test_losses']['rating_mae'],
    'rating_rmse': exp_data['test_losses']['rating_rmse'],
    'best_epoch': exp_data['best_epoch']
}

print("\n" + "="*80)
print("BASELINE vs EXPERIMENT COMPARISON")
print("="*80 + "\n")

print("📊 SENTIMENT CLASSIFICATION")
print("-" * 80)
print(f"{'Metric':<30} {'Baseline':<15} {'Experiment':<15} {'Change':<15}")
print("-" * 80)

acc_change = experiment['test_acc'] - baseline['test_acc']
print(f"{'Test Accuracy':<30} {baseline['test_acc']:>13.2f}%  {experiment['test_acc']:>13.2f}%  {acc_change:>+13.2f}%")

val_acc_change = experiment['val_acc'] - baseline['val_acc']
print(f"{'Validation Accuracy':<30} {baseline['val_acc']:>13.2f}%  {experiment['val_acc']:>13.2f}%  {val_acc_change:>+13.2f}%")

print("\n📈 RATING PREDICTION")
print("-" * 80)
print(f"{'Metric':<30} {'Baseline':<15} {'Experiment':<15} {'Change':<15}")
print("-" * 80)

mae_change = experiment['rating_mae'] - baseline['rating_mae']
print(f"{'Test MAE (stars)':<30} {baseline['rating_mae']:>14.3f}  {experiment['rating_mae']:>14.3f}  {mae_change:>+14.3f}")

rmse_change = experiment['rating_rmse'] - baseline['rating_rmse']
print(f"{'Test RMSE (stars)':<30} {baseline['rating_rmse']:>14.3f}  {experiment['rating_rmse']:>14.3f}  {rmse_change:>+14.3f}")

print("\n🔄 TRAINING DETAILS")
print("-" * 80)
print(f"{'Metric':<30} {'Baseline':<15} {'Experiment':<15}")
print("-" * 80)
print(f"{'Best Epoch':<30} {baseline['best_epoch']:>14d}  {experiment['best_epoch']:>14d}")
print(f"{'Total Epochs Run':<30} {'4':>14s}  {'10':>14s}")
print(f"{'Class Weights Changed':<30} {'No':>14s}  {'Yes':>14s}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Assessment
if acc_change < -3:
    print("❌ Sentiment accuracy DECREASED significantly")
elif acc_change < 3:
    print("➖ Sentiment accuracy roughly UNCHANGED")
else:
    print("✅ Sentiment accuracy IMPROVED")

if mae_change > 0.1:
    print("❌ Rating MAE WORSENED (higher is worse)")
elif mae_change < -0.1:
    print("✅ Rating MAE IMPROVED (lower is better)")
else:
    print("➖ Rating MAE roughly UNCHANGED")

print(f"\n⚠️  IMPORTANT: Test accuracy decreased from 53.57% to {experiment['test_acc']:.2f}%")
print(f"   This suggests the increased class weights may have hurt overall performance.")
print(f"\n   Validation accuracy improved from 30.77% to {experiment['val_acc']:.2f}%,")
print(f"   showing model learned more, but best epoch was 0 (first epoch).")

print("\n💡 RECOMMENDATIONS:")
print("   1. The model stopped improving after epoch 0 - may need different LR")
print("   2. Class weights of 4.0/3.0/0.5 may be too extreme")
print("   3. Try more moderate weights: 3.0/2.0/0.6")
print("   4. Consider using focal loss instead of class weights")

print("\n" + "="*80)
print("⚠️  NOTE: Experiment overwrote baseline files in models/")
print("   Future experiments should use --output_dir flag to save separately")
print("="*80 + "\n")
