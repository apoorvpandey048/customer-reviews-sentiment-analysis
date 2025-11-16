"""
Compare baseline vs experiment results
Generates comparison report with visualizations
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_results(results_path):
    """Load test results JSON"""
    with open(results_path) as f:
        return json.load(f)

def create_comparison_table(baseline, experiment, exp_name):
    """Create formatted comparison table"""
    
    baseline_losses = baseline.get('test_losses', {})
    exp_losses = experiment.get('test_losses', {})
    
    # Calculate improvements
    sent_acc_base = baseline_losses.get('sentiment_acc', 0) * 100
    sent_acc_exp = exp_losses.get('sentiment_acc', 0) * 100
    sent_improvement = sent_acc_exp - sent_acc_base
    
    rating_mae_base = baseline_losses.get('rating_mae', 0)
    rating_mae_exp = exp_losses.get('rating_mae', 0)
    mae_improvement = rating_mae_base - rating_mae_exp
    
    rating_rmse_base = baseline_losses.get('rating_rmse', 0)
    rating_rmse_exp = exp_losses.get('rating_rmse', 0)
    rmse_improvement = rating_rmse_base - rating_rmse_exp
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80 + "\n")
    
    print("📊 SENTIMENT CLASSIFICATION")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<15} {exp_name:<15} {'Change':<15}")
    print("-" * 80)
    print(f"{'Accuracy':<30} {sent_acc_base:>13.2f}%  {sent_acc_exp:>13.2f}%  {sent_improvement:>+13.2f}%")
    
    if 'sentiment_f1' in baseline_losses:
        sent_f1_base = baseline_losses['sentiment_f1'] * 100
        sent_f1_exp = exp_losses.get('sentiment_f1', 0) * 100
        f1_improvement = sent_f1_exp - sent_f1_base
        print(f"{'Macro F1':<30} {sent_f1_base:>13.2f}%  {sent_f1_exp:>13.2f}%  {f1_improvement:>+13.2f}%")
    
    print("\n📈 RATING PREDICTION")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<15} {exp_name:<15} {'Change':<15}")
    print("-" * 80)
    print(f"{'MAE (stars)':<30} {rating_mae_base:>14.3f}  {rating_mae_exp:>14.3f}  {mae_improvement:>+14.3f}")
    print(f"{'RMSE (stars)':<30} {rating_rmse_base:>14.3f}  {rating_rmse_exp:>14.3f}  {rmse_improvement:>+14.3f}")
    
    if 'rating_r2' in baseline_losses:
        r2_base = baseline_losses['rating_r2']
        r2_exp = exp_losses.get('rating_r2', 0)
        r2_improvement = r2_exp - r2_base
        print(f"{'R² Score':<30} {r2_base:>14.3f}  {r2_exp:>14.3f}  {r2_improvement:>+14.3f}")
    
    print("\n🏷️  ASPECT EXTRACTION")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<15} {exp_name:<15} {'Change':<15}")
    print("-" * 80)
    
    if 'aspect_f1' in baseline_losses:
        aspect_f1_base = baseline_losses['aspect_f1'] * 100
        aspect_f1_exp = exp_losses.get('aspect_f1', 0) * 100
        aspect_improvement = aspect_f1_exp - aspect_f1_base
        print(f"{'Macro F1':<30} {aspect_f1_base:>13.2f}%  {aspect_f1_exp:>13.2f}%  {aspect_improvement:>+13.2f}%")
    
    if 'aspect_hamming' in baseline_losses:
        hamming_base = baseline_losses['aspect_hamming']
        hamming_exp = exp_losses.get('aspect_hamming', 0)
        hamming_improvement = hamming_base - hamming_exp
        print(f"{'Hamming Loss':<30} {hamming_base:>14.3f}  {hamming_exp:>14.3f}  {hamming_improvement:>+14.3f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Overall assessment
    improvements = 0
    if sent_improvement > 0:
        print(f"✅ Sentiment accuracy improved by {sent_improvement:.2f}%")
        improvements += 1
    elif sent_improvement < 0:
        print(f"❌ Sentiment accuracy decreased by {abs(sent_improvement):.2f}%")
    else:
        print(f"➖ Sentiment accuracy unchanged")
    
    if mae_improvement > 0:
        print(f"✅ Rating MAE improved by {mae_improvement:.3f} stars")
        improvements += 1
    elif mae_improvement < 0:
        print(f"❌ Rating MAE worsened by {abs(mae_improvement):.3f} stars")
    else:
        print(f"➖ Rating MAE unchanged")
    
    if 'aspect_f1' in baseline_losses and 'aspect_f1' in exp_losses:
        if aspect_improvement > 0:
            print(f"✅ Aspect F1 improved by {aspect_improvement:.2f}%")
            improvements += 1
        elif aspect_improvement < 0:
            print(f"❌ Aspect F1 decreased by {abs(aspect_improvement):.2f}%")
        else:
            print(f"➖ Aspect F1 unchanged")
    
    print("\n" + "="*80)
    if improvements >= 2:
        print("🎉 EXPERIMENT SUCCESSFUL - Multiple metrics improved!")
    elif improvements == 1:
        print("⚠️  MIXED RESULTS - Some improvements, some regressions")
    else:
        print("❌ EXPERIMENT UNSUCCESSFUL - No significant improvements")
    print("="*80 + "\n")

def create_comparison_plot(baseline, experiment, exp_name, output_path):
    """Create comparison bar chart"""
    
    baseline_losses = baseline.get('test_losses', {})
    exp_losses = experiment.get('test_losses', {})
    
    # Extract metrics
    metrics = {
        'Sentiment\nAccuracy': (
            baseline_losses.get('sentiment_acc', 0) * 100,
            exp_losses.get('sentiment_acc', 0) * 100
        ),
        'Rating\nMAE': (
            baseline_losses.get('rating_mae', 0),
            exp_losses.get('rating_mae', 0)
        ),
        'Rating\nRMSE': (
            baseline_losses.get('rating_rmse', 0),
            exp_losses.get('rating_rmse', 0)
        ),
    }
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric_name, (base_val, exp_val)) in enumerate(metrics.items()):
        ax = axes[idx]
        
        bars = ax.bar(['Baseline', exp_name], [base_val, exp_val], 
                     color=['#fcc419', '#51cf66'], alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Baseline vs Experiment Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved to: {output_path}")

def main():
    """Main comparison function"""
    
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <experiment_name>")
        print("Example: python compare_results.py exp1_extended_reweighted")
        return 1
    
    exp_name = sys.argv[1]
    
    # Paths
    baseline_path = Path("models/test_results.json")
    exp_path = Path(f"experiments/{exp_name}/test_results.json")
    
    # Check if files exist
    if not baseline_path.exists():
        print(f"❌ Baseline results not found: {baseline_path}")
        return 1
    
    if not exp_path.exists():
        print(f"❌ Experiment results not found: {exp_path}")
        print(f"💡 Make sure to run training with --experiment_name={exp_name} first")
        return 1
    
    # Load results
    print(f"\n📂 Loading results...")
    print(f"  Baseline: {baseline_path}")
    print(f"  Experiment: {exp_path}")
    
    baseline = load_results(baseline_path)
    experiment = load_results(exp_path)
    
    # Create comparison
    create_comparison_table(baseline, experiment, exp_name)
    
    # Create plot
    output_dir = Path("experiments/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{exp_name}_comparison.png"
    create_comparison_plot(baseline, experiment, exp_name, plot_path)
    
    print(f"\n✅ Comparison complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
