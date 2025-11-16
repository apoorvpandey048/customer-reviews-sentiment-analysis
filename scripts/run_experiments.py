"""
Experiment Runner for Model Improvements
Runs multiple experiments and compares results automatically
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExperimentRunner:
    """Run and compare multiple experiments"""
    
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_experiment(self, name, config):
        """Run a single experiment"""
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {name}")
        print(f"{'='*80}\n")
        
        # Create experiment directory
        exp_dir = self.experiments_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            "python", "scripts/train.py",
            f"--experiment_name={name}",
            f"--output_dir={exp_dir}",
        ]
        
        # Add all config parameters
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}={value}")
        
        # Run training
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        if result.returncode != 0:
            print(f"❌ Experiment failed: {name}")
            print(result.stderr)
            return None
        
        print(f"✅ Experiment completed in {training_time:.1f} seconds")
        
        # Load results
        results_file = exp_dir / "test_results.json"
        if results_file.exists():
            with open(results_file) as f:
                test_results = json.load(f)
            
            # Store experiment info
            experiment_info = {
                'name': name,
                'config': config,
                'training_time': training_time,
                'results': test_results,
                'timestamp': start_time.isoformat()
            }
            
            self.results.append(experiment_info)
            
            # Save experiment log
            with open(self.experiments_dir / f"{name}_log.json", 'w') as f:
                json.dump(experiment_info, f, indent=2)
            
            return experiment_info
        else:
            print(f"⚠️  No results file found for {name}")
            return None
    
    def compare_results(self):
        """Generate comparison visualizations"""
        if len(self.results) < 2:
            print("Need at least 2 experiments to compare")
            return
        
        print(f"\n{'='*80}")
        print(f"COMPARING {len(self.results)} EXPERIMENTS")
        print(f"{'='*80}\n")
        
        # Create comparison directory
        comparison_dir = self.experiments_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        # Extract metrics
        names = [r['name'] for r in self.results]
        
        # Sentiment metrics
        sent_acc = [r['results']['test_losses'].get('sentiment_acc', 0) for r in self.results]
        
        # Rating metrics
        rating_mae = [r['results']['test_losses'].get('rating_mae', 0) for r in self.results]
        rating_rmse = [r['results']['test_losses'].get('rating_rmse', 0) for r in self.results]
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Experiment': names,
            'Sentiment Acc': [f"{acc*100:.2f}%" for acc in sent_acc],
            'Rating MAE': [f"{mae:.3f}" for mae in rating_mae],
            'Rating RMSE': [f"{rmse:.3f}" for rmse in rating_rmse],
            'Training Time': [f"{r['training_time']:.1f}s" for r in self.results]
        })
        
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(comparison_dir / "comparison_table.csv", index=False)
        
        # Visualizations
        self._plot_metric_comparison(names, sent_acc, "Sentiment Accuracy", 
                                     comparison_dir / "sentiment_acc_comparison.png")
        self._plot_metric_comparison(names, rating_mae, "Rating MAE (Lower is Better)", 
                                     comparison_dir / "rating_mae_comparison.png", invert=True)
        self._plot_metric_comparison(names, rating_rmse, "Rating RMSE (Lower is Better)", 
                                     comparison_dir / "rating_rmse_comparison.png", invert=True)
        
        # Calculate improvements over baseline
        if self.results[0]['name'] == 'baseline':
            self._calculate_improvements(comparison_dir)
        
        print(f"\n✅ Comparison plots saved to {comparison_dir}")
    
    def _plot_metric_comparison(self, names, values, title, save_path, invert=False):
        """Plot bar chart comparing a metric across experiments"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#ff6b6b' if invert else '#51cf66'] * len(names)
        colors[0] = '#fcc419'  # Highlight baseline
        
        bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} - Experiment Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_improvements(self, output_dir):
        """Calculate improvement percentages over baseline"""
        baseline = self.results[0]
        baseline_metrics = baseline['results']['test_losses']
        
        improvements = []
        for exp in self.results[1:]:
            exp_metrics = exp['results']['test_losses']
            
            sent_imp = ((exp_metrics.get('sentiment_acc', 0) - 
                        baseline_metrics.get('sentiment_acc', 0)) / 
                       baseline_metrics.get('sentiment_acc', 1)) * 100
            
            mae_imp = ((baseline_metrics.get('rating_mae', 1) - 
                       exp_metrics.get('rating_mae', 1)) / 
                      baseline_metrics.get('rating_mae', 1)) * 100
            
            improvements.append({
                'Experiment': exp['name'],
                'Sentiment Acc Improvement': f"{sent_imp:+.2f}%",
                'Rating MAE Improvement': f"{mae_imp:+.2f}%"
            })
        
        imp_df = pd.DataFrame(improvements)
        print(f"\n{'='*80}")
        print("IMPROVEMENTS OVER BASELINE")
        print(f"{'='*80}")
        print(imp_df.to_string(index=False))
        imp_df.to_csv(output_dir / "improvements_over_baseline.csv", index=False)


def run_all_experiments():
    """Run all planned experiments"""
    runner = ExperimentRunner()
    
    # Baseline (already run)
    print("Using existing baseline results...")
    baseline_results = Path("models/test_results.json")
    if baseline_results.exists():
        with open(baseline_results) as f:
            results = json.load(f)
        runner.results.append({
            'name': 'baseline',
            'config': {
                'batch_size': 16,
                'num_epochs': 4,
                'learning_rate': 2e-5,
                'dropout_rate': 0.3
            },
            'training_time': 60,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    # Experiment 1: Extended Training + Increased Class Weights
    runner.run_experiment('exp1_extended_reweighted', {
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 2e-5,
        'dropout_rate': 0.3,
        'early_stopping_patience': 10,  # Effectively disable early stopping
        'class_weight_negative': 4.0,
        'class_weight_neutral': 3.0,
        'class_weight_positive': 0.5,
        'sentiment_weight': 1.5,  # Increase sentiment importance
        'data_dir': 'data/processed'
    })
    
    # Experiment 2: Lower Learning Rate
    runner.run_experiment('exp2_lower_lr', {
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 1e-5,
        'dropout_rate': 0.3,
        'early_stopping_patience': 10,
        'class_weight_negative': 4.0,
        'class_weight_neutral': 3.0,
        'class_weight_positive': 0.5,
        'data_dir': 'data/processed'
    })
    
    # Experiment 3: Higher Dropout
    runner.run_experiment('exp3_higher_dropout', {
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 2e-5,
        'dropout_rate': 0.5,
        'early_stopping_patience': 10,
        'class_weight_negative': 4.0,
        'class_weight_neutral': 3.0,
        'class_weight_positive': 0.5,
        'data_dir': 'data/processed'
    })
    
    # Compare all results
    runner.compare_results()
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}\n")
    print(f"Results saved to: {runner.experiments_dir}")


if __name__ == "__main__":
    run_all_experiments()
