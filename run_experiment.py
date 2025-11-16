"""
Quick experiment runner - run improved model and compare with baseline
"""

import subprocess
import sys
from pathlib import Path

def run_experiment():
    """Run extended training with better class weights"""
    
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT: Extended Training + Class Reweighting")
    print("="*80 + "\n")
    
    print("📊 Configuration:")
    print("  • Epochs: 10 (no early stopping)")
    print("  • Class Weights: Negative=4.0, Neutral=3.0, Positive=0.5")
    print("  • Learning Rate: 2e-5")
    print("  • Sentiment Weight: 1.5 (increased importance)")
    print("\n🎯 Expected Improvements:")
    print("  • Negative F1: 0.00 → >0.20")
    print("  • Overall Accuracy: 53.57% → >60%")
    print("  • Rating MAE: 1.37 → <1.20")
    print("\n⏱️  Estimated time: ~2 minutes\n")
    
    input("Press Enter to start training...")
    
    # Run training
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--experiment_name=exp1_extended_reweighted",
        "--num_epochs=10",
        "--early_stopping_patience=10",
        "--class_weight_negative=4.0",
        "--class_weight_neutral=3.0",
        "--class_weight_positive=0.5",
        "--sentiment_weight=1.5",
        "--batch_size=16",
        "--learning_rate=2e-5",
        "--dropout_rate=0.3"
    ]
    
    print("\n🚀 Starting training...\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/evaluate.py --checkpoint_path experiments/exp1_extended_reweighted/checkpoints/best_model.pt --output_dir results/exp1")
        print("2. Compare results in TRAINING_RESULTS.md with new metrics")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_experiment())
