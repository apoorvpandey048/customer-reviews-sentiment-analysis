"""
Evaluation Script for Multi-Task Review Analysis Model

This script evaluates the trained model on the test set and generates:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrices for sentiment classification
- Rating prediction metrics (MAE, RMSE, R²)
- Aspect extraction metrics (per-aspect F1-scores)
- Visualization plots
- Detailed evaluation report

Author: Apoorv Pandey
Course: CSE3712 - Big Data Analytics
Date: November 2025
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score,
    multilabel_confusion_matrix, hamming_loss
)
from transformers import DistilBertTokenizer
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model import MultiTaskReviewModel, create_model
from src.dataset import ReviewDataset, create_dataloaders


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model_config = {
        'num_sentiments': 3,
        'num_aspects': 10,
        'dropout_rate': config['dropout_rate'],
        'freeze_bert': config['freeze_bert'],
        'pretrained_model': config['model_name']
    }
    
    model, _ = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    return model, checkpoint


def evaluate_sentiment(y_true, y_pred, output_dir: Path):
    """Evaluate sentiment classification task."""
    print("\n" + "="*80)
    print("SENTIMENT CLASSIFICATION EVALUATION")
    print("="*80)
    
    # Class names
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(output_dir / 'sentiment_classification_report.json', 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Sentiment Classification Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision.mean(),
        'macro_recall': recall.mean(),
        'macro_f1': f1.mean(),
        'per_class': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            } for i in range(len(class_names))
        }
    }
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-averaged Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-averaged Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-averaged F1-Score: {metrics['macro_f1']:.4f}")
    
    return metrics


def evaluate_rating(y_true, y_pred, output_dir: Path):
    """Evaluate rating prediction task."""
    print("\n" + "="*80)
    print("RATING PREDICTION EVALUATION")
    print("="*80)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Round predictions to nearest 0.5 for star rating
    y_pred_rounded = np.round(y_pred * 2) / 2
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    
    mae_rounded = mean_absolute_error(y_true, y_pred_rounded)
    
    print(f"\nMean Absolute Error (MAE): {mae:.4f} stars")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} stars")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE (Rounded to 0.5): {mae_rounded:.4f} stars")
    
    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot: Predicted vs True
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
    axes[0].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Rating', fontsize=12)
    axes[0].set_ylabel('Predicted Rating', fontsize=12)
    axes[0].set_title('Rating Prediction: Predicted vs True', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)
    
    # Error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                    label=f'Mean Error: {errors.mean():.3f}')
    axes[1].set_xlabel('Prediction Error (Predicted - True)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Rating Prediction Error Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rating_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Rating-wise MAE
    rating_mae = {}
    for rating in range(1, 6):
        mask = y_true == rating
        if mask.sum() > 0:
            rating_mae[rating] = mean_absolute_error(y_true[mask], y_pred[mask])
    
    print("\nPer-Rating MAE:")
    for rating, mae_val in rating_mae.items():
        print(f"  {rating} stars: {mae_val:.4f}")
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mae_rounded': float(mae_rounded),
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'per_rating_mae': {int(k): float(v) for k, v in rating_mae.items()}
    }
    
    return metrics


def evaluate_aspects(y_true, y_pred, aspect_names, output_dir: Path):
    """Evaluate aspect extraction task."""
    print("\n" + "="*80)
    print("ASPECT EXTRACTION EVALUATION")
    print("="*80)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    hamming = hamming_loss(y_true, y_pred)
    
    # Per-aspect metrics
    print("\nPer-Aspect Metrics:")
    print(f"{'Aspect':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        print(f"{aspect:<25} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
        aspect_metrics[aspect] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    print(f"\n{'Macro-Average':<25} {precision.mean():<12.4f} {recall.mean():<12.4f} {f1.mean():<12.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    
    # Plot per-aspect F1-scores
    plt.figure(figsize=(14, 6))
    x = np.arange(len(aspect_names))
    plt.bar(x, f1, alpha=0.7, color='teal', edgecolor='black')
    plt.axhline(f1.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean F1: {f1.mean():.3f}')
    plt.xlabel('Product Aspect', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Aspect Extraction: F1-Scores per Aspect', fontsize=14, fontweight='bold')
    plt.xticks(x, aspect_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'aspect_f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics = {
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'hamming_loss': float(hamming),
        'per_aspect': aspect_metrics
    }
    
    return metrics


def run_evaluation(args):
    """Main evaluation function."""
    print("="*80)
    print("MULTI-TASK REVIEW ANALYSIS - EVALUATION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint_path = Path(args.checkpoint_path)
    model, checkpoint = load_model(checkpoint_path, device)
    
    # Load test data
    print("\nLoading test dataset...")
    test_df = pd.read_parquet(Path(args.data_dir) / 'test.parquet')
    print(f"✓ Test set: {len(test_df)} samples\n")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint['config']['model_name'])
    
    # Create dataset and dataloader
    test_dataset = ReviewDataset(test_df, tokenizer, max_length=checkpoint['config']['max_length'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Get aspect names
    aspect_cols = [col for col in test_df.columns if col.startswith('aspect_')]
    aspect_names = [col.replace('aspect_', '').replace('_', ' ').title() for col in aspect_cols]
    
    # Run inference
    print("Running inference on test set...")
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_rating_preds = []
    all_rating_labels = []
    all_aspect_preds = []
    all_aspect_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions
            predictions = model.get_predictions(input_ids, attention_mask, aspect_threshold=args.aspect_threshold)
            
            # Store predictions
            all_sentiment_preds.extend(predictions['sentiment_pred'].cpu().numpy())
            all_sentiment_labels.extend(batch['sentiment_label'].numpy())
            
            all_rating_preds.extend(predictions['rating_pred'].cpu().numpy())
            all_rating_labels.extend(batch['rating'].numpy())
            
            all_aspect_preds.extend(predictions['aspect_pred'].cpu().numpy())
            all_aspect_labels.extend(batch['aspects'].numpy())
    
    # Convert to numpy arrays
    sentiment_preds = np.array(all_sentiment_preds)
    sentiment_labels = np.array(all_sentiment_labels)
    rating_preds = np.array(all_rating_preds)
    rating_labels = np.array(all_rating_labels)
    aspect_preds = np.array(all_aspect_preds)
    aspect_labels = np.array(all_aspect_labels)
    
    print(f"✓ Inference complete\n")
    
    # Evaluate each task
    sentiment_metrics = evaluate_sentiment(sentiment_labels, sentiment_preds, output_dir)
    rating_metrics = evaluate_rating(rating_labels, rating_preds, output_dir)
    aspect_metrics = evaluate_aspects(aspect_labels, aspect_preds, aspect_names, output_dir)
    
    # Combine all metrics
    all_metrics = {
        'test_samples': len(test_df),
        'model_checkpoint': str(checkpoint_path),
        'training_epoch': checkpoint['epoch'],
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sentiment': sentiment_metrics,
        'rating': rating_metrics,
        'aspects': aspect_metrics
    }
    
    # Save metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nSentiment Classification:")
    print(f"  Accuracy: {sentiment_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {sentiment_metrics['macro_f1']:.4f}")
    
    print(f"\nRating Prediction:")
    print(f"  MAE: {rating_metrics['mae']:.4f} stars")
    print(f"  RMSE: {rating_metrics['rmse']:.4f} stars")
    print(f"  R²: {rating_metrics['r2']:.4f}")
    
    print(f"\nAspect Extraction:")
    print(f"  Macro F1: {aspect_metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss: {aspect_metrics['hamming_loss']:.4f}")
    
    print(f"\n✓ Evaluation results saved to {output_dir}")
    print(f"✓ Visualizations saved to {output_dir}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80 + "\n")


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Review Analysis Model')
    
    parser.add_argument('--checkpoint_path', type=str, default='models/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--aspect_threshold', type=float, default=0.5,
                       help='Threshold for aspect predictions')
    
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
