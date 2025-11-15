"""
Training Pipeline for Multi-Task Review Analysis Model

This script trains the DistilBERT-based multi-task learning model on Amazon reviews data.
Implements training loop with:
- AdamW optimizer with weight decay
- Learning rate scheduling (linear warmup + decay)
- Class weights for imbalanced sentiment data
- Model checkpointing (best validation loss)
- TensorBoard logging
- Early stopping

Author: Apoorv Pandey
Course: CSE3712 - Big Data Analytics
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model import MultiTaskReviewModel, MultiTaskLoss, create_model
from src.dataset import ReviewDataset, create_dataloaders
from src.config import get_config


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Multi-task model
        dataloader: Training DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Multi-task loss function
        device: Device to run on
        epoch: Current epoch number
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_losses = {'total': 0, 'sentiment': 0, 'rating': 0, 'aspect': 0}
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_label = batch['sentiment_label'].to(device)
        rating = batch['rating'].to(device)
        aspects = batch['aspects'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        targets = {
            'sentiment_label': sentiment_label,
            'rating': rating,
            'aspects': aspects
        }
        
        loss, loss_dict = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average losses
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    
    return avg_losses


def evaluate(
    model: nn.Module,
    dataloader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    split: str = 'Val'
) -> dict:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Multi-task model
        dataloader: Validation/Test DataLoader
        loss_fn: Multi-task loss function
        device: Device to run on
        epoch: Current epoch number
        split: 'Val' or 'Test'
    
    Returns:
        Dictionary with average losses and metrics
    """
    model.eval()
    
    total_losses = {'total': 0, 'sentiment': 0, 'rating': 0, 'aspect': 0}
    num_batches = 0
    
    # For metrics calculation
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_rating_preds = []
    all_rating_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [{split}]')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_label = batch['sentiment_label'].to(device)
            rating = batch['rating'].to(device)
            aspects = batch['aspects'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            targets = {
                'sentiment_label': sentiment_label,
                'rating': rating,
                'aspects': aspects
            }
            
            loss, loss_dict = loss_fn(outputs, targets)
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1
            
            # Store predictions for metrics
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(sentiment_label.cpu().numpy())
            
            all_rating_preds.extend(outputs['rating_pred'].squeeze().cpu().numpy())
            all_rating_labels.extend(rating.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average losses
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    
    # Calculate metrics
    sentiment_acc = np.mean(np.array(all_sentiment_preds) == np.array(all_sentiment_labels))
    rating_mae = np.mean(np.abs(np.array(all_rating_preds) - np.array(all_rating_labels)))
    rating_rmse = np.sqrt(np.mean((np.array(all_rating_preds) - np.array(all_rating_labels)) ** 2))
    
    avg_losses['sentiment_acc'] = sentiment_acc
    avg_losses['rating_mae'] = rating_mae
    avg_losses['rating_rmse'] = rating_rmse
    
    return avg_losses


def train(config: dict):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary with hyperparameters
    """
    print("="*80)
    print("MULTI-TASK REVIEW ANALYSIS - TRAINING PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Create output directories
    output_dir = Path(config['output_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    logs_dir = output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_serializable = convert_to_serializable(config)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"✓ Configuration saved to {output_dir / 'config.json'}\n")
    
    # Load data
    print("Loading datasets...")
    data_dir = Path(config['data_dir'])
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    print(f"✓ Tokenizer loaded: {config['model_name']}\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        train_df, val_df, test_df, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        num_workers=config['num_workers']
    )
    print(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print(f"✓ Class weights: {class_weights.numpy()}\n")
    
    # Create model and loss function
    print("Creating model...")
    model_config = {
        'num_sentiments': 3,
        'num_aspects': 10,
        'dropout_rate': config['dropout_rate'],
        'freeze_bert': config['freeze_bert'],
        'pretrained_model': config['model_name'],
        'sentiment_class_weights': class_weights.tolist(),
        'sentiment_weight': config['sentiment_weight'],
        'rating_weight': config['rating_weight'],
        'aspect_weight': config['aspect_weight']
    }
    
    model, loss_fn = create_model(model_config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}\n")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"✓ Scheduler: Linear warmup + decay ({warmup_steps} warmup steps)\n")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=logs_dir)
    print(f"✓ TensorBoard logging to {logs_dir}\n")
    
    # Training loop
    print("="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        
        # Validate
        val_losses = evaluate(model, val_loader, loss_fn, device, epoch, split='Val')
        
        # Log to TensorBoard
        for key in ['total', 'sentiment', 'rating', 'aspect']:
            writer.add_scalars(f'Loss/{key}', {
                'train': train_losses[key],
                'val': val_losses[key]
            }, epoch)
        
        writer.add_scalar('Metrics/sentiment_acc', val_losses['sentiment_acc'], epoch)
        writer.add_scalar('Metrics/rating_mae', val_losses['rating_mae'], epoch)
        writer.add_scalar('Metrics/rating_rmse', val_losses['rating_rmse'], epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Print epoch summary
        print(f"\n{'Train':<10} | Loss: {train_losses['total']:.4f} | "
              f"Sent: {train_losses['sentiment']:.4f} | "
              f"Rating: {train_losses['rating']:.4f} | "
              f"Aspect: {train_losses['aspect']:.4f}")
        
        print(f"{'Val':<10} | Loss: {val_losses['total']:.4f} | "
              f"Sent: {val_losses['sentiment']:.4f} | "
              f"Rating: {val_losses['rating']:.4f} | "
              f"Aspect: {val_losses['aspect']:.4f}")
        
        print(f"{'Metrics':<10} | Sentiment Acc: {val_losses['sentiment_acc']:.4f} | "
              f"Rating MAE: {val_losses['rating_mae']:.4f} | "
              f"Rating RMSE: {val_losses['rating_rmse']:.4f}")
        
        # Save checkpoint if best validation loss
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config
            }, checkpoint_path)
            
            print(f"\n✓ Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"\n  Patience: {patience_counter}/{config['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n✗ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80 + "\n")
    
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_losses = evaluate(model, test_loader, loss_fn, device, epoch=0, split='Test')
    
    print(f"\n{'Test':<10} | Loss: {test_losses['total']:.4f} | "
          f"Sent: {test_losses['sentiment']:.4f} | "
          f"Rating: {test_losses['rating']:.4f} | "
          f"Aspect: {test_losses['aspect']:.4f}")
    
    print(f"{'Metrics':<10} | Sentiment Acc: {test_losses['sentiment_acc']:.4f} | "
          f"Rating MAE: {test_losses['rating_mae']:.4f} | "
          f"Rating RMSE: {test_losses['rating_rmse']:.4f}")
    
    # Save test results
    test_results = {
        'best_epoch': checkpoint['epoch'],
        'test_losses': test_losses,
        'val_losses': checkpoint['val_losses']
    }
    
    # Convert numpy types to Python types for JSON serialization
    test_results_serializable = convert_to_serializable(test_results)
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    print(f"\n✓ Test results saved to {output_dir / 'test_results.json'}")
    
    writer.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best Model: {checkpoint_dir / 'best_model.pt'}")
    print(f"Test Accuracy (Sentiment): {test_losses['sentiment_acc']:.4f}")
    print(f"Test MAE (Rating): {test_losses['rating_mae']:.4f}")
    print("="*80 + "\n")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train Multi-Task Review Analysis Model')
    
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save model checkpoints and logs')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='HuggingFace model identifier')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--sentiment_weight', type=float, default=1.0,
                       help='Loss weight for sentiment task')
    parser.add_argument('--rating_weight', type=float, default=0.5,
                       help='Loss weight for rating task')
    parser.add_argument('--aspect_weight', type=float, default=0.5,
                       help='Loss weight for aspect task')
    parser.add_argument('--freeze_bert', action='store_true',
                       help='Freeze BERT weights')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    config = vars(args)
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()
