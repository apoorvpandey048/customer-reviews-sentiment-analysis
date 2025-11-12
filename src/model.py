"""
Multi-Task Learning Model for Amazon Reviews Analysis

This module implements a DistilBERT-based multi-task learning architecture that jointly predicts:
1. Sentiment classification (3 classes: Negative, Neutral, Positive)
2. Rating prediction (regression: 1-5 stars)
3. Product aspect extraction (multi-label classification: 10 aspects)

Author: Apoorv Pandey
Course: CSE3712 - Big Data Analytics
Date: November 2025
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Tuple


class MultiTaskReviewModel(nn.Module):
    """
    Multi-task learning model for review analysis using DistilBERT as backbone.
    
    Architecture:
    - Shared DistilBERT encoder (distilbert-base-uncased)
    - Sentiment classification head (3 classes)
    - Rating regression head (1-5 stars)
    - Aspect extraction head (10 binary classifications)
    
    The model uses a shared representation from DistilBERT and task-specific heads
    to jointly optimize all three objectives.
    """
    
    def __init__(
        self,
        num_sentiments: int = 3,
        num_aspects: int = 10,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        pretrained_model: str = "distilbert-base-uncased"
    ):
        """
        Initialize the multi-task model.
        
        Args:
            num_sentiments: Number of sentiment classes (default: 3)
            num_aspects: Number of product aspects (default: 10)
            dropout_rate: Dropout probability for regularization (default: 0.3)
            freeze_bert: Whether to freeze DistilBERT weights (default: False)
            pretrained_model: HuggingFace model identifier (default: distilbert-base-uncased)
        """
        super(MultiTaskReviewModel, self).__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        self.hidden_size = self.bert.config.hidden_size  # 768 for base model
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Shared dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Task-specific heads
        self._build_sentiment_head(num_sentiments, dropout_rate)
        self._build_rating_head(dropout_rate)
        self._build_aspect_head(num_aspects, dropout_rate)
        
    def _build_sentiment_head(self, num_sentiments: int, dropout_rate: float):
        """
        Build sentiment classification head.
        
        Architecture: 
        - Linear(768 -> 256) + ReLU + Dropout
        - Linear(256 -> num_sentiments)
        """
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_sentiments)
        )
    
    def _build_rating_head(self, dropout_rate: float):
        """
        Build rating regression head.
        
        Architecture:
        - Linear(768 -> 128) + ReLU + Dropout
        - Linear(128 -> 1) + Sigmoid (scaled to 1-5 range)
        """
        self.rating_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def _build_aspect_head(self, num_aspects: int, dropout_rate: float):
        """
        Build aspect extraction head (multi-label classification).
        
        Architecture:
        - Linear(768 -> 256) + ReLU + Dropout
        - Linear(256 -> num_aspects)
        
        Output is logits; BCEWithLogitsLoss handles sigmoid internally.
        """
        self.aspect_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_aspects)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
        
        Returns:
            Dictionary containing:
                - sentiment_logits: Shape (batch_size, num_sentiments)
                - rating_pred: Shape (batch_size, 1), values in range [1, 5]
                - aspect_logits: Shape (batch_size, num_aspects)
        """
        # Get DistilBERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        cls_output = self.dropout(cls_output)
        
        # Task-specific predictions
        sentiment_logits = self.sentiment_head(cls_output)
        rating_raw = self.rating_head(cls_output)
        aspect_logits = self.aspect_head(cls_output)
        
        # Scale rating to 1-5 range using sigmoid: 1 + 4 * sigmoid(x)
        rating_pred = 1.0 + 4.0 * torch.sigmoid(rating_raw)
        
        return {
            'sentiment_logits': sentiment_logits,
            'rating_pred': rating_pred,
            'aspect_logits': aspect_logits
        }
    
    def get_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Get final predictions (for inference).
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask
            aspect_threshold: Threshold for aspect predictions (default: 0.5)
        
        Returns:
            Dictionary containing:
                - sentiment_pred: Predicted sentiment class (batch_size,)
                - sentiment_probs: Sentiment probabilities (batch_size, num_sentiments)
                - rating_pred: Predicted rating (batch_size,)
                - aspect_pred: Binary aspect predictions (batch_size, num_aspects)
                - aspect_probs: Aspect probabilities (batch_size, num_aspects)
        """
        outputs = self.forward(input_ids, attention_mask)
        
        # Sentiment predictions
        sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=-1)
        sentiment_pred = torch.argmax(sentiment_probs, dim=-1)
        
        # Rating predictions (already in 1-5 range)
        rating_pred = outputs['rating_pred'].squeeze(-1)
        
        # Aspect predictions
        aspect_probs = torch.sigmoid(outputs['aspect_logits'])
        aspect_pred = (aspect_probs >= aspect_threshold).float()
        
        return {
            'sentiment_pred': sentiment_pred,
            'sentiment_probs': sentiment_probs,
            'rating_pred': rating_pred,
            'aspect_pred': aspect_pred,
            'aspect_probs': aspect_probs
        }


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Computes weighted sum of:
    1. Sentiment classification loss (CrossEntropyLoss with class weights)
    2. Rating regression loss (MSELoss)
    3. Aspect extraction loss (BCEWithLogitsLoss)
    
    Loss weights can be adjusted to balance tasks.
    """
    
    def __init__(
        self,
        sentiment_weight: float = 1.0,
        rating_weight: float = 0.5,
        aspect_weight: float = 0.5,
        sentiment_class_weights: torch.Tensor = None
    ):
        """
        Initialize multi-task loss.
        
        Args:
            sentiment_weight: Weight for sentiment loss (default: 1.0)
            rating_weight: Weight for rating loss (default: 0.5)
            aspect_weight: Weight for aspect loss (default: 0.5)
            sentiment_class_weights: Class weights for imbalanced sentiment data
        """
        super(MultiTaskLoss, self).__init__()
        
        self.sentiment_weight = sentiment_weight
        self.rating_weight = rating_weight
        self.aspect_weight = aspect_weight
        
        # Loss functions
        self.sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_class_weights)
        self.rating_criterion = nn.MSELoss()
        self.aspect_criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth dictionary with keys:
                - sentiment_label: Shape (batch_size,)
                - rating: Shape (batch_size,)
                - aspects: Shape (batch_size, num_aspects)
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Sentiment loss
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment_logits'],
            targets['sentiment_label']
        )
        
        # Rating loss
        rating_loss = self.rating_criterion(
            outputs['rating_pred'].squeeze(-1),
            targets['rating'].float()
        )
        
        # Aspect loss
        aspect_loss = self.aspect_criterion(
            outputs['aspect_logits'],
            targets['aspects'].float()
        )
        
        # Weighted total loss
        total_loss = (
            self.sentiment_weight * sentiment_loss +
            self.rating_weight * rating_loss +
            self.aspect_weight * aspect_loss
        )
        
        # Return loss breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'sentiment': sentiment_loss.item(),
            'rating': rating_loss.item(),
            'aspect': aspect_loss.item()
        }
        
        return total_loss, loss_dict


def create_model(config: Dict = None) -> Tuple[MultiTaskReviewModel, MultiTaskLoss]:
    """
    Factory function to create model and loss function with configuration.
    
    Args:
        config: Configuration dictionary with model hyperparameters
    
    Returns:
        Tuple of (model, loss_function)
    """
    if config is None:
        config = {}
    
    # Model hyperparameters
    model = MultiTaskReviewModel(
        num_sentiments=config.get('num_sentiments', 3),
        num_aspects=config.get('num_aspects', 10),
        dropout_rate=config.get('dropout_rate', 0.3),
        freeze_bert=config.get('freeze_bert', False),
        pretrained_model=config.get('pretrained_model', 'distilbert-base-uncased')
    )
    
    # Loss function with class weights for imbalanced data
    sentiment_class_weights = config.get('sentiment_class_weights', None)
    if sentiment_class_weights is not None:
        sentiment_class_weights = torch.tensor(sentiment_class_weights, dtype=torch.float32)
    
    loss_fn = MultiTaskLoss(
        sentiment_weight=config.get('sentiment_weight', 1.0),
        rating_weight=config.get('rating_weight', 0.5),
        aspect_weight=config.get('aspect_weight', 0.5),
        sentiment_class_weights=sentiment_class_weights
    )
    
    return model, loss_fn


if __name__ == "__main__":
    """Test model creation and forward pass."""
    
    print("Testing Multi-Task Review Model...")
    print("="*60)
    
    # Create model
    model, loss_fn = create_model()
    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))  # vocab_size = 30522
    dummy_attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Sentiment logits shape: {outputs['sentiment_logits'].shape}")
    print(f"  Rating predictions shape: {outputs['rating_pred'].shape}")
    print(f"  Aspect logits shape: {outputs['aspect_logits'].shape}")
    
    # Test loss computation
    dummy_targets = {
        'sentiment_label': torch.randint(0, 3, (batch_size,)),
        'rating': torch.randint(1, 6, (batch_size,)),
        'aspects': torch.randint(0, 2, (batch_size, 10)).float()
    }
    
    total_loss, loss_dict = loss_fn(outputs, dummy_targets)
    
    print(f"\n✓ Loss computation successful")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Sentiment loss: {loss_dict['sentiment']:.4f}")
    print(f"  Rating loss: {loss_dict['rating']:.4f}")
    print(f"  Aspect loss: {loss_dict['aspect']:.4f}")
    
    # Test prediction mode
    with torch.no_grad():
        predictions = model.get_predictions(dummy_input_ids, dummy_attention_mask)
    
    print(f"\n✓ Prediction mode successful")
    print(f"  Sentiment predictions: {predictions['sentiment_pred']}")
    print(f"  Rating predictions: {predictions['rating_pred']}")
    print(f"  Aspect predictions shape: {predictions['aspect_pred'].shape}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
