"""
Neutral Sentiment Detection using Confidence Thresholding

This module implements a confidence-based approach to detect neutral sentiment
for reviews that fall between clearly positive and clearly negative.

Since the model was trained only on binary sentiment (positive/negative),
we use low confidence as a proxy for neutral sentiment.

Author: AI-Assisted Development
Date: November 16, 2025
"""

import numpy as np
from typing import Tuple, Dict


class NeutralDetector:
    """
    Detects neutral sentiment using confidence thresholding.
    
    The model outputs probabilities for negative, neutral (unused), and positive.
    When the maximum probability is below a threshold, we classify as neutral.
    
    Attributes:
        confidence_threshold (float): Confidence below which we predict neutral
        sentiment_labels (dict): Mapping of sentiment indices to names
    """
    
    def __init__(self, confidence_threshold: float = 0.65):
        """
        Initialize neutral detector.
        
        Args:
            confidence_threshold: Confidence below which to predict neutral.
                                 Default 0.65 means if max(probs) < 0.65, predict neutral.
                                 
        Recommended thresholds:
            - 0.60: More neutral predictions (higher recall for neutral)
            - 0.65: Balanced (recommended starting point)
            - 0.70: Fewer neutral predictions (higher precision for positive/negative)
        """
        self.confidence_threshold = confidence_threshold
        self.sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def predict_with_neutral(
        self, 
        sentiment_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict sentiment with neutral detection.
        
        Args:
            sentiment_probs: Array of shape (batch_size, num_classes) with probabilities
            
        Returns:
            predictions: Array of predicted class indices (0=Neg, 1=Neutral, 2=Pos)
            confidences: Array of confidence scores
            is_neutral: Boolean array indicating neutral predictions
            
        Example:
            >>> probs = np.array([[0.45, 0, 0.55], [0.1, 0, 0.9], [0.6, 0, 0.4]])
            >>> detector = NeutralDetector(confidence_threshold=0.65)
            >>> preds, confs, is_neutral = detector.predict_with_neutral(probs)
            >>> print(preds)  # [1, 2, 0] - first is neutral due to low confidence
        """
        # Get max probability and predicted class
        confidences = np.max(sentiment_probs, axis=1)
        raw_predictions = np.argmax(sentiment_probs, axis=1)
        
        # Apply neutral detection
        is_neutral = confidences < self.confidence_threshold
        predictions = raw_predictions.copy()
        predictions[is_neutral] = 1  # Set to neutral (index 1)
        
        return predictions, confidences, is_neutral
    
    def predict_single(
        self, 
        sentiment_probs: np.ndarray
    ) -> Dict[str, any]:
        """
        Predict sentiment for a single review with detailed output.
        
        Args:
            sentiment_probs: Array of shape (num_classes,) with probabilities
            
        Returns:
            Dictionary with:
                - sentiment: Predicted sentiment name
                - sentiment_index: Predicted class index
                - confidence: Confidence score
                - is_neutral: Whether classified as neutral
                - probabilities: Original probability distribution
                
        Example:
            >>> probs = np.array([0.45, 0, 0.55])
            >>> detector = NeutralDetector(confidence_threshold=0.65)
            >>> result = detector.predict_single(probs)
            >>> print(result)
            {
                'sentiment': 'Neutral',
                'sentiment_index': 1,
                'confidence': 0.55,
                'is_neutral': True,
                'probabilities': {'Negative': 0.45, 'Neutral': 0.0, 'Positive': 0.55}
            }
        """
        confidence = np.max(sentiment_probs)
        raw_prediction = np.argmax(sentiment_probs)
        
        # Check if neutral
        is_neutral = confidence < self.confidence_threshold
        prediction = 1 if is_neutral else raw_prediction
        
        return {
            'sentiment': self.sentiment_labels[prediction],
            'sentiment_index': int(prediction),
            'confidence': float(confidence),
            'is_neutral': bool(is_neutral),
            'probabilities': {
                'Negative': float(sentiment_probs[0]),
                'Neutral': float(sentiment_probs[1]),
                'Positive': float(sentiment_probs[2])
            },
            'reasoning': self._get_reasoning(confidence, is_neutral, raw_prediction)
        }
    
    def _get_reasoning(
        self, 
        confidence: float, 
        is_neutral: bool, 
        raw_prediction: int
    ) -> str:
        """Generate human-readable reasoning for the prediction."""
        if is_neutral:
            return (f"Classified as neutral because confidence ({confidence:.3f}) "
                   f"is below threshold ({self.confidence_threshold:.3f}). "
                   f"Model is uncertain between positive and negative.")
        else:
            sentiment = self.sentiment_labels[raw_prediction]
            return (f"Classified as {sentiment} with high confidence ({confidence:.3f}). "
                   f"Clear sentiment detected.")
    
    def evaluate_threshold(
        self,
        sentiment_probs: np.ndarray,
        true_labels: np.ndarray,
        thresholds: np.ndarray = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Evaluate different confidence thresholds to find optimal value.
        
        Useful for tuning the threshold on a validation set.
        
        Args:
            sentiment_probs: Array of shape (n_samples, num_classes)
            true_labels: Array of true sentiment labels
            thresholds: Array of thresholds to try (default: 0.5 to 0.9 in steps of 0.05)
            
        Returns:
            Dictionary mapping threshold -> metrics
            
        Example:
            >>> results = detector.evaluate_threshold(val_probs, val_labels)
            >>> best_threshold = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        """
        if thresholds is None:
            thresholds = np.arange(0.5, 0.91, 0.05)
        
        results = {}
        
        for thresh in thresholds:
            temp_detector = NeutralDetector(confidence_threshold=thresh)
            preds, confs, is_neutral = temp_detector.predict_with_neutral(sentiment_probs)
            
            # Calculate metrics
            accuracy = (preds == true_labels).mean()
            neutral_rate = is_neutral.mean()
            
            # For non-neutral predictions, check accuracy
            non_neutral_mask = ~is_neutral
            if non_neutral_mask.sum() > 0:
                non_neutral_accuracy = (preds[non_neutral_mask] == true_labels[non_neutral_mask]).mean()
            else:
                non_neutral_accuracy = 0.0
            
            results[float(thresh)] = {
                'accuracy': float(accuracy),
                'neutral_rate': float(neutral_rate),
                'non_neutral_accuracy': float(non_neutral_accuracy),
                'num_neutral': int(is_neutral.sum())
            }
        
        return results
    
    def get_optimal_threshold(
        self,
        sentiment_probs: np.ndarray,
        true_labels: np.ndarray,
        metric: str = 'accuracy'
    ) -> Tuple[float, Dict]:
        """
        Find optimal confidence threshold.
        
        Args:
            sentiment_probs: Validation set probabilities
            true_labels: Validation set true labels
            metric: Metric to optimize ('accuracy', 'non_neutral_accuracy', 'neutral_rate')
            
        Returns:
            best_threshold: Optimal threshold value
            metrics: Metrics at that threshold
        """
        results = self.evaluate_threshold(sentiment_probs, true_labels)
        best_threshold = max(results.items(), key=lambda x: x[1][metric])[0]
        return best_threshold, results[best_threshold]


# Example usage
if __name__ == "__main__":
    print("🎯 Neutral Sentiment Detection - Demo\n")
    
    # Create detector
    detector = NeutralDetector(confidence_threshold=0.65)
    
    # Example predictions
    examples = [
        {"probs": np.array([0.1, 0, 0.9]), "text": "Absolutely amazing product!"},
        {"probs": np.array([0.9, 0, 0.1]), "text": "Terrible quality, very disappointed"},
        {"probs": np.array([0.45, 0, 0.55]), "text": "It's okay, has some good and bad points"},
        {"probs": np.array([0.55, 0, 0.45]), "text": "Mixed feelings about this product"},
    ]
    
    print("Example Predictions:")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        result = detector.predict_single(example["probs"])
        print(f"\nExample {i}: {example['text']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities: Neg={result['probabilities']['Negative']:.3f}, "
              f"Pos={result['probabilities']['Positive']:.3f}")
        print(f"  Reasoning: {result['reasoning']}")
    
    print("\n" + "=" * 80)
    print("\n💡 Usage Tips:")
    print("  • Threshold 0.65 is a good starting point")
    print("  • Lower threshold (0.60) → More neutral predictions")
    print("  • Higher threshold (0.70) → Fewer neutral predictions")
    print("  • Use evaluate_threshold() on validation set to tune")
    print("  • Monitor neutral prediction rate in production")
    
    print("\n✅ Neutral detection ready for use!")
