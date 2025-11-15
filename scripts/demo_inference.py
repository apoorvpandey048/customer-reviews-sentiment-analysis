"""
Demo Inference Script for Multi-Task Review Analysis
Demonstrates model predictions on sample reviews
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import create_model
from transformers import DistilBertTokenizer
import json


def load_trained_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = create_model(
        num_sentiment_classes=3,
        num_aspects=10,
        dropout=0.3
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model


def predict_review(model, tokenizer, review_text: str, device: str = "cpu"):
    """Make predictions for a single review"""
    
    # Tokenize
    encoding = tokenizer(
        review_text,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Process outputs
    sentiment_logits, rating_pred, aspect_logits = outputs
    
    # Get sentiment prediction
    sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0]
    sentiment_idx = torch.argmax(sentiment_probs).item()
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    
    # Get rating prediction
    predicted_rating = rating_pred[0].item()
    
    # Get aspect predictions (threshold at 0.5)
    aspect_probs = torch.sigmoid(aspect_logits)[0]
    aspect_names = [
        'Quality', 'Price', 'Value For Money', 'Shipping', 
        'Packaging', 'Customer Service', 'Ease Of Use',
        'Functionality', 'Durability', 'Overall Experience'
    ]
    predicted_aspects = [
        aspect_names[i] for i, prob in enumerate(aspect_probs)
        if prob > 0.5
    ]
    
    return {
        'review_text': review_text,
        'sentiment': {
            'label': sentiment_labels[sentiment_idx],
            'confidence': sentiment_probs[sentiment_idx].item(),
            'probabilities': {
                label: prob.item() 
                for label, prob in zip(sentiment_labels, sentiment_probs)
            }
        },
        'rating': {
            'predicted': round(predicted_rating, 2),
            'rounded': max(1, min(5, round(predicted_rating)))
        },
        'aspects': {
            'detected': predicted_aspects,
            'scores': {
                name: round(score.item(), 3)
                for name, score in zip(aspect_names, aspect_probs)
            }
        }
    }


def demo():
    """Run demo with sample reviews"""
    
    # Configuration
    checkpoint_path = "models/checkpoints/best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if model exists
    if not Path(checkpoint_path).exists():
        print("❌ ERROR: No trained model found!")
        print(f"   Expected: {checkpoint_path}")
        print("\nPlease train the model first:")
        print("   python scripts/train.py --batch_size 16 --num_epochs 10")
        return
    
    # Load model and tokenizer
    print("Initializing demo...")
    print(f"Device: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = load_trained_model(checkpoint_path, device)
    
    # Sample reviews for testing
    sample_reviews = [
        "This product is amazing! Great quality and fast shipping.",
        "Terrible experience. Poor quality and arrived damaged.",
        "It's okay, nothing special. Average product at best.",
        "Excellent value for money! Highly recommend to everyone.",
        "Not worth the price. Very disappointed with this purchase."
    ]
    
    print("\n" + "="*70)
    print("DEMO: Multi-Task Review Analysis")
    print("="*70 + "\n")
    
    # Make predictions
    results = []
    for i, review in enumerate(sample_reviews, 1):
        print(f"\n{'─'*70}")
        print(f"Review #{i}: \"{review}\"")
        print(f"{'─'*70}")
        
        prediction = predict_review(model, tokenizer, review, device)
        results.append(prediction)
        
        # Display results
        print(f"\n📊 Predictions:")
        print(f"   Sentiment: {prediction['sentiment']['label']} "
              f"(confidence: {prediction['sentiment']['confidence']:.2%})")
        print(f"   Rating: {prediction['rating']['predicted']:.2f} stars "
              f"(rounded: {prediction['rating']['rounded']}★)")
        print(f"   Aspects: {', '.join(prediction['aspects']['detected']) if prediction['aspects']['detected'] else 'None detected'}")
        
        # Show sentiment probabilities
        print(f"\n   Sentiment Breakdown:")
        for label, prob in prediction['sentiment']['probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"      {label:8s}: {bar:20s} {prob:.2%}")
    
    # Save results
    output_path = Path("results/demo_predictions.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"✓ Demo complete! Results saved to: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    demo()
