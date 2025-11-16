"""
Quick Test - Model Inference with Neutral Detection

This script demonstrates how to use the trained model with neutral detection
for single review prediction.

Author: AI-Assisted Development
Date: November 16, 2025
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import sys

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Import model directly to avoid preprocessing dependencies
import importlib.util
model_spec = importlib.util.spec_from_file_location("model", "src/model.py")
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
MultiTaskReviewModel = model_module.MultiTaskReviewModel

from scripts.neutral_detection import NeutralDetector

def predict_review(text: str, model, tokenizer, neutral_detector, device):
    """
    Predict sentiment for a single review.
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        
        sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1).cpu().numpy()[0]
        rating_pred = outputs['rating_pred'].cpu().numpy()[0][0]
    
    # Apply neutral detection
    result = neutral_detector.predict_single(sentiment_probs)
    result['rating_prediction'] = float(rating_pred)
    
    return result

def main():
    print("🚀 Loading Model...")
    print("=" * 80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = MultiTaskReviewModel(
        num_sentiments=3,
        num_aspects=10,
        dropout_rate=0.3,
        pretrained_model='distilbert-base-uncased'
    ).to(device)
    
    # Load trained weights
    checkpoint_path = Path('experiments/exp2_expanded_data/checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        print("   Make sure you're running from the project root directory")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded (trained epoch {checkpoint['epoch']})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("✅ Tokenizer loaded")
    
    # Initialize neutral detector
    neutral_detector = NeutralDetector(confidence_threshold=0.65)
    print("✅ Neutral detector initialized (threshold: 0.65)")
    
    print("\n" + "=" * 80)
    print("📝 Testing on Sample Reviews")
    print("=" * 80)
    
    # Test examples
    examples = [
        {
            "text": "This product is absolutely amazing! Best purchase ever, highly recommend!",
            "expected": "Positive"
        },
        {
            "text": "Terrible quality, broke after one day. Complete waste of money. Very disappointed.",
            "expected": "Negative"
        },
        {
            "text": "It's okay, nothing special. Some features are good, others not so much.",
            "expected": "Neutral (low confidence)"
        },
        {
            "text": "Mixed feelings about this. Great design but poor functionality.",
            "expected": "Neutral (low confidence)"
        },
        {
            "text": "Exceeded my expectations! Fast shipping, excellent quality, perfect!",
            "expected": "Positive"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}: {example['expected']}")
        print(f"{'─' * 80}")
        print(f"Review: \"{example['text'][:100]}{'...' if len(example['text']) > 100 else ''}\"")
        print()
        
        result = predict_review(example['text'], model, tokenizer, neutral_detector, device)
        
        # Determine sentiment emoji
        emoji = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}
        
        print(f"🎯 Prediction:")
        print(f"   Sentiment:  {emoji.get(result['sentiment'], '❓')} {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Rating:     {'⭐' * int(round(result['rating_prediction']))} ({result['rating_prediction']:.2f} stars)")
        print(f"   Is Neutral: {'Yes' if result['is_neutral'] else 'No'}")
        print()
        print(f"📊 Probabilities:")
        print(f"   Negative: {result['probabilities']['Negative']:.1%}")
        print(f"   Positive: {result['probabilities']['Positive']:.1%}")
        print()
        print(f"💭 Reasoning:")
        print(f"   {result['reasoning']}")
    
    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    print()
    print("📚 Next Steps:")
    print("  1. Review the deployment decision: docs/deployment_decision.md")
    print("  2. Read implementation guide: docs/implementation_guide.md")
    print("  3. Set up API or batch processing")
    print("  4. Configure monitoring dashboard")
    print()
    print("💡 Tip: Adjust confidence threshold (0.60-0.70) based on your use case:")
    print("   - Lower (0.60): More neutral predictions (conservative)")
    print("   - Higher (0.70): Fewer neutral predictions (aggressive)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the project root directory")
        print("  2. Check that the model checkpoint exists:")
        print("     experiments/exp2_expanded_data/checkpoints/best_model.pt")
        print("  3. Ensure all dependencies are installed:")
        print("     pip install torch transformers pandas numpy")
