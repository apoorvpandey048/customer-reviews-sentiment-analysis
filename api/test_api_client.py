"""
Example API Client - Test the Sentiment Analysis API
Shows how to make requests to the API
"""

import requests
import json
from typing import List, Dict

# API configuration
API_BASE_URL = "http://127.0.0.1:8001"

def test_health():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API is healthy!")
        print(f"   Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Device: {data['device']}")
        print(f"   Version: {data['version']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
    print()

def predict_single(text: str, threshold: float = 0.65):
    """Predict sentiment for a single review"""
    print(f"📝 Analyzing review: '{text[:50]}...'")
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={
            "text": text,
            "confidence_threshold": threshold
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Rating: {'⭐' * int(round(result['rating']))} ({result['rating']} stars)")
        print(f"   Is Neutral: {result['is_neutral']}")
        print(f"   Probabilities: Neg={result['probabilities']['Negative']}%, Pos={result['probabilities']['Positive']}%")
        
        # Show detected aspects
        detected_aspects = [k for k, v in result['aspects'].items() if v]
        if detected_aspects:
            print(f"   Aspects: {', '.join(detected_aspects)}")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()
    return response.json() if response.status_code == 200 else None

def predict_batch(reviews: List[str], threshold: float = 0.65):
    """Predict sentiment for multiple reviews"""
    print(f"📊 Analyzing batch of {len(reviews)} reviews...")
    
    response = requests.post(
        f"{API_BASE_URL}/predict_batch",
        json={
            "reviews": reviews,
            "confidence_threshold": threshold
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Batch prediction successful!")
        print(f"\n📈 Summary:")
        print(f"   Total reviews: {result['summary']['total_reviews']}")
        print(f"   Sentiment distribution:")
        for sentiment, count in result['summary']['sentiment_distribution'].items():
            print(f"      {sentiment}: {count}")
        print(f"   Average confidence: {result['summary']['average_confidence']}%")
        print(f"   Average rating: {result['summary']['average_rating']} stars")
        
        print(f"\n📝 Individual predictions:")
        for i, pred in enumerate(result['predictions'][:5], 1):  # Show first 5
            print(f"   {i}. {pred['sentiment']} ({pred['confidence']}%) - {pred['rating']} stars")
        
        if len(result['predictions']) > 5:
            print(f"   ... and {len(result['predictions']) - 5} more")
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()
    return response.json() if response.status_code == 200 else None

def main():
    """Run API tests"""
    print("=" * 60)
    print("🚀 SENTIMENT ANALYSIS API - CLIENT TEST")
    print("=" * 60)
    print()
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Single prediction - Positive
    predict_single(
        "This product is absolutely amazing! Great quality and fast delivery.",
        threshold=0.65
    )
    
    # Test 3: Single prediction - Negative
    predict_single(
        "Terrible quality. Broke after one day. Would not recommend.",
        threshold=0.65
    )
    
    # Test 4: Single prediction - Neutral (ambiguous)
    predict_single(
        "It's okay. Nothing special but does the job.",
        threshold=0.65
    )
    
    # Test 5: Batch prediction
    reviews = [
        "Excellent product! Highly recommended!",
        "Poor quality, not worth the money.",
        "Average product. Nothing impressive.",
        "Best purchase ever! 5 stars!",
        "Disappointed with this product.",
        "Good value for money.",
        "Not what I expected. Returning it.",
        "Pretty decent for the price.",
        "Amazing quality and fast shipping!",
        "Waste of money. Very disappointed."
    ]
    predict_batch(reviews, threshold=0.65)
    
    # Test 6: Different confidence threshold
    print("🎯 Testing with lower threshold (0.60) for more neutral predictions:")
    predict_single(
        "It's okay. Nothing special but does the job.",
        threshold=0.60
    )
    
    print("=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error!")
        print("   Make sure the API is running:")
        print("   python api/sentiment_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")
