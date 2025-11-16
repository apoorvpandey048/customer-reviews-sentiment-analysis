# Implementation Guide - Production Deployment

**Model:** Experiment 2 with Neutral Detection  
**Date:** November 16, 2025  
**Status:** Ready for Implementation

---

## Quick Start

### 1. Test the Model Locally

```python
# File: test_model_inference.py
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import sys
sys.path.insert(0, str(Path.cwd()))

from src.model import MultiTaskReviewModel
from scripts.neutral_detection import NeutralDetector

# Load model
device = torch.device('cpu')  # Use 'cuda' for GPU
checkpoint = torch.load('experiments/exp2_expanded_data/checkpoints/best_model.pt', map_location=device)

model = MultiTaskReviewModel(
    num_sentiments=3,
    num_aspects=10,
    dropout_rate=0.3,
    pretrained_model='distilbert-base-uncased'
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Initialize neutral detector
neutral_detector = NeutralDetector(confidence_threshold=0.65)

# Test on sample review
review_text = "This product is okay, some good features but also some issues"

# Tokenize
encoding = tokenizer(
    review_text,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Get prediction
with torch.no_grad():
    outputs = model(
        input_ids=encoding['input_ids'].to(device),
        attention_mask=encoding['attention_mask'].to(device)
    )
    
    sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1).cpu().numpy()[0]
    rating_pred = outputs['rating_pred'].cpu().numpy()[0]

# Apply neutral detection
result = neutral_detector.predict_single(sentiment_probs)

print(f"Review: {review_text}")
print(f"\nSentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Rating: {rating_pred:.2f} stars")
print(f"Reasoning: {result['reasoning']}")
```

### 2. Run the Test

```powershell
cd "C:\Users\Apoor\customer_review_sentiment analysis\customer-reviews-sentiment-analysis"
python test_model_inference.py
```

---

## Production Integration

### Option A: REST API (Recommended)

Create a FastAPI service for model inference:

```python
# File: api/sentiment_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from typing import List, Dict
import numpy as np

app = FastAPI(title="Sentiment Analysis API")

# Load model at startup (do this once)
@app.on_event("startup")
async def load_model():
    global model, tokenizer, neutral_detector, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    from src.model import MultiTaskReviewModel
    model = MultiTaskReviewModel(
        num_sentiments=3,
        num_aspects=10,
        dropout_rate=0.3,
        pretrained_model='distilbert-base-uncased'
    ).to(device)
    
    checkpoint = torch.load('experiments/exp2_expanded_data/checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Initialize neutral detector
    from scripts.neutral_detection import NeutralDetector
    neutral_detector = NeutralDetector(confidence_threshold=0.65)
    
    print("✅ Model loaded successfully!")

# Define request/response models
class ReviewRequest(BaseModel):
    text: str
    review_id: str = None

class ReviewResponse(BaseModel):
    review_id: str
    sentiment: str
    sentiment_index: int
    confidence: float
    rating_prediction: float
    is_neutral: bool
    probabilities: Dict[str, float]
    reasoning: str

@app.post("/predict", response_model=ReviewResponse)
async def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment for a single review.
    """
    try:
        # Tokenize
        encoding = tokenizer(
            request.text,
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
        
        return ReviewResponse(
            review_id=request.review_id or "unknown",
            sentiment=result['sentiment'],
            sentiment_index=result['sentiment_index'],
            confidence=result['confidence'],
            rating_prediction=float(rating_pred),
            is_neutral=result['is_neutral'],
            probabilities=result['probabilities'],
            reasoning=result['reasoning']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(requests: List[ReviewRequest]):
    """
    Predict sentiment for multiple reviews (batch processing).
    """
    results = []
    for req in requests:
        result = await predict_sentiment(req)
        results.append(result)
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Install API dependencies:**
```powershell
pip install fastapi uvicorn
```

**Run the API:**
```powershell
python api/sentiment_api.py
```

**Test the API:**
```powershell
# Using curl (or Postman)
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Great product, highly recommend!\", \"review_id\": \"123\"}'
```

### Option B: Batch Processing Script

For processing large CSV files:

```python
# File: scripts/batch_predict.py
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.model import MultiTaskReviewModel
from scripts.neutral_detection import NeutralDetector

def batch_predict(input_csv: str, output_csv: str, batch_size: int = 16):
    """
    Predict sentiment for all reviews in a CSV file.
    
    Args:
        input_csv: Path to input CSV with 'text' column
        output_csv: Path to save results
        batch_size: Number of reviews to process at once
    """
    # Load data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiTaskReviewModel(
        num_sentiments=3,
        num_aspects=10,
        dropout_rate=0.3,
        pretrained_model='distilbert-base-uncased'
    ).to(device)
    
    checkpoint = torch.load('experiments/exp2_expanded_data/checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    neutral_detector = NeutralDetector(confidence_threshold=0.65)
    
    # Predict
    print(f"Processing {len(df)} reviews...")
    predictions = []
    confidences = []
    rating_preds = []
    is_neutral_list = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
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
            
            sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
            rating_pred = outputs['rating_pred'].cpu().numpy()
        
        # Apply neutral detection
        preds, confs, is_neutral = neutral_detector.predict_with_neutral(sentiment_probs)
        
        predictions.extend(preds.tolist())
        confidences.extend(confs.tolist())
        rating_preds.extend(rating_pred.flatten().tolist())
        is_neutral_list.extend(is_neutral.tolist())
    
    # Add results to dataframe
    df['predicted_sentiment'] = predictions
    df['sentiment_name'] = df['predicted_sentiment'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    df['confidence'] = confidences
    df['predicted_rating'] = rating_preds
    df['is_neutral'] = is_neutral_list
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Total reviews: {len(df)}")
    print(f"  Negative: {(df['predicted_sentiment'] == 0).sum()} ({(df['predicted_sentiment'] == 0).mean()*100:.1f}%)")
    print(f"  Neutral: {(df['predicted_sentiment'] == 1).sum()} ({(df['predicted_sentiment'] == 1).mean()*100:.1f}%)")
    print(f"  Positive: {(df['predicted_sentiment'] == 2).sum()} ({(df['predicted_sentiment'] == 2).mean()*100:.1f}%)")
    print(f"  Mean confidence: {df['confidence'].mean():.3f}")
    print(f"  Mean rating: {df['predicted_rating'].mean():.2f} stars")

if __name__ == "__main__":
    batch_predict(
        input_csv="data/new_reviews.csv",
        output_csv="data/new_reviews_predicted.csv",
        batch_size=16
    )
```

**Usage:**
```powershell
python scripts/batch_predict.py
```

---

## Monitoring Setup

### Create Monitoring Script

```python
# File: scripts/monitor_model.py
import pandas as pd
import numpy as np
from datetime import datetime
import json

def monitor_predictions(predictions_csv: str):
    """
    Generate monitoring report for model predictions.
    """
    df = pd.read_csv(predictions_csv)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(df),
        "sentiment_distribution": {
            "negative": int((df['predicted_sentiment'] == 0).sum()),
            "neutral": int((df['predicted_sentiment'] == 1).sum()),
            "positive": int((df['predicted_sentiment'] == 2).sum())
        },
        "confidence_stats": {
            "mean": float(df['confidence'].mean()),
            "std": float(df['confidence'].std()),
            "min": float(df['confidence'].min()),
            "max": float(df['confidence'].max()),
            "low_confidence_rate": float((df['confidence'] < 0.6).mean())
        },
        "neutral_rate": float((df['predicted_sentiment'] == 1).mean()),
        "mean_rating": float(df['predicted_rating'].mean())
    }
    
    # Check for alerts
    alerts = []
    if report["confidence_stats"]["mean"] < 0.7:
        alerts.append("⚠️ Average confidence is low (<0.7)")
    if report["neutral_rate"] > 0.25:
        alerts.append("⚠️ High neutral prediction rate (>25%)")
    if report["confidence_stats"]["low_confidence_rate"] > 0.15:
        alerts.append("⚠️ Many low-confidence predictions (>15%)")
    
    report["alerts"] = alerts
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/monitoring_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Monitoring Report - {report['timestamp']}")
    print(f"  Total predictions: {report['total_predictions']}")
    print(f"  Negative: {report['sentiment_distribution']['negative']} ({report['sentiment_distribution']['negative']/report['total_predictions']*100:.1f}%)")
    print(f"  Neutral: {report['sentiment_distribution']['neutral']} ({report['sentiment_distribution']['neutral']/report['total_predictions']*100:.1f}%)")
    print(f"  Positive: {report['sentiment_distribution']['positive']} ({report['sentiment_distribution']['positive']/report['total_predictions']*100:.1f}%)")
    print(f"  Mean confidence: {report['confidence_stats']['mean']:.3f}")
    print(f"  Low confidence rate: {report['confidence_stats']['low_confidence_rate']*100:.1f}%")
    
    if alerts:
        print(f"\n🚨 Alerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print(f"\n✅ No alerts - system operating normally")
    
    return report

if __name__ == "__main__":
    monitor_predictions("data/new_reviews_predicted.csv")
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Error analysis reviewed and approved
- [ ] Neutral detection tested on sample data
- [ ] API/batch script tested locally
- [ ] Monitoring dashboard configured
- [ ] Alerting rules defined
- [ ] Rollback plan documented

### Staging Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests (50+ sample reviews)
- [ ] Monitor for 24 hours
- [ ] Check confidence distribution
- [ ] Review neutral prediction rate (target: 10-20%)
- [ ] Verify latency < 200ms

### Production Deployment
- [ ] Deploy with 10% traffic split
- [ ] Monitor metrics hourly for first 24 hours
- [ ] Compare with baseline (if available)
- [ ] Collect user feedback
- [ ] Gradually increase to 100% over 1 week

### Post-Deployment
- [ ] Weekly performance reports
- [ ] Monthly retraining with new data
- [ ] Quarterly model updates
- [ ] Continuous threshold tuning

---

## Troubleshooting

### Issue: Low Confidence Predictions

**Symptoms:** Many predictions with confidence < 0.6

**Solutions:**
1. Lower neutral threshold to 0.60 (more conservative)
2. Route to human review queue
3. Check for data drift (compare with training distribution)
4. Consider retraining with more recent data

### Issue: High Neutral Rate

**Symptoms:** >25% of predictions classified as neutral

**Solutions:**
1. Increase threshold to 0.70 (more aggressive)
2. Validate with human review - are these truly ambiguous?
3. May indicate need for true 3-class model (Experiment 3)
4. Check review quality (spam, incomplete reviews)

### Issue: Accuracy Drop

**Symptoms:** Accuracy falls below 85%

**Solutions:**
1. Check for data drift (domain shift, seasonality)
2. Verify infrastructure (model version, dependencies)
3. Collect ground truth labels from recent predictions
4. Retrain model with recent data
5. Consider rolling back to previous version

---

## Next Steps

1. **This Week:**
   - [ ] Test inference locally
   - [ ] Set up API or batch script
   - [ ] Create monitoring dashboard

2. **Next Week:**
   - [ ] Deploy to staging
   - [ ] Monitor and tune threshold
   - [ ] Collect feedback

3. **Month 2:**
   - [ ] Deploy to production (gradual rollout)
   - [ ] Begin Experiment 3 data collection
   - [ ] Optimize performance

4. **Month 3:**
   - [ ] Train 3-class model
   - [ ] A/B test approaches
   - [ ] Full production deployment

---

## Support & Resources

- **Documentation:** `docs/deployment_decision.md`
- **Error Analysis:** `notebooks/error_analysis.ipynb`
- **Neutral Detection:** `scripts/neutral_detection.py`
- **Model Checkpoint:** `experiments/exp2_expanded_data/checkpoints/best_model.pt`

**Questions?** Contact the ML team or check the project README.

---

**Status:** ✅ Ready for Implementation  
**Last Updated:** November 16, 2025
