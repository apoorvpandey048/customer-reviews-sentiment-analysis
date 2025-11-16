"""
FastAPI REST API for Sentiment Analysis Model
Serves the trained MultiTaskReviewModel with neutral detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import importlib.util
import sys
from pathlib import Path
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model directly to avoid preprocessing dependencies
spec = importlib.util.spec_from_file_location("model", project_root / "src" / "model.py")
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
MultiTaskReviewModel = model_module.MultiTaskReviewModel

from transformers import DistilBertTokenizer
from scripts.neutral_detection import NeutralDetector

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Multi-task sentiment analysis with rating prediction and aspect detection",
    version="1.0.0"
)

# Global variables for model, tokenizer, and detector
model = None
tokenizer = None
neutral_detector = None
device = None

# Pydantic models for request/response
class ReviewRequest(BaseModel):
    text: str = Field(..., description="Review text to analyze", min_length=1)
    confidence_threshold: Optional[float] = Field(0.65, description="Threshold for neutral detection (0.60-0.70)", ge=0.5, le=0.9)

class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of review texts to analyze")
    confidence_threshold: Optional[float] = Field(0.65, description="Threshold for neutral detection", ge=0.5, le=0.9)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    rating: float
    is_neutral: bool
    probabilities: Dict[str, float]
    aspects: Dict[str, bool]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str

@app.on_event("startup")
async def load_model():
    """Load model, tokenizer, and neutral detector on startup"""
    global model, tokenizer, neutral_detector, device
    
    print("🚀 Loading model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✅ Tokenizer loaded")
    
    # Load model
    checkpoint_path = project_root / 'experiments' / 'exp2_expanded_data' / 'checkpoints' / 'best_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultiTaskReviewModel(
        num_sentiments=3,  # Model trained with 3 classes
        num_aspects=10,
        dropout_rate=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded (trained epoch {checkpoint['epoch']})")
    
    # Initialize neutral detector
    neutral_detector = NeutralDetector(confidence_threshold=0.65)
    print("✅ Neutral detector initialized")
    
    print("🎉 API ready!")

def predict_review(text: str, threshold: float = 0.65) -> Dict:
    """Make prediction for a single review"""
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        
        # Sentiment
        sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1)[0]
        sentiment_pred = torch.argmax(sentiment_probs).item()
        confidence = sentiment_probs[sentiment_pred].item()
        
        # Sentiment mapping
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_emoji_map = {0: "😞", 1: "😐", 2: "😊"}
        sentiment_label = sentiment_map[sentiment_pred]
        sentiment_emoji = sentiment_emoji_map[sentiment_pred]
        
        # Apply neutral detection override if confidence is low
        is_neutral = (confidence < threshold) or (sentiment_pred == 1)
        if confidence < threshold and sentiment_pred != 1:
            sentiment_label = "Neutral"
            sentiment_emoji = "😐"
            is_neutral = True
        
        # Rating
        rating = outputs['rating_pred'][0].item()
        rating = max(1.0, min(5.0, rating))  # Clip to [1, 5]
        
        # Aspects
        aspect_probs = torch.sigmoid(outputs['aspect_logits'])[0]
        aspect_names = [
            'quality', 'price', 'delivery', 'packaging', 
            'customer_service', 'features', 'durability', 
            'ease_of_use', 'value_for_money', 'design'
        ]
        aspects = {
            name: (prob.item() > 0.5)
            for name, prob in zip(aspect_names, aspect_probs)
        }
        
        return {
            'sentiment': f"{sentiment_emoji} {sentiment_label}",
            'confidence': round(confidence * 100, 1),
            'rating': round(rating, 2),
            'is_neutral': is_neutral,
            'probabilities': {
                'Negative': round(sentiment_probs[0].item() * 100, 1),
                'Neutral': round(sentiment_probs[1].item() * 100, 1),
                'Positive': round(sentiment_probs[2].item() * 100, 1)
            },
            'aspects': aspects
        }

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    """Predict sentiment, rating, and aspects for a single review"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")
    
    try:
        result = predict_review(request.text, request.confidence_threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchReviewRequest):
    """Predict sentiment for multiple reviews"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 reviews per batch")
    
    try:
        predictions = []
        for text in request.reviews:
            if text.strip():
                result = predict_review(text, request.confidence_threshold)
                predictions.append(result)
        
        # Calculate summary statistics
        sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
        total_confidence = 0
        total_rating = 0
        
        for pred in predictions:
            sentiment_type = pred['sentiment'].split()[1]  # Extract sentiment from "😊 Positive"
            sentiment_counts[sentiment_type] += 1
            total_confidence += pred['confidence']
            total_rating += pred['rating']
        
        n = len(predictions)
        summary = {
            'total_reviews': n,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': round(total_confidence / n, 1) if n > 0 else 0,
            'average_rating': round(total_rating / n, 2) if n > 0 else 0
        }
        
        return {
            'predictions': predictions,
            'summary': summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Sentiment Analysis API...")
    print("📖 API documentation will be available at: http://127.0.0.1:8001/docs")
    print("🔍 Health check: http://127.0.0.1:8001/health")
    print("\n⏳ Loading model... (this may take a few seconds)\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
