# 🚀 Model Improvement Plan

## Current Baseline Performance

### Model: DistilBERT Multi-Task Learning (Baseline)
**Training Configuration:**
- Epochs: 4 (early stopping)
- Batch Size: 16
- Learning Rate: 2e-5
- Dropout: 0.3
- Loss Weights: Sentiment=1.0, Rating=0.5, Aspect=0.3

**Baseline Results:**
```
Sentiment Classification:
  - Test Accuracy: 53.57%
  - Macro F1: 0.36
  - Per-class: Positive (F1=0.69), Neutral (F1=0.40), Negative (F1=0.00)

Rating Prediction:
  - MAE: 1.37 stars
  - RMSE: 1.53 stars
  - R²: -0.40

Aspect Extraction:
  - Macro F1: 0.05
  - Hamming Loss: 0.34
```

**Problems Identified:**
1. ⚠️ **Class Imbalance:** Negative class completely ignored (0% F1)
2. ⚠️ **Overregularization:** Model predicts ~3 stars for everything
3. ⚠️ **Poor Generalization:** Similar predictions for different inputs
4. ⚠️ **Small Dataset:** Only 123 training samples
5. ⚠️ **Early Stopping:** Training stopped too early (epoch 4)

---

## 📋 Improvement Experiments

### Experiment 1: Class Weighting & Loss Rebalancing
**Hypothesis:** Negative class is underrepresented; increase its weight

**Changes:**
- Increase class weights for minority classes
- Rebalance multi-task loss weights
- Current weights: [2.05, 2.41, 0.48] → New: [4.0, 3.0, 0.5]

**Expected Impact:**
- ✅ Better negative sentiment detection
- ⚠️ May reduce overall accuracy but improve F1

---

### Experiment 2: Extended Training
**Hypothesis:** Model stopped training too early

**Changes:**
- Remove early stopping OR increase patience to 5
- Train for full 10 epochs
- Monitor for overfitting

**Expected Impact:**
- ✅ Better convergence
- ✅ Lower training loss
- ⚠️ Risk of overfitting on small dataset

---

### Experiment 3: Learning Rate Adjustment
**Hypothesis:** Learning rate 2e-5 may be too high for this small dataset

**Changes:**
- Try: 1e-5 (lower), 5e-5 (higher), 3e-5 (middle)
- Use learning rate warmup

**Expected Impact:**
- ✅ More stable training
- ✅ Better convergence

---

### Experiment 4: Increased Dropout
**Hypothesis:** Model is overfitting to training patterns

**Changes:**
- Current: 0.3 → New: 0.5
- Add dropout to task heads

**Expected Impact:**
- ✅ Better generalization
- ⚠️ May need more epochs

---

### Experiment 5: Data Augmentation
**Hypothesis:** 123 samples is too small; synthetic data can help

**Changes:**
- Back-translation (English → German → English)
- Synonym replacement
- Random word deletion
- Target: 2x data (246 samples)

**Expected Impact:**
- ✅ Better generalization
- ✅ More robust to variations
- ⚠️ May introduce noise

---

### Experiment 6: Ensemble Methods
**Hypothesis:** Multiple models can capture different patterns

**Changes:**
- Train 3 models with different seeds
- Average predictions
- Weighted voting

**Expected Impact:**
- ✅ More robust predictions
- ✅ Reduced variance
- ⚠️ 3x training time

---

### Experiment 7: Gradient Accumulation
**Hypothesis:** Larger effective batch size helps stability

**Changes:**
- Current: Batch size 16
- New: Batch size 4, Accumulation 4 (effective 16)
- Or: Batch size 8, Accumulation 4 (effective 32)

**Expected Impact:**
- ✅ More stable gradients
- ✅ Better convergence

---

## 🎯 Recommended Implementation Order

### Phase 1: Quick Wins (30 min)
1. ✅ **Experiment 2:** Extended training (10 epochs, no early stopping)
2. ✅ **Experiment 1:** Increase class weights
3. ✅ **Experiment 3:** Try learning rate 1e-5

### Phase 2: Medium Effort (1-2 hours)
4. ✅ **Experiment 4:** Increase dropout to 0.5
5. ✅ **Experiment 7:** Gradient accumulation

### Phase 3: Advanced (2-3 hours)
6. ✅ **Experiment 5:** Data augmentation
7. ✅ **Experiment 6:** Ensemble (if time permits)

---

## 📊 Comparison Methodology

### Metrics to Track:
```python
# For each experiment, track:
{
    "experiment_name": "...",
    "hyperparameters": {...},
    "training_time": "...",
    "results": {
        "sentiment": {
            "accuracy": ...,
            "macro_f1": ...,
            "per_class_f1": {...},
            "confusion_matrix": [[...]]
        },
        "rating": {
            "mae": ...,
            "rmse": ...,
            "r2": ...
        },
        "aspect": {
            "macro_f1": ...,
            "hamming_loss": ...
        }
    },
    "improvement_over_baseline": {
        "sentiment_acc": "+X%",
        "rating_mae": "-X stars",
        ...
    }
}
```

### Visualization Plan:
1. **Bar Charts:** Compare accuracy/F1 across experiments
2. **Line Charts:** Training/validation loss curves
3. **Confusion Matrices:** Side-by-side comparison
4. **Scatter Plots:** Rating predictions (actual vs predicted)
5. **Heatmaps:** Per-class improvements
6. **Radar Charts:** Multi-metric comparison

---

## 📝 Documentation Structure

For each experiment, document:

```markdown
## Experiment X: [Name]

### Motivation
Why we're trying this technique...

### Implementation
Code changes made...

### Training Process
- Epochs: X
- Time: X minutes
- Loss curves: [chart]

### Results
[Comparison table with baseline]

### Analysis
What improved? What got worse? Why?

### Conclusion
Keep this change? Yes/No and why.
```

---

## 🚀 Let's Start!

Which experiment would you like to try first?

**Recommended:** Start with Experiment 2 (Extended Training) + Experiment 1 (Class Weights)
This is the quickest way to see improvement!

Commands:
```bash
# Experiment 2 + 1: Extended training with better class weights
python scripts/train.py \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --early_stopping_patience 10 \
    --sentiment_class_weights 4.0 3.0 0.5 \
    --experiment_name "extended_training_reweighted"
```

Ready to start? 🚀
