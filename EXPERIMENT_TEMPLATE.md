# Experiment Documentation Template

Use this template for each experiment to maintain consistent documentation.

---

## Experiment: [NAME]

**Date:** [YYYY-MM-DD]  
**Experiment ID:** [exp_number]  
**Status:** [Completed / In Progress / Failed]

---

### 1. Motivation & Hypothesis

**Problem Identified:**
- [What specific issue are we trying to solve?]
- [What metric is underperforming?]

**Hypothesis:**
- [What change do we expect will improve the model?]
- [Why do we think this will work?]

**Expected Improvements:**
- Metric 1: [Current value] → [Target value]
- Metric 2: [Current value] → [Target value]

---

### 2. Implementation Details

**Changes Made:**
```
Parameter/Component | Baseline Value | Experiment Value | Reason
--------------------|----------------|------------------|--------
[e.g., class_weight_negative] | 2.05 | 4.0 | Increase penalty for misclassifying negative reviews
[e.g., num_epochs] | 4 | 10 | Allow more training time
```

**Code Changes:**
- File: [path/to/file]
  - Line [X]: [Description of change]
- File: [path/to/file]
  - Line [Y]: [Description of change]

**Training Command:**
```bash
python scripts/train.py \
    --experiment_name=[exp_name] \
    --parameter1=[value1] \
    --parameter2=[value2] \
    ...
```

---

### 3. Training Results

**Training Configuration:**
- Epochs: [number]
- Batch Size: [number]
- Learning Rate: [value]
- Early Stopping: [Yes/No, patience=X]
- Device: [CPU/GPU]

**Training Metrics:**
```
Epoch | Train Loss | Val Loss | Best? | Notes
------|------------|----------|-------|------
1     | X.XXX      | X.XXX    | ✓     | 
2     | X.XXX      | X.XXX    |       |
...
```

**Best Checkpoint:**
- Epoch: [number]
- Validation Loss: [value]
- Training Time: [X minutes]

---

### 4. Test Results

#### 4.1 Sentiment Classification

**Overall Metrics:**
```
Metric           | Baseline | Experiment | Change    | % Change
-----------------|----------|------------|-----------|----------
Accuracy         | 53.57%   | XX.XX%     | ±X.XX%    | ±X.X%
Macro F1         | 0.36     | X.XX       | ±X.XX     | ±X.X%
Weighted F1      | X.XX     | X.XX       | ±X.XX     | ±X.X%
```

**Per-Class Performance:**
```
Class     | Baseline F1 | Experiment F1 | Change  | Analysis
----------|-------------|---------------|---------|----------
Negative  | 0.00        | X.XX          | +X.XX   | [Why did it improve/worsen?]
Neutral   | 0.40        | X.XX          | ±X.XX   | [Why did it improve/worsen?]
Positive  | 0.69        | X.XX          | ±X.XX   | [Why did it improve/worsen?]
```

**Confusion Matrix Analysis:**
- ![Confusion Matrix](path/to/confusion_matrix.png)
- **Observations:**
  - [What patterns do we see?]
  - [Which classes are being confused?]
  - [Did the change help with specific misclassifications?]

#### 4.2 Rating Prediction

**Overall Metrics:**
```
Metric     | Baseline | Experiment | Change    | % Change
-----------|----------|------------|-----------|----------
MAE        | 1.37     | X.XX       | ±X.XX     | ±X.X%
RMSE       | 1.53     | X.XX       | ±X.XX     | ±X.X%
R²         | -0.40    | X.XX       | ±X.XX     | ±X.X%
```

**Prediction Analysis:**
- ![Rating Scatter Plot](path/to/rating_scatter.png)
- **Observations:**
  - [Is the model still over-regularizing (predicting ~3 for everything)?]
  - [What's the prediction variance?]
  - [Are extreme ratings (1, 5) being predicted correctly?]

#### 4.3 Aspect Extraction

**Overall Metrics:**
```
Metric         | Baseline | Experiment | Change    | % Change
---------------|----------|------------|-----------|----------
Macro F1       | 0.05     | X.XX       | ±X.XX     | ±X.X%
Hamming Loss   | 0.34     | X.XX       | ±X.XX     | ±X.X%
```

**Per-Aspect Performance:**
```
Aspect       | Baseline F1 | Experiment F1 | Change  
-------------|-------------|---------------|---------
Price        | X.XX        | X.XX          | ±X.XX
Quality      | X.XX        | X.XX          | ±X.XX
Battery      | X.XX        | X.XX          | ±X.XX
...
```

---

### 5. Comparison Visualizations

Include all generated comparison charts:

1. **Overall Metrics Comparison**
   - ![Metrics Comparison](experiments/comparisons/exp_name_comparison.png)

2. **Confusion Matrices Side-by-Side**
   - Baseline: ![](results/sentiment_confusion_matrix.png)
   - Experiment: ![](results/exp_name/sentiment_confusion_matrix.png)

3. **Training Curves**
   - ![Training Loss](path/to/training_curves.png)

4. **Per-Class F1 Comparison**
   - ![F1 Comparison](path/to/f1_comparison.png)

---

### 6. Analysis & Interpretation

#### 6.1 What Improved?

**Positive Changes:**
- [Metric X improved by Y%]
  - **Reason:** [Why did this parameter change help?]
  - **Mechanism:** [What did it fix in the model behavior?]

#### 6.2 What Worsened?

**Negative Changes:**
- [Metric X worsened by Y%]
  - **Reason:** [Why did this parameter change hurt?]
  - **Trade-off:** [What trade-off was made?]

#### 6.3 Unexpected Results

**Surprises:**
- [What didn't behave as expected?]
- [Why might this have happened?]

---

### 7. Technical Insights

**What We Learned:**
- [Insight 1 about the model behavior]
- [Insight 2 about the dataset]
- [Insight 3 about hyperparameters]

**Why This Technique Worked/Failed:**
- [Explain the underlying mechanism]
- [Connect to ML theory (e.g., regularization, optimization, etc.)]

**Reading the Graphs:**
- [How to interpret the confusion matrix changes]
- [What the scatter plot tells us about prediction behavior]
- [How F1 scores reflect model confidence]

---

### 8. Conclusions & Recommendations

**Overall Assessment:**
- ✅ **Success** / ⚠️ **Mixed** / ❌ **Failure**

**Key Findings:**
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

**Recommendations:**
- **Keep:** [Which changes should be retained?]
- **Discard:** [Which changes should be reverted?]
- **Further Investigation:** [What should we try next?]

**Next Experiments:**
1. [Follow-up experiment 1]
2. [Follow-up experiment 2]

---

### 9. Files & Artifacts

**Model Checkpoint:**
- Path: `experiments/[exp_name]/checkpoints/best_model.pt`
- Size: [XX] MB
- SHA256: [hash]

**Results:**
- `experiments/[exp_name]/test_results.json`
- `experiments/[exp_name]/config.json`
- `results/[exp_name]/evaluation_metrics.json`

**Visualizations:**
- `results/[exp_name]/sentiment_confusion_matrix.png`
- `results/[exp_name]/rating_prediction_analysis.png`
- `results/[exp_name]/aspect_f1_scores.png`

**Logs:**
- `experiments/[exp_name]/logs/` (TensorBoard)

---

### 10. Reproducibility

**Command to Reproduce:**
```bash
# Training
python scripts/train.py \
    --experiment_name=[exp_name] \
    --seed=42 \
    [... all parameters]

# Evaluation
python scripts/evaluate.py \
    --checkpoint_path=experiments/[exp_name]/checkpoints/best_model.pt \
    --output_dir=results/[exp_name]

# Comparison
python compare_results.py [exp_name]
```

**Environment:**
- Python: [version]
- PyTorch: [version]
- Transformers: [version]
- Device: [CPU/GPU type]
- Random Seed: [number]

---

## Appendix: Sample Predictions

Include 3-5 examples showing how predictions changed:

**Example 1:**
- Text: "[review text]"
- True Labels: Sentiment=[X], Rating=[X], Aspects=[...]
- Baseline Predictions: Sentiment=[X], Rating=[X], Aspects=[...]
- Experiment Predictions: Sentiment=[X], Rating=[X], Aspects=[...]
- **Analysis:** [Why did the prediction change?]

---

*End of Experiment Report*
