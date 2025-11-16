# Model Improvement Workflow

Complete guide to running experiments, comparing results, and documenting improvements.

## 🎯 Current Status

**Baseline Performance (Already Trained):**
- ✅ Sentiment Accuracy: **53.57%** (Target: >60%)
- ✅ Rating MAE: **1.37 stars** (Target: <1.20)
- ✅ Aspect F1: **0.05** (Target: >0.10)

**Problems Identified:**
1. ❌ Negative class completely ignored (F1 = 0.00)
2. ❌ Model over-regularizes (predicts ~3 stars for everything)
3. ❌ Early stopping too aggressive (stopped at epoch 4)
4. ❌ Class imbalance not properly addressed
5. ❌ Small dataset (only 123 training samples)

---

## 🚀 Quick Start - Run Your First Experiment

### Step 1: Run Experiment 1 (Extended Training + Class Reweighting)

This experiment addresses problems #1, #3, and #4 above.

```bash
# Option A: Use the simplified script
python run_experiment.py

# Option B: Run directly with custom parameters
python scripts/train.py \
    --experiment_name=exp1_extended_reweighted \
    --num_epochs=10 \
    --early_stopping_patience=10 \
    --class_weight_negative=4.0 \
    --class_weight_neutral=3.0 \
    --class_weight_positive=0.5 \
    --sentiment_weight=1.5 \
    --batch_size=16 \
    --learning_rate=2e-5
```

**What this does:**
- 🔧 Increases negative class weight: 2.05 → 4.0 (to fix F1=0.00)
- 📈 Extends training: 4 epochs → 10 epochs (more learning time)
- ⚖️ Increases sentiment importance: 1.0 → 1.5 (prioritize classification)
- ⏱️ Takes ~2-3 minutes on CPU

### Step 2: Evaluate the Results

```bash
python scripts/evaluate.py \
    --checkpoint_path experiments/exp1_extended_reweighted/checkpoints/best_model.pt \
    --output_dir results/exp1
```

**Generated files:**
- `results/exp1/evaluation_metrics.json` - All metrics
- `results/exp1/sentiment_confusion_matrix.png` - Classification viz
- `results/exp1/rating_prediction_analysis.png` - Regression viz
- `results/exp1/aspect_f1_scores.png` - Multi-label viz

### Step 3: Compare with Baseline

```bash
python compare_results.py exp1_extended_reweighted
```

**Output:**
- 📊 Formatted comparison table (printed to console)
- 📈 Comparison bar charts (saved to `experiments/comparisons/`)
- ✅ Success/failure assessment
- 📝 Summary of improvements

### Step 4: Document Your Findings

```bash
# Copy the template
cp EXPERIMENT_TEMPLATE.md experiments/exp1_extended_reweighted.md

# Fill in the sections with your results
# See example below
```

---

## 📚 Planned Experiments

### Experiment 1: Extended Training + Class Reweighting ⭐ **START HERE**
**Problem Solved:** Negative class ignored, early stopping  
**Changes:**
- Epochs: 4 → 10
- Class weights: [2.05, 2.41, 0.48] → [4.0, 3.0, 0.5]
- Sentiment weight: 1.0 → 1.5

**Expected Improvements:**
- Negative F1: 0.00 → >0.20
- Overall Acc: 53.57% → >60%
- Rating MAE: 1.37 → <1.30

**Time:** ~2 minutes

---

### Experiment 2: Lower Learning Rate
**Problem Solved:** May be overshooting optimal weights  
**Changes:**
- Learning rate: 2e-5 → 1e-5
- Keep improvements from Exp 1

**Command:**
```bash
python scripts/train.py \
    --experiment_name=exp2_lower_lr \
    --num_epochs=10 \
    --learning_rate=1e-5 \
    --class_weight_negative=4.0 \
    --class_weight_neutral=3.0 \
    --class_weight_positive=0.5
```

---

### Experiment 3: Higher Dropout
**Problem Solved:** Potential overfitting  
**Changes:**
- Dropout: 0.3 → 0.5
- Keep improvements from Exp 1

**Command:**
```bash
python scripts/train.py \
    --experiment_name=exp3_higher_dropout \
    --num_epochs=10 \
    --dropout_rate=0.5 \
    --class_weight_negative=4.0 \
    --class_weight_neutral=3.0 \
    --class_weight_positive=0.5
```

---

### Experiment 4: Learning Rate Sweep
**Problem Solved:** Find optimal learning rate  
**Changes:**
- Try 3 learning rates: 1e-5, 3e-5, 5e-5
- Compare all three

**Command:**
```bash
# Run all three
python scripts/train.py --experiment_name=exp4_lr_1e5 --learning_rate=1e-5 --num_epochs=10
python scripts/train.py --experiment_name=exp4_lr_3e5 --learning_rate=3e-5 --num_epochs=10
python scripts/train.py --experiment_name=exp4_lr_5e5 --learning_rate=5e-5 --num_epochs=10
```

---

## 📊 How to Read the Results

### Understanding Confusion Matrices

```
                Predicted
              Neg  Neu  Pos
Actual Neg   [ 0    2    1 ]  ← Negative reviews
       Neu   [ 0    4    3 ]  ← Neutral reviews  
       Pos   [ 0    5   13 ]  ← Positive reviews
```

**What to look for:**
- ✅ **Diagonal values should be high** (correct predictions)
- ❌ **Off-diagonal = mistakes**
- 🔍 **First row all zeros?** → Model never predicts negative (problem!)

**Improvements to watch:**
- Baseline: Neg row all zeros → Exp: Some diagonal values
- Baseline: Everything predicted as Pos → Exp: More balanced

---

### Understanding Rating Scatter Plots

```
True Rating (y-axis) vs Predicted Rating (x-axis)
5 ★  •     •    •
4 ★    •  •••  •
3 ★  •••••••••••••  ← Problem: Everything at 3!
2 ★    •  •••  •
1 ★  •     •    •
     1   2   3   4   5
```

**What to look for:**
- ✅ **Points along diagonal** = accurate predictions
- ❌ **Vertical line at x=3** = over-regularization (predicts ~3 for all)
- ✅ **Spread out points** = model making varied predictions

**Improvements to watch:**
- Baseline: Vertical line at 3 → Exp: More scattered (using full range)
- MAE: 1.37 → Lower (predictions closer to true values)

---

### Understanding F1 Scores

**Formula:** F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Interpretation:**
- **F1 = 0.00** → Model never predicts this class (or always wrong)
- **F1 = 0.50** → Model is somewhat reliable
- **F1 = 0.70+** → Good performance
- **F1 = 1.00** → Perfect (rare)

**Per-Class Example:**
```
Class      Baseline F1  Exp F1   Interpretation
Negative   0.00         0.35     ✅ Huge win! Now detects negatives
Neutral    0.40         0.45     ✅ Small improvement
Positive   0.69         0.67     ⚠️  Slight drop (acceptable trade-off)
```

**Overall Macro F1:** Average of all three (want this to increase)

---

## 📝 Documentation Checklist

For each experiment, document:

### 1. Motivation ✅
- [ ] What problem are we solving?
- [ ] Why this approach?
- [ ] Expected improvements

### 2. Implementation ✅
- [ ] Parameter changes table
- [ ] Training command
- [ ] Code modifications (if any)

### 3. Results ✅
- [ ] Training time & best epoch
- [ ] All metrics in comparison tables
- [ ] Confusion matrix analysis
- [ ] Scatter plot analysis

### 4. Analysis ✅
- [ ] What improved and why?
- [ ] What worsened and why?
- [ ] Unexpected results?

### 5. Insights ✅
- [ ] How to read the graphs for this experiment
- [ ] Why this technique worked/failed
- [ ] Connection to ML theory

### 6. Conclusion ✅
- [ ] Keep or discard changes?
- [ ] Next experiments to try?
- [ ] Recommendations

---

## 🎨 Visualization Guide

### 1. Comparison Bar Charts
**Shows:** Baseline vs Experiment side-by-side  
**Generated by:** `compare_results.py`  
**Explains:** Overall metric improvements at a glance

### 2. Confusion Matrices
**Shows:** Classification errors per class  
**Generated by:** `scripts/evaluate.py`  
**Explains:** Which classes are confused, which are improving

### 3. Rating Scatter Plots
**Shows:** Predicted vs True ratings  
**Generated by:** `scripts/evaluate.py`  
**Explains:** Over-regularization, prediction spread

### 4. F1 Bar Charts
**Shows:** Per-class or per-aspect F1 scores  
**Generated by:** `scripts/evaluate.py`  
**Explains:** Individual component performance

### 5. Training Curves (TensorBoard)
**Shows:** Loss over time during training  
**Access:** `tensorboard --logdir experiments/[exp_name]/logs`  
**Explains:** Learning progress, overfitting, convergence

---

## ⚡ Quick Reference Commands

```bash
# Run experiment with custom name
python scripts/train.py --experiment_name=my_exp [parameters]

# Evaluate experiment
python scripts/evaluate.py \
    --checkpoint_path experiments/my_exp/checkpoints/best_model.pt \
    --output_dir results/my_exp

# Compare with baseline
python compare_results.py my_exp

# View training curves
tensorboard --logdir experiments/my_exp/logs

# Test on custom reviews
python scripts/demo_inference.py
```

---

## 📁 File Structure After Experiments

```
experiments/
├── exp1_extended_reweighted/
│   ├── checkpoints/best_model.pt
│   ├── config.json
│   ├── test_results.json
│   └── logs/ (TensorBoard)
├── exp2_lower_lr/
│   └── ... (same structure)
├── comparisons/
│   ├── exp1_extended_reweighted_comparison.png
│   └── exp2_lower_lr_comparison.png
└── exp1_extended_reweighted.md (documentation)

results/
├── exp1/
│   ├── evaluation_metrics.json
│   ├── sentiment_confusion_matrix.png
│   ├── rating_prediction_analysis.png
│   └── aspect_f1_scores.png
└── exp2/
    └── ... (same structure)
```

---

## 🎓 Best Practices

### Running Experiments
1. ✅ **One change at a time** (unless combining related changes)
2. ✅ **Use descriptive experiment names** (e.g., `exp1_extended_reweighted`)
3. ✅ **Document before running** (fill motivation section first)
4. ✅ **Save all outputs** (don't overwrite previous results)

### Comparing Results
1. ✅ **Always compare to baseline** (not just previous experiment)
2. ✅ **Look at multiple metrics** (don't optimize for just one)
3. ✅ **Check all three tasks** (sentiment, rating, aspect)
4. ✅ **Inspect visualizations** (numbers alone don't tell full story)

### Documentation
1. ✅ **Explain WHY** (not just what changed)
2. ✅ **Include graphs** (visual evidence of improvement)
3. ✅ **Note surprises** (unexpected results are valuable)
4. ✅ **Make it reproducible** (include exact commands)

---

## 🎯 Success Criteria

Your experiments are successful when:

1. **Negative F1 > 0.20** (currently 0.00)
2. **Overall Accuracy > 60%** (currently 53.57%)
3. **Rating MAE < 1.20** (currently 1.37)
4. **Rating predictions use full range** (not just ~3)
5. **Comprehensive documentation** (anyone can reproduce)

---

## 📞 Troubleshooting

**Problem:** Training is too slow  
**Solution:** Reduce batch size or epochs, or use gradient accumulation

**Problem:** Out of memory  
**Solution:** Lower batch size: `--batch_size=8`

**Problem:** Results not improving  
**Solution:** Check TensorBoard logs for training curves, may need different approach

**Problem:** Can't compare experiments  
**Solution:** Ensure test_results.json exists in both baseline and experiment dirs

---

## 🎉 Next Steps

1. **Run Experiment 1** (~2 min)
2. **Compare results** (~1 min)
3. **Document findings** (~10 min)
4. **If successful:** Try Experiment 2 or 3
5. **If unsuccessful:** Analyze why, adjust parameters, retry

**Goal:** Improve at least 2 out of 3 metrics significantly!

---

*Good luck with your experiments! 🚀*
