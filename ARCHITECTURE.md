# Project Architecture & File Inventory

## 📁 Complete File Structure (As Built)

```
customer-reviews-sentiment-analysis/
│
├── 📄 README.md                      ✅ COMPLETE (3,500 words)
├── 📄 LICENSE                        ✅ COMPLETE (MIT License)
├── 📄 requirements.txt               ✅ COMPLETE (40+ packages)
├── 📄 .gitignore                     ✅ COMPLETE
├── 📄 PROJECT_STATUS.md              ✅ COMPLETE (Progress tracker)
├── 📄 QUICK_START.md                 ✅ COMPLETE (Implementation guide)
├── 📄 COMPLETION_SUMMARY.md          ✅ COMPLETE (This session summary)
│
├── 📄 config.py                      ⚠️ LEGACY (move to src/)
├── 📄 data_loader.py                 ⚠️ LEGACY (move to src/)
├── 📄 preprocessing.py               ⚠️ LEGACY (move to src/)
│
├── 📁 src/                           ✅ CREATED
│   ├── 📄 __init__.py                ✅ COMPLETE (Package init)
│   ├── 📄 config.py                  ✅ COMPLETE (220 lines)
│   ├── 📄 utils.py                   ✅ COMPLETE (500 lines)
│   ├── 📄 data_loader.py             ⏳ TODO (Template in QUICK_START)
│   ├── 📄 preprocessing.py           ⏳ TODO (Template in QUICK_START)
│   ├── 📄 model.py                   ⏳ TODO (Template in QUICK_START)
│   ├── 📄 dataset.py                 ⏳ TODO (Template in QUICK_START)
│   └── 📄 visualization.py           ⏳ TODO (Optional)
│
├── 📁 data/                          ✅ CREATED (Empty, ready for data)
│   ├── 📁 raw/                       └─ For downloaded datasets
│   └── 📁 processed/                 └─ For cleaned datasets
│
├── 📁 notebooks/                     ✅ CREATED (Empty, ready for notebooks)
│   ├── 📄 eda_analysis.ipynb         ⏳ TODO (Template in QUICK_START)
│   ├── 📄 model_experimentation.ipynb ⏳ TODO (Optional)
│   └── 📄 results_visualization.ipynb ⏳ TODO (Optional)
│
├── 📁 scripts/                       ✅ CREATED (Empty, ready for scripts)
│   ├── 📄 download_data.py           ⏳ TODO (Template in QUICK_START)
│   ├── 📄 preprocess_data.py         ⏳ TODO (Template in QUICK_START)
│   ├── 📄 train.py                   ⏳ TODO (Template in QUICK_START)
│   └── 📄 evaluate.py                ⏳ TODO (Template in QUICK_START)
│
├── 📁 models/                        ✅ CREATED (Empty, for saved models)
│   ├── 📁 checkpoints/               └─ For training checkpoints
│   ├── 📄 multitask_model_best.pt    ⏳ (Generated after training)
│   └── 📄 config.json                ⏳ (Generated after training)
│
├── 📁 results/                       ✅ CREATED (Empty, for metrics)
│   ├── 📄 metrics.json               ⏳ (Generated after evaluation)
│   ├── 📄 training_history.csv       ⏳ (Generated during training)
│   └── 📄 training.log               ⏳ (Generated during training)
│
├── 📁 visualizations/                ✅ CREATED (Empty, for plots)
│   ├── 📁 eda/                       └─ For EDA plots
│   └── 📁 modeling/                  └─ For model performance plots
│
├── 📁 tests/                         ✅ CREATED (Empty, ready for tests)
│   ├── 📄 test_data_loader.py        ⏳ TODO (Template in QUICK_START)
│   ├── 📄 test_preprocessing.py      ⏳ TODO (Template in QUICK_START)
│   └── 📄 test_model.py              ⏳ TODO (Template in QUICK_START)
│
└── 📁 docs/                          ✅ CREATED
    ├── 📄 literature_review.md       ✅ COMPLETE (5,200 words, 20+ citations)
    ├── 📄 report.md                  ⏳ TODO (Template in QUICK_START)
    ├── 📄 presentation_slides.md     ⏳ TODO (Template in QUICK_START)
    └── 📄 system_architecture.png    ⏳ TODO (Create diagram)
```

**Legend:**
- ✅ **COMPLETE** - Fully implemented and documented
- ⏳ **TODO** - Needs implementation (templates provided)
- ⚠️ **LEGACY** - Old files, consider moving to src/

---

## 🎯 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  HuggingFace Datasets → Amazon Reviews 2023 →       │   │
│  │  Download & Filter (4 categories) → Save Parquet    │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Text Cleaning → Tokenization (DistilBERT) →        │   │
│  │  Sentiment Labeling → Feature Engineering →         │   │
│  │  Train/Val/Test Split (70/15/15)                    │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐        ┌─────────────────────┐
│     EDA      │        │   MODEL TRAINING    │
│  Analysis &  │        │  ┌──────────────┐   │
│ Visualization│        │  │ DistilBERT   │   │
│              │        │  │   Encoder    │   │
│ • Ratings    │        │  │   (Shared)   │   │
│ • Text Stats │        │  └──────┬───────┘   │
│ • Categories │        │         │           │
│ • Wordclouds │        │  ┌──────┴────────┐  │
│ • 15+ Plots  │        │  │  Task Heads:  │  │
│              │        │  │               │  │
└──────────────┘        │  │ 1. Sentiment  │  │
                        │  │ 2. Helpfulness│  │
                        │  │ 3. Aspects    │  │
                        │  └───────────────┘  │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │    EVALUATION        │
                        │  • Accuracy, F1      │
                        │  • RMSE, MAE         │
                        │  • Confusion Matrix  │
                        │  • Per-category      │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   RESULTS & REPORT   │
                        │  • Metrics JSON      │
                        │  • Visualizations    │
                        │  • Documentation     │
                        │  • Presentation      │
                        └──────────────────────┘
```

---

## 📊 Code Statistics

### Files Created (This Session)

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| Documentation | 6 | ~15,000 words | ✅ Complete |
| Source Code | 3 | ~800 lines | ✅ Complete |
| Configuration | 1 | 220 lines | ✅ Complete |
| Project Setup | 3 | - | ✅ Complete |
| **Total** | **13** | **~1,020 lines + 15K words** | **✅** |

### Files Needed (Next Phase)

| Category | Files | Estimated Lines | Priority |
|----------|-------|-----------------|----------|
| Data Pipeline | 2 | ~600 lines | 🔴 High |
| Preprocessing | 1 | ~400 lines | 🔴 High |
| Model | 2 | ~500 lines | 🔴 High |
| Scripts | 4 | ~800 lines | 🔴 High |
| Notebooks | 1 | ~300 cells | 🔴 High |
| Tests | 3 | ~300 lines | 🟡 Medium |
| Final Docs | 3 | ~5,000 words | 🔴 High |
| **Total** | **16** | **~2,600 lines + 5K words** | - |

---

## 🔄 Data Flow Diagram

```
Raw Amazon Reviews (JSON/Parquet)
        │
        ├─► Filter by Category (Electronics, Books, Home, Beauty)
        │
        ├─► Sample (250K per category = 1M total)
        │
        ├─► Clean Text
        │   ├─► Remove URLs, HTML
        │   ├─► Expand contractions
        │   └─► Normalize (lowercase, etc.)
        │
        ├─► Feature Engineering
        │   ├─► Sentiment Labels (from ratings)
        │   ├─► Helpfulness Scores (votes ratio)
        │   └─► Aspect Keywords (extraction)
        │
        ├─► Tokenization (DistilBERT)
        │   ├─► input_ids
        │   ├─► attention_mask
        │   └─► max_length=256
        │
        ├─► Split Data
        │   ├─► Train (70%)
        │   ├─► Val (15%)
        │   └─► Test (15%)
        │
        └─► Save Processed (Parquet)
                │
                ├─► data/processed/train.parquet
                ├─► data/processed/val.parquet
                └─► data/processed/test.parquet
```

---

## 🧠 Multi-Task Model Architecture

```
Input: Review Text
        │
        ├─► DistilBERT Tokenizer
        │   └─► [input_ids, attention_mask]
        │
        ▼
┌───────────────────────────────────┐
│   DistilBERT Encoder (Shared)     │
│   • 6 Transformer Layers          │
│   • Hidden Dim: 768               │
│   • Dropout: 0.1                  │
└─────────────┬─────────────────────┘
              │ [CLS] Token Output
              │
    ┌─────────┴─────────┬─────────────┐
    │                   │             │
    ▼                   ▼             ▼
┌───────────┐   ┌──────────────┐  ┌─────────────┐
│ Sentiment │   │ Helpfulness  │  │   Aspects   │
│   Head    │   │     Head     │  │    Head     │
│           │   │              │  │             │
│ Linear    │   │  Linear      │  │  Linear     │
│ 768→256   │   │  768→128     │  │  768→256    │
│ ReLU      │   │  ReLU        │  │  ReLU       │
│ 256→3     │   │  128→1       │  │  256→10     │
│           │   │              │  │             │
│ Output:   │   │  Output:     │  │  Output:    │
│ 3 classes │   │  Score [0-1] │  │  10 labels  │
└───────────┘   └──────────────┘  └─────────────┘
  (Softmax)      (Sigmoid/Linear)   (Sigmoid)
```

**Loss Function**:
```
Total Loss = λ₁·CE_sentiment + λ₂·MSE_helpfulness + λ₃·BCE_aspects

Where:
  λ₁ = 1.0  (Sentiment weight)
  λ₂ = 0.5  (Helpfulness weight)
  λ₃ = 0.3  (Aspects weight)
```

---

## 📈 Training Pipeline Flow

```
1. Initialize
   ├─► Load config (src/config.py)
   ├─► Set random seed (reproducibility)
   ├─► Setup logger
   └─► Check device (CPU/GPU)

2. Load Data
   ├─► Load train/val datasets
   ├─► Create DataLoaders (batch_size=32)
   └─► Verify data integrity

3. Create Model
   ├─► Initialize MultiTaskReviewModel
   ├─► Load pre-trained DistilBERT
   ├─► Move model to device
   └─► Print model summary

4. Setup Optimization
   ├─► Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
   ├─► Scheduler: Linear with warmup
   └─► Loss functions for each task

5. Training Loop (10 epochs)
   ├─► For each epoch:
   │   ├─► Train phase
   │   │   ├─► Forward pass
   │   │   ├─► Calculate multi-task loss
   │   │   ├─► Backward pass
   │   │   └─► Update weights
   │   │
   │   ├─► Validation phase
   │   │   ├─► No gradient update
   │   │   ├─► Calculate metrics
   │   │   └─► Track performance
   │   │
   │   ├─► Log metrics
   │   ├─► Save checkpoint if best
   │   └─► Early stopping check
   │
   └─► End training

6. Save & Report
   ├─► Save best model
   ├─► Save training history
   ├─► Generate plots
   └─► Print summary
```

---

## 📚 Documentation Hierarchy

```
Level 1: Project Overview
└─► README.md
    ├─► Quick project description
    ├─► Installation instructions
    ├─► Usage guide
    └─► Syllabus mapping

Level 2: Academic Documentation
├─► docs/literature_review.md
│   ├─► Theoretical foundations
│   ├─► 20+ academic sources
│   └─► Research gap analysis
│
├─► docs/report.md (TODO)
│   ├─► Complete methodology
│   ├─► Results & discussion
│   └─► Business insights
│
└─► docs/presentation_slides.md (TODO)
    └─► Concise project summary

Level 3: Implementation Guides
├─► QUICK_START.md
│   ├─► Step-by-step instructions
│   ├─► Code templates
│   └─► Testing procedures
│
└─► PROJECT_STATUS.md
    └─► Progress tracking

Level 4: Code Documentation
├─► src/config.py
│   └─► Configuration options
│
├─► src/utils.py
│   └─► Utility functions
│
└─► Source files (with docstrings)
    └─► Function-level documentation

Level 5: Interactive Documentation
└─► Jupyter Notebooks
    ├─► EDA analysis
    ├─► Model experiments
    └─► Results visualization
```

---

## 🎓 Course Alignment Matrix

| Module | Topic | File/Section | Status |
|--------|-------|--------------|--------|
| **Module 1** | Big Data Intro | README.md § Syllabus | ✅ |
| | 3Vs/5Vs | literature_review.md § 2 | ✅ |
| **Module 2** | Data Preprocessing | src/preprocessing.py | ⏳ |
| | Data Cleaning | src/data_loader.py | ⏳ |
| **Module 3** | MapReduce | src/config.py (CHUNK_SIZE) | ✅ |
| | Distributed Concepts | literature_review.md § 6 | ✅ |
| **Module 4** | Data Storage | data/ (Parquet format) | ✅ |
| | NoSQL Concepts | README.md § Syllabus | ✅ |
| **Module 5** | Statistics | notebooks/eda_analysis.ipynb | ⏳ |
| | Visualization | src/visualization.py | ⏳ |
| **Module 6** | ML Algorithms | src/model.py | ⏳ |
| | Model Evaluation | scripts/evaluate.py | ⏳ |
| **Module 7** | Text Analytics | src/preprocessing.py | ⏳ |
| | Applications | docs/report.md | ⏳ |

---

## ✅ Quality Checklist Progress

### Documentation ✅ (100%)
- [x] README with syllabus mapping
- [x] Literature review (20+ sources)
- [x] Installation instructions
- [x] Usage guide
- [x] Course outcomes mapping
- [x] Quick start guide

### Code Structure ✅ (100%)
- [x] Organized directory structure
- [x] Configuration management
- [x] Utility functions
- [x] Package initialization
- [x] Dependencies specified

### Implementation ⏳ (0%)
- [ ] Data loading pipeline
- [ ] Preprocessing functions
- [ ] EDA notebook
- [ ] Model architecture
- [ ] Training scripts
- [ ] Evaluation scripts

### Testing ⏳ (0%)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Test coverage

### Final Deliverables ⏳ (15%)
- [x] Project report structure
- [x] Literature review
- [ ] Results section
- [ ] Presentation slides
- [ ] Architecture diagram

---

## 🚀 Next Session Quick Start

### What to Do First
```powershell
# 1. Navigate to project
cd "c:\Users\Apoor\customer_review_sentiment analysis\customer-reviews-sentiment-analysis"

# 2. Activate environment (if created)
.\venv\Scripts\Activate.ps1

# 3. Install dependencies (if not done)
pip install -r requirements.txt

# 4. Test setup
python src/config.py
python src/utils.py

# 5. Start with data pipeline
# Open QUICK_START.md and follow Phase 2A
```

### Key Files to Reference
1. **QUICK_START.md** - Step-by-step implementation guide
2. **PROJECT_STATUS.md** - What's done and what's needed
3. **src/config.py** - All configuration options
4. **src/utils.py** - Available utility functions

---

**Remember**: The foundation is solid. Focus on implementation using the templates provided!

