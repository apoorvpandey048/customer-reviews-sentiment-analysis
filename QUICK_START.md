# Quick Start Guide - Amazon Reviews Sentiment Analysis Project

**Course**: CSE3712 Big Data Analytics  
**Project Status**: Foundation Complete (35%) - Ready for Implementation Phase

---

## 🎯 What's Been Done

✅ **Complete project structure** with all directories  
✅ **Comprehensive README.md** with syllabus mapping  
✅ **Academic literature review** (5,200 words, 20+ sources)  
✅ **Configuration system** (`src/config.py`)  
✅ **Utility functions** (`src/utils.py`)  
✅ **Updated requirements.txt** with all dependencies  

---

## 🚀 Getting Started Right Now

### Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd "c:\Users\Apoor\customer_review_sentiment analysis\customer-reviews-sentiment-analysis"

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 2: Verify Setup

```python
# Test configuration
python src/config.py

# Test utilities
python src/utils.py
```

---

## 📋 What to Build Next (Priority Order)

### Phase 2A: Complete Data Loading (1-2 hours)

The existing `data_loader.py` needs to be moved to `src/` and enhanced.

**File**: `src/data_loader.py`

**Key Functions to Implement**:
```python
1. download_amazon_reviews() - Use HuggingFace datasets
2. filter_by_category() - Filter for 4 categories
3. sample_reviews() - Get 250K per category
4. clean_and_validate() - Remove duplicates, handle missing values
5. save_to_parquet() - Efficient storage
6. load_processed_data() - Load previously saved data
```

**Command to Test**:
```powershell
python src/data_loader.py --categories all --samples 1000  # Test with 1K first
```

### Phase 2B: Text Preprocessing (2-3 hours)

**File**: `src/preprocessing.py`

**Key Functions to Implement**:
```python
1. clean_text() - Remove URLs, HTML, normalize
2. expand_contractions() - "don't" → "do not"
3. tokenize_reviews() - DistilBERT tokenizer
4. create_sentiment_labels() - Map ratings to positive/neutral/negative
5. calculate_helpfulness() - helpful_votes / total_votes
6. extract_aspects() - Identify product aspects mentioned
7. preprocess_pipeline() - Complete pipeline
```

**Example Usage**:
```python
from src.preprocessing import preprocess_pipeline

processed_df = preprocess_pipeline(
    raw_df,
    max_length=256,
    sentiment_mapping={1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}
)
```

### Phase 2C: Automation Scripts (1 hour)

**File**: `scripts/download_data.py`

```python
import argparse
from src.data_loader import download_and_process_data
from src.config import CATEGORIES, SAMPLE_SIZE_PER_CATEGORY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', default=CATEGORIES)
    parser.add_argument('--samples', type=int, default=SAMPLE_SIZE_PER_CATEGORY)
    args = parser.parse_args()
    
    download_and_process_data(args.categories, args.samples)

if __name__ == "__main__":
    main()
```

**File**: `scripts/preprocess_data.py`

```python
import argparse
from src.preprocessing import preprocess_pipeline
from src.utils import train_val_test_split

def main():
    # Load raw data, preprocess, split, save
    pass
```

---

## 📊 Phase 3: EDA Notebook (3-4 hours)

### Create: `notebooks/eda_analysis.ipynb`

**Structure**:

```markdown
# Amazon Reviews EDA - CSE3712 Big Data Analytics

## 1. Setup & Data Loading
- Import libraries
- Load processed data
- Display basic info

## 2. Univariate Analysis
- Rating distribution (histogram)
- Review length distribution
- Helpful votes distribution
- Category distribution (bar chart)

## 3. Bivariate Analysis
- Rating vs Category (grouped bar)
- Sentiment vs Category (stacked bar)
- Helpfulness vs Rating (scatter)
- Review length vs Helpfulness (scatter)

## 4. Text Analysis
- Word clouds (overall and per category)
- Top 20 words per category
- Readability scores (Flesch-Kincaid)
- Average sentence length

## 5. Statistical Tests
- Chi-square test: Category vs Sentiment
- T-test: Helpful vs Not Helpful review length
- ANOVA: Rating differences across categories

## 6. Correlation Analysis
- Correlation heatmap
- Feature importance analysis

## 7. Key Insights
- Summary of findings
- Business implications
- Data quality notes

## 8. Visualizations Export
- Save all figures to visualizations/eda/
```

**Code Template to Start**:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
sys.path.append('..')
from src.config import *
from src.utils import *

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Load data
df = pd.read_parquet(PROCESSED_DATA_DIR / 'train.parquet')

# Display info
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
df.head()
```

---

## 🧠 Phase 4: Model Implementation (4-5 hours)

### Create: `src/model.py`

**Architecture**:

```python
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class MultiTaskReviewModel(nn.Module):
    """
    Multi-task learning model for Amazon reviews.
    
    Tasks:
    1. Sentiment Classification (3 classes)
    2. Helpfulness Regression (continuous)
    3. Aspect Extraction (multi-label, 10 aspects)
    """
    
    def __init__(self, model_name='distilbert-base-uncased', 
                 num_sentiment_classes=3, num_aspects=10):
        super().__init__()
        
        # Shared encoder (DistilBERT)
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size  # 768
        
        # Task-specific heads
        # 1. Sentiment classification
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_sentiment_classes)
        )
        
        # 2. Helpfulness regression
        self.helpfulness_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # 3. Aspect extraction (multi-label)
        self.aspect_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_aspects)
        )
    
    def forward(self, input_ids, attention_mask):
        # Shared encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Task predictions
        sentiment_logits = self.sentiment_head(pooled_output)
        helpfulness_score = self.helpfulness_head(pooled_output)
        aspect_logits = self.aspect_head(pooled_output)
        
        return {
            'sentiment': sentiment_logits,
            'helpfulness': helpfulness_score,
            'aspects': aspect_logits
        }
```

### Create: `src/dataset.py`

```python
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class ReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            row['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'sentiment_label': row['sentiment_label'],
            'helpfulness_score': row['helpfulness_score'],
            'aspect_labels': row['aspect_labels']  # Multi-hot encoded
        }
```

---

## 🏋️ Phase 5: Training Pipeline (3-4 hours)

### Create: `scripts/train.py`

**Key Components**:

1. **Argument Parsing**
```python
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5)
args = parser.parse_args()
```

2. **Model Setup**
```python
model = MultiTaskReviewModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, ...)
```

3. **Training Loop**
```python
for epoch in range(args.epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    if val_loss < best_val_loss:
        save_model(model, 'best_model.pt')
        best_val_loss = val_loss
```

4. **Multi-Task Loss**
```python
def calculate_multi_task_loss(outputs, labels, weights):
    # Sentiment: Cross-entropy
    sentiment_loss = F.cross_entropy(outputs['sentiment'], labels['sentiment'])
    
    # Helpfulness: MSE
    helpfulness_loss = F.mse_loss(outputs['helpfulness'], labels['helpfulness'])
    
    # Aspects: Binary cross-entropy (multi-label)
    aspect_loss = F.binary_cross_entropy_with_logits(outputs['aspects'], labels['aspects'])
    
    # Weighted combination
    total_loss = (
        weights['sentiment'] * sentiment_loss +
        weights['helpfulness'] * helpfulness_loss +
        weights['aspects'] * aspect_loss
    )
    
    return total_loss, {
        'sentiment': sentiment_loss.item(),
        'helpfulness': helpfulness_loss.item(),
        'aspects': aspect_loss.item()
    }
```

---

## 📈 Phase 6: Evaluation (2-3 hours)

### Create: `scripts/evaluate.py`

```python
# Load model
model = load_model('models/best_model.pt')

# Inference
predictions = []
for batch in test_loader:
    with torch.no_grad():
        outputs = model(batch['input_ids'], batch['attention_mask'])
        predictions.append(outputs)

# Calculate metrics
sentiment_metrics = calculate_metrics(y_true, y_pred, task='classification')
helpfulness_metrics = calculate_metrics(y_true, y_pred, task='regression')

# Save results
save_json(metrics, 'results/metrics.json')

# Visualizations
plot_confusion_matrix(cm, class_names, save_path='visualizations/modeling/cm.png')
```

---

## ✅ Testing (1-2 hours)

### Create: `tests/test_preprocessing.py`

```python
import pytest
from src.preprocessing import clean_text, expand_contractions

def test_clean_text():
    text = "This is GREAT! Visit http://example.com"
    cleaned = clean_text(text, remove_urls=True, lowercase=True)
    assert "http" not in cleaned
    assert cleaned.islower()

def test_expand_contractions():
    text = "I don't think it's good"
    expanded = expand_contractions(text)
    assert "do not" in expanded
    assert "it is" in expanded
```

---

## 📝 Final Documentation (2-3 hours)

### Create: `docs/report.md`

**Template Structure**:

```markdown
# Amazon Reviews Multi-Task Learning: Project Report
## CSE3712 Big Data Analytics

### Abstract
[150-200 words summarizing project]

### 1. Introduction
- Background
- Problem statement
- Objectives
- Course outcomes mapping

### 2. Literature Review
[Summarize docs/literature_review.md]

### 3. System Design
- Architecture diagram
- Component overview
- Technology stack

### 4. Data Collection & Preprocessing
- Dataset description
- Data acquisition method
- Preprocessing pipeline
- Data quality measures

### 5. Exploratory Data Analysis
- Key findings from EDA
- Statistical insights
- Visualizations

### 6. Methodology
- Model architecture
- Training procedure
- Hyperparameters
- Multi-task learning approach

### 7. Results
- Performance metrics table
- Comparison: Single-task vs Multi-task
- Category-specific analysis

### 8. Discussion
- Interpretation of results
- Limitations
- Business implications

### 9. Conclusion
- Summary
- Future work

### 10. References
```

---

## 🎯 Success Checklist

Before submission, ensure:

- [ ] All code runs without errors
- [ ] README.md is accurate and complete
- [ ] Literature review has proper citations
- [ ] EDA notebook has 15+ visualizations
- [ ] Model achieves reasonable performance (>80% sentiment accuracy)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Project report is comprehensive
- [ ] Presentation slides are ready
- [ ] All files are properly documented
- [ ] Syllabus coverage is clearly mapped
- [ ] CO/PO mapping is explicit

---

## 💡 Pro Tips

1. **Start Small**: Test with 1,000 reviews before full dataset
2. **Save Checkpoints**: Don't lose progress from crashes
3. **Document as You Go**: Don't leave documentation for the end
4. **Version Control**: Commit frequently with clear messages
5. **Ask for Help**: Reference error messages when stuck
6. **Test Incrementally**: Don't wait until the end to test
7. **Time Management**: Allocate 2-3 focused sessions
8. **Backup Everything**: Use Git and cloud storage

---

## 📞 Next Session Goals

**Immediate Priorities**:
1. Complete `src/data_loader.py` and `src/preprocessing.py`
2. Download sample data (1K-10K reviews) for testing
3. Create EDA notebook with at least 10 visualizations
4. Begin model implementation

**Expected Outcome**: 
- Functional data pipeline
- Comprehensive EDA
- Model architecture ready for training

---

## 🔗 Useful Links

- [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [DistilBERT Documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [PyTorch Multi-Task Learning](https://pytorch.org/tutorials/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Remember**: The foundation is solid. Focus on implementation, testing, and documentation. You've got this! 🚀
