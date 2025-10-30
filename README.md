# Multi-Task Amazon Review Analysis

A comprehensive machine learning project for analyzing Amazon product reviews with multiple tasks including sentiment analysis, helpfulness prediction, and aspect extraction.

## Project Overview

This project implements a multi-task learning approach to analyze Amazon product reviews from the 2023 dataset. The system performs several key tasks:

- **Sentiment Analysis**: Classify reviews as Positive, Neutral, or Negative based on star ratings
- **Helpfulness Prediction**: Predict whether a review will be helpful to customers
- **Text Analysis**: Extract insights from review content including readability and aspect extraction

## Dataset

Uses the Amazon Reviews 2023 dataset from HuggingFace (McAuley-Lab/Amazon-Reviews-2023) with focus on:
- Electronics
- Books  
- Home and Kitchen
- Beauty and Personal Care

## Project Structure

```
├── data/                    # Data storage
│   ├── raw/                # Raw dataset files
│   └── processed/          # Cleaned and processed data
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
├── models/                 # Saved model files
├── results/                # Experiment results
├── visualizations/         # Generated plots and charts
│   └── eda/               # Exploratory data analysis plots
├── config.py              # Hyperparameters configuration
├── data_loader.py         # Dataset loading and preparation
├── preprocessing.py       # Text cleaning and preprocessing
└── requirements.txt       # Python dependencies
```

## Features

- **Data Preprocessing**: Advanced text cleaning, tokenization, and feature engineering
- **Multi-Task Learning**: Simultaneous training on multiple related tasks
- **Comprehensive EDA**: Statistical analysis, visualizations, and insights
- **Scalable Pipeline**: Handles large datasets with streaming support
- **Modern ML Stack**: Built with PyTorch, Transformers, and scikit-learn

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and process data:
```python
from data_loader import load_amazon_reviews
load_amazon_reviews()
```

3. Run preprocessing:
```python
from preprocessing import preprocess_reviews
preprocess_reviews()
```

4. Explore data:
```bash
jupyter notebook notebooks/eda.ipynb
```

## Configuration

Key hyperparameters can be adjusted in `config.py`:
- Learning rate: 2e-5
- Batch size: 16
- Max sequence length: 512
- Training epochs: 5

## License

MIT License - see LICENSE file for details.
