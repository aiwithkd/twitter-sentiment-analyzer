# Twitter Sentiment Analyzer

End-to-end sentiment analysis pipeline on 3,000 tweets across 5 topics — using TF-IDF feature engineering and Logistic Regression, with full model evaluation and an interactive results dashboard.

**Live Demo → [aiwithkd.github.io/twitter-sentiment-analyzer](https://aiwithkd.github.io/twitter-sentiment-analyzer)**

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 95.0% |
| ROC-AUC (weighted OVR) | 0.997 |
| 5-Fold CV F1 (weighted) | 94.97% ± 0.63% |
| TF-IDF Features | 844 (unigram + bigram) |
| Train / Test Split | 2,400 / 600 (stratified) |

## Repository Structure

```
twitter-sentiment-analyzer/
├── src/
│   ├── generate_data.py     # Simulates 3,000 labeled tweets across 5 topics
│   └── train_model.py       # Full ML pipeline — preprocessing, TF-IDF, LR, evaluation
├── data/
│   └── raw/tweets.csv       # Raw tweet dataset (3,000 records)
├── output/
│   └── results.json         # Pre-computed model results consumed by dashboard
├── index.html               # Interactive results dashboard (Chart.js)
├── requirements.txt
└── README.md
```

## Pipeline Overview

### 1. Data Generation (`src/generate_data.py`)
Generates 3,000 simulated tweets across 5 topics (tech, sports, politics, entertainment, finance) with injected noise — cross-topic vocabulary, ambiguous phrases, and informal language — to produce realistic classification difficulty.

### 2. Text Preprocessing (`src/train_model.py`)
- Lowercase normalisation
- URL and @mention removal via Regex
- Hashtag symbol stripping (word retained)
- Custom stopword filtering
- Minimum token length filtering

### 3. Feature Engineering — TF-IDF
- Unigram + bigram tokenisation (`ngram_range=(1,2)`)
- `sublinear_tf=True` for log-scaled term frequency
- `min_df=2` to remove single-occurrence noise terms
- `max_features=8000` cap

### 4. Model — Logistic Regression
- `class_weight='balanced'` to handle label imbalance
- `solver='lbfgs'` for multinomial classification
- `C=1.0` L2 regularisation
- 5-fold stratified cross-validation

### 5. Evaluation
- Accuracy, Precision, Recall, F1 per class
- Confusion matrix
- ROC-AUC (one-vs-rest, weighted)
- Top TF-IDF feature coefficients per sentiment class
- Per-prediction confidence scores and probability distributions

## Running Locally

```bash
git clone https://github.com/aiwithkd/twitter-sentiment-analyzer
cd twitter-sentiment-analyzer
pip install -r requirements.txt

python src/generate_data.py    # creates data/raw/tweets.csv
python src/train_model.py      # trains model, creates output/results.json

# open index.html via local server
python -m http.server 8000
# visit http://localhost:8000
```

## Tech Stack

| Tool | Role |
|---|---|
| Python | Pipeline orchestration |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| scikit-learn | TF-IDF, Logistic Regression, evaluation metrics |
| Regex | Text cleaning and normalisation |
| Chart.js | Dashboard visualisations |
| GitHub Pages | Static hosting |

---

*Built by [Kunal Deokar](https://github.com/aiwithkd)*
