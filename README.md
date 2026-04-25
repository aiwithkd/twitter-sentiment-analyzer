# Twitter Sentiment Analyzer

End-to-end sentiment analysis pipeline on 3,000 tweets across 5 topics — using TF-IDF feature engineering and Logistic Regression, with full model evaluation and an interactive results dashboard.

**Live Demo → [aiwithkd.github.io/twitter-sentiment-analyzer](https://aiwithkd.github.io/twitter-sentiment-analyzer)**

---

## What This Project Does and Why

Sentiment analysis is the task of automatically determining the emotional tone behind a piece of text — positive, negative, or neutral. It's one of the most common real-world NLP applications used in customer feedback analysis, brand monitoring, and social media analytics.

This project builds a complete ML pipeline from raw text data to a deployed, interactive dashboard — covering every step a data scientist would follow in a real engagement: data preparation, text preprocessing, feature engineering, model training, evaluation, and result interpretation.

---

## Results at a Glance

| Metric | Score | What it means |
|---|---|---|
| Test Accuracy | 95.0% | 570 out of 600 test tweets correctly classified |
| ROC-AUC (weighted OVR) | 0.997 | Near-perfect class separation ability |
| 5-Fold CV F1 (weighted) | 94.97% ± 0.63% | Consistent across different data splits — model is not overfitting |
| TF-IDF Features | 844 | Number of informative word/bigram features extracted |
| Train / Test Split | 2,400 / 600 | Stratified — each split preserves original class ratios |

---

## Repository Structure

```
twitter-sentiment-analyzer/
├── src/
│   ├── generate_data.py     # Simulates 3,000 labeled tweets with realistic noise
│   └── train_model.py       # Full ML pipeline — preprocessing → TF-IDF → LR → evaluation
├── data/
│   └── raw/tweets.csv       # Raw tweet dataset (3,000 records, 7 columns)
├── output/
│   └── results.json         # Pre-computed model results consumed by the dashboard
├── index.html               # Interactive results dashboard (Chart.js, no server needed)
├── requirements.txt         # Pinned dependencies
└── README.md
```

---

## Step-by-Step Pipeline

### Step 1 — Data Generation (`src/generate_data.py`)

**What it does:**
Generates 3,000 simulated tweets with ground-truth sentiment labels across 5 topics: tech, sports, politics, entertainment, finance.

**Why simulate data instead of using a real dataset?**
Real Twitter data requires API access and has licensing restrictions. Simulated data with controlled vocabulary lets us demonstrate the full pipeline clearly. The principles apply identically to real tweet data.

**How noise is injected to make it realistic:**
- 12% of tweets use cross-topic vocabulary — a sports tweet might accidentally use tech-sounding words
- 8% of tweets are replaced with ambiguous phrases like "could go either way" or "mixed feelings about this" — these are genuinely hard to classify
- 15% of tweets have informal noise words prepended (tbh, ngl, lol, smh) — mimicking real social media language
- Sentiment class weights are imbalanced on purpose: 45% positive, 30% negative, 25% neutral — reflecting real-world distribution where positive tweets dominate

This noise design is the reason the model achieves 95% and not 100% — 100% accuracy on NLP data is always a red flag.

**Dataset columns:**
| Column | Description |
|---|---|
| tweet_id | Unique identifier |
| text | Raw tweet text |
| sentiment | Ground truth label: positive / negative / neutral |
| topic | One of: tech, sports, politics, entertainment, finance |
| timestamp | Simulated posting date (2024) |
| likes | Simulated engagement count |
| retweets | Simulated retweet count |

---

### Step 2 — Text Preprocessing (`src/train_model.py`)

**Why preprocessing matters:**
Raw text is noisy. Machine learning models work on numbers, not words. Before converting text to numbers, we need to reduce noise and standardise the vocabulary so the model learns signal, not noise.

**Steps applied:**
1. **Lowercase** — `"LOVE"` and `"love"` should be treated as the same word
2. **URL removal** — `http://t.co/xyz` adds no sentiment signal
3. **Mention removal** — `@user_1234` is noise; we strip it with Regex: `re.sub(r"@\w+", "", text)`
4. **Hashtag normalisation** — `#amazing` becomes `amazing` — we keep the word, drop the symbol
5. **Non-alphabetic character removal** — punctuation, numbers, emojis replaced with spaces
6. **Whitespace normalisation** — multiple spaces collapsed to single space
7. **Stopword filtering** — common words like "the", "is", "and" carry no sentiment signal and inflate the vocabulary. Custom stopword list used (not NLTK) to keep dependencies minimal
8. **Minimum token length** — tokens under 3 characters removed

**Key design choice — custom stopwords over NLTK:**
NLTK's stopword list sometimes removes useful words for sentiment (e.g. "not", "but"). A custom minimal list gives more control.

---

### Step 3 — Feature Engineering: TF-IDF

**What is TF-IDF?**
TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into a numerical matrix where each row is a tweet and each column is a word or phrase. The value in each cell reflects how important that word is to that tweet relative to the whole dataset.

- **TF (Term Frequency):** How often a word appears in a specific tweet. A word that appears 3 times in a 10-word tweet gets TF = 0.3
- **IDF (Inverse Document Frequency):** Penalises words that appear in almost every tweet (they carry no discriminative power). "the" appears everywhere so it gets a low IDF. "crashing" appears only in negative tweets so it gets a high IDF
- **TF-IDF = TF × IDF** — high score means the word is both frequent in this tweet and rare across the dataset

**Parameters chosen and why:**

| Parameter | Value | Reason |
|---|---|---|
| `ngram_range=(1,2)` | Unigrams + bigrams | Captures phrases like "not good" and "best ever" — single words miss negation context |
| `max_features=8000` | Cap at 8,000 | Prevents overfitting from rare words; keeps matrix manageable |
| `min_df=2` | Minimum 2 documents | Words appearing only once are noise, not signal |
| `sublinear_tf=True` | Log-scaled TF | Prevents very frequent words from dominating. TF becomes `1 + log(tf)` |
| `strip_accents='unicode'` | Normalise accents | Treats "café" and "cafe" identically |

**Output:** A sparse matrix of shape (3000, 844) — 3,000 tweets × 844 TF-IDF features

---

### Step 4 — Model Training: Logistic Regression

**Why Logistic Regression for text classification?**
Despite its simple name, Logistic Regression is one of the strongest baselines for text classification. It works very well with high-dimensional sparse features (like TF-IDF), is fast to train, highly interpretable (you can read which words drive each class), and rarely overfits when regularisation is applied. For NLP tasks where features >> samples, it often outperforms more complex models.

**Parameters and reasoning:**

| Parameter | Value | Reason |
|---|---|---|
| `solver='lbfgs'` | L-BFGS optimiser | Efficient for multinomial classification, handles sparse data well |
| `C=1.0` | Regularisation strength | Default L2 regularisation. Lower C = stronger regularisation = simpler model |
| `class_weight='balanced'` | Auto-balance classes | Dataset is imbalanced (45/30/25). This weights the loss function so the model doesn't ignore minority classes |
| `max_iter=1000` | Training iterations | Ensures convergence without hitting the default 100-iteration limit |
| `random_state=42` | Reproducibility | Guarantees identical results on every run |

**What `class_weight='balanced'` does exactly:**
It computes `n_samples / (n_classes * np.bincount(y))` per class and uses that as a multiplier in the loss. So if negative tweets are underrepresented, misclassifying a negative tweet costs more in training, forcing the model to pay more attention to it.

---

### Step 5 — Model Evaluation

**Why accuracy alone is not enough:**
If 90% of tweets were positive and we predicted "positive" for everything, we'd get 90% accuracy — but the model would be useless. That's why we report multiple metrics.

**Metrics explained:**

| Metric | Formula | What it tells you |
|---|---|---|
| Accuracy | Correct / Total | Overall correctness — misleading with imbalanced classes |
| Precision | TP / (TP + FP) | Of all tweets predicted as positive, how many actually were? |
| Recall | TP / (TP + FN) | Of all actual positive tweets, how many did we catch? |
| F1 Score | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall — best single metric for imbalanced data |
| ROC-AUC | Area under ROC curve | Model's ability to rank correct class higher than incorrect. 1.0 = perfect, 0.5 = random |
| CV F1 ± std | 5-fold cross-validation | Tests if performance is consistent or if the model got lucky on one split |

**Confusion Matrix interpretation:**
A confusion matrix shows where the model makes mistakes. Diagonal = correct predictions. Off-diagonal = errors. For example, if neutral tweets are frequently misclassified as positive, that tells us the neutral class needs more distinctive vocabulary.

**Top TF-IDF Coefficients:**
Logistic Regression assigns a weight to each feature per class. The highest weights reveal which words most strongly push the model toward that class. For example, "love" and "amazing" will have the highest positive coefficient — this is model interpretability and is directly used in real production systems to audit and explain model decisions.

---

### Step 6 — Dashboard

The dashboard reads `output/results.json` — pre-computed by `train_model.py` — and renders:
- 5 metric cards (accuracy, AUC, CV F1, features, test size)
- Precision / Recall / F1 grouped bar chart per class
- Confusion matrix with color-coded cells
- Sentiment distribution doughnut chart
- Monthly sentiment trend line chart
- Sentiment by topic stacked bar chart
- Top discriminative TF-IDF features per sentiment class
- Filterable prediction table with per-tweet confidence scores and probability distributions

---

## Interview Talking Points

**"Walk me through your sentiment analysis project"**
Start with: *"I built an end-to-end NLP pipeline — data prep, TF-IDF feature engineering, Logistic Regression with balanced class weights, and full evaluation including ROC-AUC and 5-fold cross-validation. The model achieved 95% accuracy on a held-out test set."*

**"Why TF-IDF over word embeddings like Word2Vec?"**
TF-IDF is interpretable, fast, and works extremely well for short text classification when you have clear vocabulary signals. Word2Vec captures semantic similarity but is harder to interpret, slower, and typically needs more data to shine. For a 3,000 tweet classification problem, TF-IDF is the right tool.

**"Why Logistic Regression over Random Forest or SVM?"**
Logistic Regression is highly effective on sparse high-dimensional data — exactly what TF-IDF produces. It's also fully interpretable via coefficients, making it easy to explain which features drive predictions. SVMs can be slightly more accurate but are slower and less interpretable. Random Forests don't work as well with sparse TF-IDF matrices.

**"How did you handle class imbalance?"**
Used `class_weight='balanced'` which reweights the loss function so minority classes contribute proportionally during training. An alternative would be SMOTE oversampling, but that works better on dense feature spaces, not sparse TF-IDF matrices.

**"What does your confusion matrix tell you?"**
The neutral class has the most misclassifications — it's predicted as positive or negative in some cases. This makes intuitive sense: neutral tweets are often borderline, ambiguous, or topic-dependent, making them harder to separate from mildly positive/negative ones.

**"How would you improve this further?"**
1. Use n-gram ranges up to trigrams for better phrase capture
2. Add sentiment-specific lexicon features (VADER scores) as additional features alongside TF-IDF
3. Fine-tune a pre-trained transformer like DistilBERT — would likely push accuracy above 97%
4. Collect real Twitter data via the Twitter API for production use

---

## Running Locally

```bash
git clone https://github.com/aiwithkd/twitter-sentiment-analyzer
cd twitter-sentiment-analyzer
pip install -r requirements.txt

python src/generate_data.py    # creates data/raw/tweets.csv
python src/train_model.py      # trains model, creates output/results.json

python -m http.server 8000
# open http://localhost:8000
```

## Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.9+ | Pipeline orchestration |
| pandas | 2.2.0 | Data loading and manipulation |
| numpy | 1.26.4 | Numerical operations |
| scikit-learn | 1.4.0 | TF-IDF, Logistic Regression, all evaluation metrics |
| Regex (re) | stdlib | Text cleaning and normalisation |
| Chart.js | 4.4.2 | Dashboard visualisations |
| GitHub Pages | — | Static hosting |

---

*Built by [Kunal Deokar](https://github.com/aiwithkd)*
