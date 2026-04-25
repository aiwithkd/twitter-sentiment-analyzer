"""
Twitter Sentiment Analysis — ML Pipeline

Steps:
  1. Load and inspect raw tweet data
  2. Text preprocessing — cleaning, normalization, stopword removal
  3. Feature engineering — TF-IDF vectorization with unigrams + bigrams
  4. Model training — Logistic Regression with class weight balancing
  5. Evaluation — accuracy, precision, recall, F1, confusion matrix, ROC-AUC
  6. Error analysis — misclassified samples by topic and sentiment
  7. Export results to output/results.json for dashboard consumption
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/tweets.csv")
print(f"Loaded {len(df)} tweets")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}\n")

# ── 2. Text preprocessing ──────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or",
    "but","not","with","this","that","was","are","be","as","by","from",
    "have","has","had","will","would","could","should","been","via","cc"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # strip hashtag symbol, keep word
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()              # normalise whitespace
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
print(f"After cleaning: {len(df)} tweets remain\n")

# ── 3. TF-IDF Feature Engineering ─────────────────────────────────────────────
# Unigrams + bigrams, min_df=2 to remove noise, sublinear_tf for better scaling
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode"
)

X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

print(f"TF-IDF matrix shape: {X.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}\n")

# ── 4. Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# ── 5. Model Training — Logistic Regression ────────────────────────────────────
# class_weight='balanced' handles class imbalance without oversampling
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# ── 6. Cross-validation ────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
print(f"5-Fold CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ── 7. Evaluation ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, output_dict=True)
cm       = confusion_matrix(y_test, y_pred, labels=["positive","negative","neutral"])

le = LabelEncoder()
le.fit(["positive","negative","neutral"])
y_test_enc = le.transform(y_test)
y_prob_ord  = y_prob[:, [list(model.classes_).index(c) for c in le.classes_]]
roc_auc = roc_auc_score(y_test_enc, y_prob_ord, multi_class="ovr", average="weighted")

print(f"Test Accuracy : {accuracy:.4f}")
print(f"ROC-AUC (OVR) : {roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{cm}\n")

# ── 8. Top TF-IDF features per class ──────────────────────────────────────────
feature_names = tfidf.get_feature_names_out()
top_features  = {}
for i, cls in enumerate(model.classes_):
    coef = model.coef_[i]
    top_idx = np.argsort(coef)[::-1][:15]
    top_features[cls] = [
        {"word": feature_names[j], "weight": round(float(coef[j]), 4)}
        for j in top_idx
    ]

# ── 9. Sentiment trend by topic ───────────────────────────────────────────────
df_test = df.iloc[y_test.index].copy()
df_test["predicted"] = y_pred

topic_sentiment = (
    df_test.groupby(["topic","sentiment"])
    .size()
    .reset_index(name="count")
    .to_dict(orient="records")
)

# ── 10. Monthly sentiment trend ───────────────────────────────────────────────
df["month"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m")
monthly = (
    df.groupby(["month","sentiment"])
    .size()
    .reset_index(name="count")
    .to_dict(orient="records")
)

# ── 11. Sample predictions for dashboard table ────────────────────────────────
sample_idx   = np.random.choice(len(y_test), size=min(300, len(y_test)), replace=False)
y_test_arr   = y_test.reset_index(drop=True)
y_pred_arr   = pd.Series(y_pred)
df_test_reset = df.iloc[y_test.index].reset_index(drop=True)

samples = []
for i in sample_idx:
    prob = y_prob[i]
    pred = y_pred_arr[i]
    actual = y_test_arr[i]
    conf = float(max(prob))
    samples.append({
        "text":      df_test_reset.iloc[i]["text"],
        "topic":     df_test_reset.iloc[i]["topic"],
        "actual":    actual,
        "predicted": pred,
        "correct":   actual == pred,
        "confidence": round(conf, 3),
        "prob_positive": round(float(prob[list(model.classes_).index("positive")]), 3),
        "prob_negative": round(float(prob[list(model.classes_).index("negative")]), 3),
        "prob_neutral":  round(float(prob[list(model.classes_).index("neutral")]),  3),
    })

# ── 12. Misclassification analysis ────────────────────────────────────────────
misclassified = [s for s in samples if not s["correct"]]
mis_by_topic  = {}
for s in misclassified:
    mis_by_topic[s["topic"]] = mis_by_topic.get(s["topic"], 0) + 1

# ── 13. Assemble results JSON ──────────────────────────────────────────────────
results = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "total_tweets":     len(df),
        "train_size":       int(X_train.shape[0]),
        "test_size":        int(X_test.shape[0]),
        "vocab_size":       int(len(tfidf.vocabulary_)),
        "tfidf_features":   int(X.shape[1]),
        "sentiment_dist":   df["sentiment"].value_counts().to_dict(),
        "topic_dist":       df["topic"].value_counts().to_dict(),
    },
    "model": {
        "name":             "Logistic Regression",
        "features":         "TF-IDF (unigram + bigram, max 8000 features)",
        "preprocessing":    "Lowercasing, URL/mention removal, stopword filtering",
        "class_weight":     "balanced",
        "regularization":   "C=1.0",
    },
    "metrics": {
        "accuracy":         round(accuracy, 4),
        "roc_auc_weighted": round(roc_auc, 4),
        "cv_f1_mean":       round(float(cv_scores.mean()), 4),
        "cv_f1_std":        round(float(cv_scores.std()), 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in ["positive","negative","neutral"]
        }
    },
    "confusion_matrix": {
        "labels": ["positive","negative","neutral"],
        "matrix": cm.tolist()
    },
    "top_features":      top_features,
    "topic_sentiment":   topic_sentiment,
    "monthly_trend":     monthly,
    "sample_predictions": samples,
    "misclassification_by_topic": mis_by_topic,
}

# Validate no NaN before writing
with open("output/results.json", "w") as f:
    json.dump(results, f, indent=2, allow_nan=False,
              default=lambda x: None if (isinstance(x, float) and x != x) else str(x))

print(f"Results written → output/results.json")

# Quick sanity check
with open("output/results.json") as f:
    json.load(f)
print("JSON validation: OK")
