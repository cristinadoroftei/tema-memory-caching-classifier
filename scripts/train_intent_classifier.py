"""
Train Intent Classifier — TF-IDF + LogisticRegression

Reads data/intent_training_data.csv, trains a sklearn pipeline,
evaluates on a test split, and saves the model to data/intent_classifier.joblib.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# === 1. Load data ===
DATA_DIR = Path(__file__).parent.parent / "data"
df = pd.read_csv(DATA_DIR / "intent_training_data.csv")

print(f"Loaded {len(df)} examples")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

X = df["query"]
y = df["label"]

# === 2. Train/test split (80/20) ===
# stratify=y ensures each label has proportional representation in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} examples, Test: {len(X_test)} examples\n")

# === 3. Build pipeline: TF-IDF → LogisticRegression ===
pipeline = Pipeline([
    # TF-IDF: converts text → numeric vectors
    #   - analyzer="word": split on words
    #   - ngram_range=(1,2): use single words AND pairs of consecutive words
    #     e.g. "valoare totală" becomes a feature, not just "valoare" and "totală" separately
    #   - max_features=5000: keep top 5000 most informative features
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=5000,
    )),
    # LogisticRegression: learns decision boundaries between the 3 classes
    #   - max_iter=1000: enough iterations to converge
    ("clf", LogisticRegression(max_iter=1000)),
])

# === 4. Train ===
pipeline.fit(X_train, y_train)

# === 5. Evaluate on test set ===
y_pred = pipeline.predict(X_test)
print("=== Classification Report (test set) ===")
print(classification_report(y_test, y_pred))

# === 6. Cross-validation on full dataset ===
# 5-fold: trains 5 times, each time holding out a different 20%
# gives a more robust accuracy estimate than a single split
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"=== Cross-validation (5-fold) ===")
print(f"Scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})\n")

# === 7. Retrain on full dataset before saving ===
# We evaluated on splits; now train on ALL data for the production model
pipeline.fit(X, y)

# === 8. Save model ===
model_path = DATA_DIR / "intent_classifier.joblib"
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")

# === 9. Quick sanity check ===
test_queries = [
    "Caută documente despre licitații",
    "Câte contracte au fost semnate în 2023?",
    "Fă un rezumat al achizițiilor directe",
]
print("\n=== Sanity check ===")
for q in test_queries:
    label = pipeline.predict([q])[0]
    proba = pipeline.predict_proba([q])[0]
    confidence = max(proba)
    print(f"  '{q}' → {label} ({confidence:.0%})")
