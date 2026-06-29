"""
Intent Classifier — loads a trained TF-IDF + LogisticRegression pipeline
and predicts intent (search / extract / summarize) for a given query.
"""
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "data" / "intent_classifier.joblib"


class IntentClassifier:
    """Lightweight intent classifier using a saved sklearn pipeline."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.pipeline = joblib.load(model_path)
        # Extract the label classes (e.g. ['extract', 'search', 'summarize'])
        self.labels = self.pipeline.classes_
        logger.info(f"IntentClassifier loaded from {model_path} — labels: {list(self.labels)}")

    def predict(self, query: str) -> dict:
        """
        Predict intent for a query.

        Returns:
            {
                "label": "search" | "extract" | "summarize",
                "confidence": 0.0-1.0,
                "all_scores": {"search": 0.3, "extract": 0.6, "summarize": 0.1}
            }
        """
        label = self.pipeline.predict([query])[0]
        probas = self.pipeline.predict_proba([query])[0]
        confidence = float(max(probas))
        all_scores = {lbl: float(p) for lbl, p in zip(self.labels, probas)}

        logger.info(f"[INTENT] '{query}' → {label} ({confidence:.0%})")
        return {
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores,
        }
