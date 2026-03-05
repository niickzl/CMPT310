"""Baseline sentiment model for YouTube comment analysis.

Uses a Logistic Regression classifier with TF-IDF vectorization as the
Milestone 1 baseline. The model is trained on a small set of labeled
seed phrases and can be swapped out for DistilBERT in Milestone 2.

Pipeline:
    cleaned text → TF-IDF vectorizer → Logistic Regression → label + confidence
"""

import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Seed training data ─────────────────────────────────────────────────────────
# A small but balanced set of labeled phrases to bootstrap the model.
# Replace or augment this with a real labeled dataset (e.g. SST-2, IMDB)
# for better accuracy.

_SEED_TEXTS = [
    # Positive
    "this video is amazing and very helpful",
    "great content keep it up",
    "loved this so much thank you",
    "best video on this topic hands down",
    "really well explained and clear",
    "this is exactly what I was looking for",
    "incredible work you are very talented",
    "watched the whole thing could not stop",
    "this changed my perspective thank you",
    "super informative and entertaining",
    "one of the best channels on youtube",
    "you deserve way more subscribers",
    "this video helped me so much",
    "absolutely fantastic production quality",
    "I learn something new every time",
    "please make more videos like this",
    "this is gold pure gold",
    "finally someone explained it properly",
    "outstanding video highly recommend",
    "your editing is top notch",
    # Negative
    "this video is a waste of time",
    "terrible explanation totally confused now",
    "worst video I have ever seen",
    "disliked and unsubscribed immediately",
    "this is completely wrong and misleading",
    "so boring I fell asleep",
    "the audio quality is awful",
    "clickbait title does not match content",
    "you clearly do not know what you are talking about",
    "this made no sense at all",
    "very disappointing expected much better",
    "stop making videos you are bad at this",
    "full of mistakes and misinformation",
    "unwatchable garbage content",
    "nobody asked for this video",
    "this channel is going downhill fast",
    "total waste of my time",
    "the worst explanation I have heard",
    "annoying voice and bad editing",
    "nothing useful here move on",
    # Neutral / Mixed (labeled negative to keep binary simple)
    "it was okay nothing special",
    "not sure how I feel about this",
    "some parts were good others not so much",
]

_SEED_LABELS = (
    [1] * 20  # positive
    + [0] * 20  # negative
    + [0] * 3   # neutral → negative bucket for binary model
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")


# ── Model training ─────────────────────────────────────────────────────────────

def _build_pipeline() -> Pipeline:
    """Build and return a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=10_000,
            sublinear_tf=True,    # log normalization
            strip_accents="unicode",
            analyzer="word",
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])


def train(texts: list[str], labels: list[int]) -> Pipeline:
    """Train and return a sentiment pipeline.

    Args:
        texts:  List of cleaned comment strings.
        labels: List of int labels (1 = positive, 0 = negative).

    Returns:
        A fitted sklearn Pipeline.
    """
    pipeline = _build_pipeline()
    pipeline.fit(texts, labels)
    logger.info("Model trained on %d samples.", len(texts))
    return pipeline


def save(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    """Persist a trained pipeline to disk."""
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to %s", path)


def load(path: str = MODEL_PATH) -> Pipeline:
    """Load a persisted pipeline from disk."""
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return pipeline


def get_or_train() -> Pipeline:
    """Return a ready-to-use pipeline, training from seed data if needed.

    Loads from disk if a saved model exists; otherwise trains on seed
    data and saves it for subsequent calls.
    """
    if os.path.exists(MODEL_PATH):
        return load()

    logger.info("No saved model found — training on seed data.")
    pipeline = train(_SEED_TEXTS, _SEED_LABELS)
    save(pipeline)
    return pipeline


# ── Inference ──────────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str          # "positive" | "negative"
    score: float        # confidence 0.0 – 1.0
    is_positive: bool


def predict(texts: list[str], pipeline: Pipeline | None = None) -> list[SentimentResult]:
    """Run sentiment inference on a list of cleaned comment texts.

    Args:
        texts:    List of preprocessed comment strings.
        pipeline: Optional pre-loaded pipeline. Loads automatically if None.

    Returns:
        A list of SentimentResult, one per input text, in the same order.
    """
    if not texts:
        return []

    if pipeline is None:
        pipeline = get_or_train()

    proba = pipeline.predict_proba(texts)  # shape: (n, 2)
    results = []
    for row in proba:
        neg_conf, pos_conf = row[0], row[1]
        is_positive = pos_conf >= 0.5
        results.append(SentimentResult(
            label="positive" if is_positive else "negative",
            score=float(pos_conf if is_positive else neg_conf),
            is_positive=is_positive,
        ))

    return results


def summarize(results: list[SentimentResult]) -> dict:
    """Aggregate a list of SentimentResults into dashboard-ready stats.

    Returns:
        {
            "positive_count": int,
            "negative_count": int,
            "positive_pct": float,   # 0–100
            "negative_pct": float,   # 0–100
            "avg_confidence": float, # 0–1
        }
    """
    if not results:
        return {
            "positive_count": 0,
            "negative_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "avg_confidence": 0.0,
        }

    total = len(results)
    pos = sum(1 for r in results if r.is_positive)
    neg = total - pos
    avg_conf = float(np.mean([r.score for r in results]))

    return {
        "positive_count": pos,
        "negative_count": neg,
        "positive_pct": round(pos / total * 100, 1),
        "negative_pct": round(neg / total * 100, 1),
        "avg_confidence": round(avg_conf, 3),
    }