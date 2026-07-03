"""
model.py
--------
The "ML model for candidate ranking" deliverable.

Rather than hand-tuning fixed weights for each feature, a
GradientBoostingRegressor is trained to map the 4-dimensional feature
vector (tfidf_similarity, skill_overlap_ratio, experience_match_ratio,
education_match_ratio) to a 0-100 fit score. This lets the model learn
non-linear interactions (e.g. strong skill overlap can compensate for
slightly under-target experience) instead of a purely additive formula.

Because no real historical hiring-outcome dataset is available for this
assignment, `generate_synthetic_training_data()` builds a labeled
dataset from a known ground-truth scoring function plus noise, and the
model is trained on that. This is called out explicitly as a limitation
in the README - in production this model would be retrained on real
labeled outcomes (e.g. "was this candidate shortlisted / hired").
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.features import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ranker_model.pkl")


def _ground_truth_score(features: np.ndarray, rng: np.random.Generator) -> float:
    """
    Known scoring function used only to generate synthetic labels.
    Weights loosely mirror real-world hiring priorities: skill match and
    semantic similarity matter most, experience and education secondary,
    with a non-linear bonus when both skills and similarity are high
    (captures "well-rounded strong fit" candidates).
    """
    tfidf_sim, skill_overlap, exp_ratio, edu_ratio = features
    base = (
        0.35 * skill_overlap
        + 0.30 * tfidf_sim
        + 0.20 * exp_ratio
        + 0.15 * edu_ratio
    )
    synergy_bonus = 0.1 * (skill_overlap * tfidf_sim)  # non-linear interaction
    noise = rng.normal(0, 0.03)
    score = base + synergy_bonus + noise
    return float(np.clip(score, 0, 1) * 100)


def generate_synthetic_training_data(n_samples: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_samples, len(FEATURE_NAMES)))
    y = np.array([_ground_truth_score(row, rng) for row in X])
    return X, y


def train_model(save: bool = True) -> GradientBoostingRegressor:
    X, y = generate_synthetic_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"[train_model] Validation MAE: {mae:.2f} (score scale 0-100)")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"[train_model] Saved model to {MODEL_PATH}")

    return model


def load_model() -> GradientBoostingRegressor:
    if not os.path.exists(MODEL_PATH):
        print("[load_model] No saved model found, training a new one...")
        return train_model(save=True)
    return joblib.load(MODEL_PATH)


def predict_score(model: GradientBoostingRegressor, feature_vector: np.ndarray) -> float:
    score = model.predict(feature_vector.reshape(1, -1))[0]
    return float(np.clip(score, 0, 100))
