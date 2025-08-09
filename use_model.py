"""Utility helpers to load and run the trained CS2 predictor model.

Example
-------
```
from use_model import load_model, predict

model = load_model()
features = {...}  # dictionary or DataFrame with required feature columns
pred, prob = predict(model, features)
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import pandas as pd

from main import EnsemblePredictor

# Default locations for the trained model artifacts.
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"


def load_model(
    model_path: Union[str, Path] = MODEL_PATH,
    feature_columns_path: Union[str, Path] = FEATURES_PATH,
) -> EnsemblePredictor:
    """Load the pre-trained ensemble model.

    Parameters
    ----------
    model_path:
        Path to the ensemble ``model.joblib`` file.
    feature_columns_path:
        Path to the ``feature_columns.joblib`` file.

    Returns
    -------
    EnsemblePredictor
        Instance ready to generate predictions.
    """
    return EnsemblePredictor.load(str(model_path), str(feature_columns_path))


def _ensure_dataframe(data: Union[pd.DataFrame, Dict[str, Iterable]]) -> pd.DataFrame:
    """Convert mapping-like ``data`` into a one-row :class:`DataFrame` if needed."""
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame([data])


def predict(
    model: EnsemblePredictor,
    data: Union[pd.DataFrame, Dict[str, Iterable]],
) -> Tuple[pd.Series, Iterable[float]]:
    """Generate class predictions and win probabilities.

    Parameters
    ----------
    model:
        Loaded :class:`EnsemblePredictor` instance.
    data:
        Features describing the match. May be a DataFrame or a mapping of
        column names to values.

    Returns
    -------
    tuple
        ``(prediction, probability)`` where ``prediction`` is the predicted
        winner label and ``probability`` is the probability of team1 winning.
    """
    features = _ensure_dataframe(data)
    pred = model.predict(features)
    proba = model.predict_proba(features)[:, 1]
    return pred, proba


__all__ = ["load_model", "predict"]
