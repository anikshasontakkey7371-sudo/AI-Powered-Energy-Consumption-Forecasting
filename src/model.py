"""
Modeling & Evaluation Module for Energy Consumption Forecasting.

This module provides:
1. Chronological time-series train/test splitting (no shuffle, no future leakage).
2. Naive Persistence Baseline model (predicts previous hour's consumption).
3. RandomForestRegressor training with sensible default hyperparameters.
4. Metric evaluation: MAE, RMSE, and R-squared (R2).
5. Model persistence and artifact serialization using Joblib.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series chronologically: earlier observations for training,
    later observations for testing.

    NEVER randomly shuffle time-series data, as doing so leaks future patterns into the past.

    Args:
        df: Processed DataFrame containing 'Datetime' column, sorted ascending.
        split_ratio: Fraction of historical observations reserved for training (e.g. 0.80).

    Returns:
        Tuple of (train_df, test_df).
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")

    n_total = len(df)
    split_idx = int(n_total * split_ratio)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Verification of strict chronological separation
    max_train_date = pd.to_datetime(train_df["Datetime"]).max()
    min_test_date = pd.to_datetime(test_df["Datetime"]).min()

    if max_train_date >= min_test_date:
        raise ValueError(
            f"Chronological split failed: max train date ({max_train_date}) "
            f"is >= min test date ({min_test_date})"
        )

    logger.info(
        "Chronological Split: %d train records (%s to %s) | %d test records (%s to %s)",
        len(train_df),
        train_df["Datetime"].min(),
        train_df["Datetime"].max(),
        len(test_df),
        test_df["Datetime"].min(),
        test_df["Datetime"].max(),
    )
    return train_df, test_df


class NaivePersistenceBaseline:
    """
    Simple persistence baseline model for time series forecasting.

    Hypothesis: The energy consumption in the current hour is equal to
    the energy consumption in the immediate previous hour (y_t = y_{t-1}).
    In time series literature, this is known as the 'Naive 1' or 'Persistence' forecast.
    Any machine learning model must convincingly outperform this simple benchmark
    to justify its operational deployment.
    """

    def __init__(self, lag_feature: str = "lag_1"):
        self.lag_feature = lag_feature
        self.model_name = "Naive Baseline (Previous Hour)"

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NaivePersistenceBaseline":
        """No training required for naive persistence."""
        if self.lag_feature not in X.columns:
            raise KeyError(f"Required lag feature '{self.lag_feature}' not found in X.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the immediate previous hour consumption (lag_1)."""
        if self.lag_feature not in X.columns:
            raise KeyError(f"Required lag feature '{self.lag_feature}' not found in X.")
        return X[self.lag_feature].to_numpy()


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor on chronological training features.

    Random Forest is an ensemble of decision trees that excels at capturing
    non-linear interactions between temporal features (hour, weekday) and
    autoregressive lags without requiring feature scaling.

    Args:
        X_train: Feature matrix for training.
        y_train: Target values for training.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each decision tree to control overfitting.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at each leaf node.
        random_state: Seed for deterministic reproducibility.

    Returns:
        Fitted RandomForestRegressor model.
    """
    logger.info("Training RandomForestRegressor with %d estimators (max_depth=%d)...", n_estimators, max_depth)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=1,
    )
    rf_model.fit(X_train, y_train)
    logger.info("RandomForestRegressor training completed successfully.")
    return rf_model


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    unit: str = "MWh",
) -> Dict[str, Any]:
    """
    Calculate core time-series regression performance metrics:
    - MAE (Mean Absolute Error): Average absolute magnitude of errors in original units.
    - RMSE (Root Mean Squared Error): Penalizes larger errors more heavily.
    - R-squared (R2): Proportion of variance explained by the model compared to mean baseline.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.
        model_name: Human-readable label for reporting.
        unit: Unit of measurement (e.g. 'MWh').

    Returns:
        Dictionary containing metric values and formatted descriptions.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    # root_mean_squared_error is standard in scikit-learn >= 1.4
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = float(root_mean_squared_error(y_true, y_pred))
    except ImportError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "model_name": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2_score": round(r2, 4),
        "unit": unit,
        "sample_size": len(y_true),
    }

    logger.info(
        "Evaluation [%s] -> MAE: %.3f %s | RMSE: %.3f %s | R²: %.4f",
        model_name,
        metrics["mae"],
        unit,
        metrics["rmse"],
        unit,
        metrics["r2_score"],
    )
    return metrics


def save_model_artifacts(
    model: Any,
    feature_names: List[str],
    metrics_summary: pd.DataFrame,
    output_path: Path,
    metadata: Dict[str, Any] = None,
) -> None:
    """
    Serialize model and associated metadata to disk using Joblib.

    Saving feature names alongside the model ensures that during deployment or
    recursive forecasting, the input columns are matched exactly in name and order.

    Args:
        model: Trained model object.
        feature_names: List of feature names used during training.
        metrics_summary: DataFrame of comparison metrics.
        output_path: Target .joblib file destination.
        metadata: Optional dictionary of additional provenance metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_bundle = {
        "model": model,
        "feature_names": feature_names,
        "metrics_summary": metrics_summary.to_dict(orient="records"),
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "energy_unit": "MWh",
            **(metadata or {}),
        },
    }

    # 1. Save unified bundle
    joblib.dump(artifact_bundle, output_path)

    # 2. Save feature names explicitly as a standalone Joblib file (Requirement 11)
    feature_names_path = output_path.parent / "feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)

    # 3. Save convenience aliases for standard estimators and feature list
    joblib.dump(feature_names, output_path.parent / "features.joblib")
    joblib.dump(model, output_path.parent / "model.joblib")

    logger.info("Saved model artifact to '%s'.", output_path)
    logger.info("Saved feature names artifact to '%s'.", feature_names_path)


def load_model_artifacts(model_path: Path) -> Dict[str, Any]:
    """
    Load saved model artifact bundle from disk using Joblib.

    Args:
        model_path: Path to the .joblib file.

    Returns:
        Dictionary bundle with keys ['model', 'feature_names', 'metrics_summary', 'metadata'].
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    artifact = joblib.load(model_path)
    logger.info("Loaded model artifact from '%s'.", model_path)
    return artifact
