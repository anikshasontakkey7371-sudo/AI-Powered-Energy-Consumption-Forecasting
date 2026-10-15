"""
Tests for Model Training, Chronological Splitting, and Evaluation.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.model import (
    NaivePersistenceBaseline,
    calculate_metrics,
    chronological_split,
    load_model_artifacts,
    save_model_artifacts,
    train_random_forest,
)


@pytest.fixture
def dummy_dataset():
    """Create a dummy time series dataset with features."""
    dates = pd.date_range("2024-01-01", periods=500, freq="h")
    np.random.seed(42)
    energy = 50 + np.sin(np.arange(500) / 10) * 15 + np.random.normal(0, 1, 500)

    df = pd.DataFrame(
        {
            "Datetime": dates,
            "hour": dates.hour,
            "dayofweek": dates.dayofweek,
            "month": dates.month,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "lag_1": pd.Series(energy).shift(1).bfill(),
            "lag_2": pd.Series(energy).shift(2).bfill(),
            "lag_24": pd.Series(energy).shift(24).bfill(),
            "rolling_mean_24": pd.Series(energy).shift(1).rolling(24).mean().bfill(),
            "Energy": energy,
        }
    )
    return df


def test_chronological_split(dummy_dataset):
    """Verify chronological split guarantees earlier data in train, later data in test."""
    train_df, test_df = chronological_split(dummy_dataset, split_ratio=0.8)

    assert len(train_df) == 400
    assert len(test_df) == 100

    # Strict chronological partition test
    assert train_df["Datetime"].max() < test_df["Datetime"].min()
    assert test_df["Datetime"].iloc[0] == train_df["Datetime"].iloc[-1] + pd.Timedelta(hours=1)


def test_naive_persistence_baseline(dummy_dataset):
    """Verify naive baseline predicts the lag_1 value."""
    baseline = NaivePersistenceBaseline(lag_feature="lag_1")
    X = dummy_dataset[["lag_1", "hour"]]
    preds = baseline.predict(X)

    assert len(preds) == len(dummy_dataset)
    assert np.array_equal(preds, dummy_dataset["lag_1"].to_numpy())


def test_random_forest_training_and_metrics(dummy_dataset):
    """Verify Random Forest model trains, predicts, and metrics compute accurately."""
    feature_cols = ["hour", "dayofweek", "lag_1", "lag_2", "rolling_mean_24"]
    X_train = dummy_dataset.iloc[:400][feature_cols]
    y_train = dummy_dataset.iloc[:400]["Energy"]
    X_test = dummy_dataset.iloc[400:][feature_cols]
    y_test = dummy_dataset.iloc[400:]["Energy"]

    rf_model = train_random_forest(X_train, y_train, n_estimators=10, max_depth=5, random_state=42)
    preds = rf_model.predict(X_test)

    assert len(preds) == 100
    assert np.isfinite(preds).all()

    metrics = calculate_metrics(y_test.to_numpy(), preds, model_name="RandomForestTest")
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["r2_score"] > 0.5  # Signal is strong sinusoidal curve


def test_save_and_load_model_artifacts(tmp_path: Path, dummy_dataset):
    """Verify Joblib serialization and deserialization preserves model and feature list."""
    feature_cols = ["hour", "dayofweek", "lag_1"]
    X = dummy_dataset[feature_cols]
    y = dummy_dataset["Energy"]

    rf_model = train_random_forest(X, y, n_estimators=10, max_depth=4, random_state=42)
    metrics_summary = pd.DataFrame([{"model_name": "TestModel", "mae": 1.2, "rmse": 1.5, "r2_score": 0.85}])

    artifact_path = tmp_path / "test_model.joblib"
    save_model_artifacts(rf_model, feature_cols, metrics_summary, artifact_path)

    assert artifact_path.exists()
    assert (tmp_path / "feature_names.joblib").exists()
    assert (tmp_path / "model.joblib").exists()
    assert joblib.load(tmp_path / "feature_names.joblib") == feature_cols

    loaded = load_model_artifacts(artifact_path)
    assert "model" in loaded
    assert "feature_names" in loaded
    assert loaded["feature_names"] == feature_cols

    # Ensure loaded model predicts identically
    orig_preds = rf_model.predict(X.iloc[:5])
    loaded_preds = loaded["model"].predict(X.iloc[:5])
    assert np.allclose(orig_preds, loaded_preds)


def test_baseline_and_random_forest_use_identical_test_period(dummy_dataset):
    """Verify Baseline and Random Forest evaluate on the exact same test timestamps."""
    train_df, test_df = chronological_split(dummy_dataset, split_ratio=0.8)
    feature_cols = ["hour", "dayofweek", "lag_1", "lag_2", "rolling_mean_24"]

    baseline = NaivePersistenceBaseline(lag_feature="lag_1")
    rf_model = train_random_forest(
        train_df[feature_cols], train_df["Energy"], n_estimators=10, max_depth=4, random_state=42
    )

    base_preds = baseline.predict(test_df[feature_cols])
    rf_preds = rf_model.predict(test_df[feature_cols])

    # Both predictions must evaluate against the exact same test target
    assert len(base_preds) == len(rf_preds) == len(test_df)
    assert not np.isnan(base_preds).any()
    assert not np.isnan(rf_preds).any()
