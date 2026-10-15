"""
Tests for 24-Hour Recursive Multi-Step Forecasting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import create_feature_pipeline
from src.forecasting import generate_recursive_24h_forecast, save_forecast
from src.model import train_random_forest


@pytest.fixture
def trained_pipeline_artifacts():
    """Build a mini dataset and fit a quick Random Forest model for testing."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=400, freq="h")
    np.random.seed(42)
    # Realistic energy values
    energy = 45.0 + 10.0 * np.sin(np.arange(400) * 2 * np.pi / 24) + np.random.normal(0, 1, 400)
    df = pd.DataFrame({"Datetime": dates, "Energy": energy})

    X, y, feature_names, _ = create_feature_pipeline(df, drop_na=True)
    model = train_random_forest(X, y, n_estimators=10, max_depth=5, random_state=42)

    return model, df, feature_names


def test_generate_recursive_24h_forecast(trained_pipeline_artifacts):
    """Verify 24-hour recursive forecast generates 24 consecutive non-negative predictions."""
    model, df_clean, feature_names = trained_pipeline_artifacts

    forecast_df = generate_recursive_24h_forecast(
        model=model,
        historical_df=df_clean,
        feature_names=feature_names,
        forecast_horizon=24,
        unit="MWh",
    )

    # 1. Exact horizon length
    assert len(forecast_df) == 24
    assert "Datetime" in forecast_df.columns
    assert "Forecast_Energy" in forecast_df.columns
    assert "Unit" in forecast_df.columns
    assert "Data_Source" in forecast_df.columns

    # 2. No nulls or infinite values
    assert not forecast_df["Forecast_Energy"].isna().any()
    assert (forecast_df["Forecast_Energy"] >= 0).all()

    # 3. Check chronological continuity
    last_hist_dt = pd.to_datetime(df_clean["Datetime"].iloc[-1])
    first_fore_dt = pd.to_datetime(forecast_df["Datetime"].iloc[0])
    last_fore_dt = pd.to_datetime(forecast_df["Datetime"].iloc[-1])

    assert first_fore_dt == last_hist_dt + pd.Timedelta(hours=1)
    assert last_fore_dt == last_hist_dt + pd.Timedelta(hours=24)

    # Verify each consecutive forecast step is exactly 1 hour
    fore_dts = pd.to_datetime(forecast_df["Datetime"])
    diffs = fore_dts.diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_save_forecast(tmp_path: Path):
    """Verify forecast is written to CSV properly."""
    sample_df = pd.DataFrame(
        {
            "Datetime": ["2024-01-02 01:00:00", "2024-01-02 02:00:00"],
            "Forecast_Energy": [54.2, 53.8],
            "Unit": ["MWh", "MWh"],
        }
    )
    dest_path = tmp_path / "outputs" / "next_24_hours.csv"
    save_forecast(sample_df, dest_path)

    assert dest_path.exists()
    reloaded = pd.read_csv(dest_path)
    assert len(reloaded) == 2
    assert "Forecast_Energy" in reloaded.columns


def test_recursive_forecast_uses_only_historical_and_earlier_predictions():
    """Verify recursive steps feed earlier forecast predictions strictly forward as lags."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({"Datetime": dates, "Energy": [50.0] * 200})

    class PredictConstantModel:
        """Mock model predicting an identifiable constant sequence."""
        def predict(self, X):
            # Predict lag_1 + 5.0
            return X["lag_1"].to_numpy() + 5.0

    feature_names = [
        "hour", "dayofweek", "month", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
        "rolling_mean_24", "rolling_std_24", "rolling_mean_168"
    ]
    model = PredictConstantModel()

    forecast_df = generate_recursive_24h_forecast(
        model=model,
        historical_df=df,
        feature_names=feature_names,
        forecast_horizon=3,
        unit="MWh",
    )

    # Step 1: last historical energy is 50.0 -> predicts 55.0
    assert forecast_df["Forecast_Energy"].iloc[0] == 55.0
    # Step 2: lag_1 is step 1 prediction (55.0) -> predicts 60.0
    assert forecast_df["Forecast_Energy"].iloc[1] == 60.0
    # Step 3: lag_1 is step 2 prediction (60.0) -> predicts 65.0
    assert forecast_df["Forecast_Energy"].iloc[2] == 65.0
