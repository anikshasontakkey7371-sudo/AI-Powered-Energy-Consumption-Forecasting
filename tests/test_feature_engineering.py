"""
Tests for Leakage-Free Feature Engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    create_feature_pipeline,
)


@pytest.fixture
def sample_hourly_df():
    """Create a sample 300-hour DataFrame for feature testing."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=300, freq="h")
    # Linear values to make lag and rolling calculations trivially verifiable
    energy_vals = np.arange(1, 301, dtype=float)
    return pd.DataFrame({"Datetime": dates, "Energy": energy_vals})


def test_calendar_features(sample_hourly_df):
    """Verify temporal calendar feature extraction."""
    feat_df = add_calendar_features(sample_hourly_df)

    assert "hour" in feat_df.columns
    assert "dayofweek" in feat_df.columns
    assert "month" in feat_df.columns
    assert "is_weekend" in feat_df.columns

    # 2024-01-01 is a Monday (dayofweek=0, is_weekend=0)
    assert feat_df["hour"].iloc[0] == 0
    assert feat_df["dayofweek"].iloc[0] == 0
    assert feat_df["month"].iloc[0] == 1
    assert feat_df["is_weekend"].iloc[0] == 0

    # Test hour 15 on Saturday 2024-01-06 (index 135)
    # 2024-01-06 is Saturday (dayofweek=5, is_weekend=1)
    sat_idx = 24 * 5 + 15
    assert feat_df["dayofweek"].iloc[sat_idx] == 5
    assert feat_df["is_weekend"].iloc[sat_idx] == 1
    assert feat_df["hour"].iloc[sat_idx] == 15


def test_lag_features(sample_hourly_df):
    """Verify autoregressive lag feature calculations."""
    feat_df = add_lag_features(sample_hourly_df, lags=[1, 2, 24])

    assert np.isnan(feat_df["lag_1"].iloc[0])
    # lag_1 at index 1 should equal Energy at index 0 (which is 1.0)
    assert feat_df["lag_1"].iloc[1] == sample_hourly_df["Energy"].iloc[0]
    # lag_2 at index 2 should equal Energy at index 0
    assert feat_df["lag_2"].iloc[2] == sample_hourly_df["Energy"].iloc[0]
    # lag_24 at index 24 should equal Energy at index 0
    assert feat_df["lag_24"].iloc[24] == sample_hourly_df["Energy"].iloc[0]


def test_no_future_data_leakage_in_rolling_features(sample_hourly_df):
    """
    CRITICAL TEST: Ensure rolling statistics do NOT include the target value y_t.

    If Energy is [1, 2, 3, ..., 300], at row index 24 (the 25th observation, where Energy = 25.0):
    - A leaky rolling mean of size 24 would include index 1..24 or 0..24 (including 25.0).
    - A leakage-free rolling mean over historical observations strictly computes the mean
      of index 0 to 23 (values 1 through 24).
    The mean of 1..24 is (1 + 24)/2 = 12.5.
    If leakage occurred and 25 was included, the mean would be 13.0 or higher.
    """
    feat_df = add_rolling_features(sample_hourly_df)

    assert "rolling_mean_24" in feat_df.columns
    assert "rolling_std_24" in feat_df.columns
    assert "rolling_mean_168" in feat_df.columns

    # First 24 rows must be NaN because index 24 is the first point with 24 past observations (0..23)
    assert feat_df["rolling_mean_24"].iloc[:24].isna().all()

    # Index 24: past 24 observations are indices 0 to 23 (values 1.0 to 24.0)
    expected_mean_at_24 = np.mean(sample_hourly_df["Energy"].iloc[0:24])  # 12.5
    assert np.isclose(feat_df["rolling_mean_24"].iloc[24], expected_mean_at_24)
    assert feat_df["rolling_mean_24"].iloc[24] == 12.5


def test_create_feature_pipeline(sample_hourly_df):
    """Verify full feature pipeline extracts expected columns and drops warmup NaNs."""
    X, y, feature_names, df_feat = create_feature_pipeline(sample_hourly_df, drop_na=True)

    # Longest lag/window is 168 hours; 300 - 168 = 132 rows
    assert len(X) == 132
    assert len(y) == 132
    assert not X.isna().any().any(), "No NaNs should remain after warm-up drop."
    assert "hour" in feature_names
    assert "lag_1" in feature_names
    assert "rolling_mean_24" in feature_names
