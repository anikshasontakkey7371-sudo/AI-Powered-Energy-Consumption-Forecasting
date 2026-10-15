"""
Feature Engineering Module for Time-Series Energy Forecasting.

This module builds time-series features strictly without lookahead bias or future-data leakage:
1. Temporal/Calendar features (Hour, Day of Week, Month, Is Weekend).
2. Autoregressive Lag features (lag_1, lag_2, lag_3, lag_24, lag_48, lag_168).
3. Rolling-window features (rolling_mean_24, rolling_std_24, rolling_mean_168) calculated
   strictly on shifted (lagged) observations so that at time t, only data up to t-1 is used.
"""

from typing import List, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# List of lag offsets in hours
# 1, 2, 3: immediate recent autocorrelation
# 24, 48: daily periodicity (yesterday, two days ago)
# 168: weekly periodicity (same hour last week)
LAG_HOURS: List[int] = [1, 2, 3, 24, 48, 168]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar/temporal features from the 'Datetime' column.

    Features created:
    - 'hour': Hour of the day [0-23] (captures daily diurnal demand curve).
    - 'dayofweek': Day of the week [0=Monday, 6=Sunday] (workday vs weekend shift).
    - 'month': Month of the year [1-12] (annual heating/cooling seasonality).
    - 'is_weekend': Binary indicator (1 if Saturday/Sunday, else 0).

    Args:
        df: DataFrame containing 'Datetime' column.

    Returns:
        DataFrame with additional calendar feature columns.
    """
    df_feat = df.copy()
    dt = pd.to_datetime(df_feat["Datetime"])

    df_feat["hour"] = dt.dt.hour
    df_feat["dayofweek"] = dt.dt.dayofweek
    df_feat["month"] = dt.dt.month
    df_feat["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    return df_feat


def add_lag_features(df: pd.DataFrame, target_col: str = "Energy", lags: List[int] = None) -> pd.DataFrame:
    """
    Create autoregressive lag features for the target variable.

    Each lag represents y_{t - k}. This is historical past data and contains no future leakage.

    Args:
        df: DataFrame sorted chronologically.
        target_col: Name of the column to lag (e.g., 'Energy').
        lags: List of integer hour shifts. Defaults to LAG_HOURS.

    Returns:
        DataFrame with lag columns added (e.g. 'lag_1', 'lag_24').
    """
    if lags is None:
        lags = LAG_HOURS

    df_feat = df.copy()
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    return df_feat


def add_rolling_features(df: pd.DataFrame, target_col: str = "Energy") -> pd.DataFrame:
    """
    Compute rolling-window statistics WITHOUT future-data leakage.

    CRITICAL LEAKAGE PREVENTION:
    At timestamp t, the target value y_t is unknown when forecasting.
    Therefore, rolling statistics MUST be calculated over y_{t-1}, y_{t-2}, ...,
    NOT including y_t. We achieve this by shifting the target by 1 step before
    computing the rolling aggregation:
        shift(1).rolling(window=W).mean()

    Features created:
    - 'rolling_mean_24': 24-hour moving average of past consumption (t-24 to t-1).
    - 'rolling_std_24': 24-hour moving standard deviation of past consumption.
    - 'rolling_mean_168': 7-day (168-hour) moving average of past consumption.

    Args:
        df: DataFrame sorted chronologically.
        target_col: Column on which to calculate rolling statistics.

    Returns:
        DataFrame with rolling feature columns added.
    """
    df_feat = df.copy()

    # Shift by 1 hour to exclude the current observation y_t
    shifted_target = df_feat[target_col].shift(1)

    # 24-hour rolling window on historical data
    df_feat["rolling_mean_24"] = shifted_target.rolling(window=24, min_periods=24).mean()
    df_feat["rolling_std_24"] = shifted_target.rolling(window=24, min_periods=24).std()

    # 7-day (168 hours) rolling window on historical data
    df_feat["rolling_mean_168"] = shifted_target.rolling(window=168, min_periods=168).mean()

    return df_feat


def create_feature_pipeline(
    df: pd.DataFrame,
    target_col: str = "Energy",
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Execute the full end-to-end feature engineering pipeline.

    Steps:
    1. Verify dataset is sorted chronologically.
    2. Add calendar features.
    3. Add lag features.
    4. Add leakage-free rolling features.
    5. Drop warm-up rows (first 168 hours) that contain NaN values due to lag windows.
    6. Separate feature matrix X, target y, and metadata.

    Args:
        df: Input cleaned DataFrame with 'Datetime' and target_col.
        target_col: Target column name.
        drop_na: Whether to drop initial warm-up rows with NaN features.

    Returns:
        Tuple of (X DataFrame, y Series, list of feature names, processed full DataFrame).
    """
    # 1. Ensure chronological order
    df_sorted = df.sort_values(by="Datetime").reset_index(drop=True)

    # 2. Add calendar features
    df_feat = add_calendar_features(df_sorted)

    # 3. Add lag features
    df_feat = add_lag_features(df_feat, target_col=target_col, lags=LAG_HOURS)

    # 4. Add rolling features (on lagged data)
    df_feat = add_rolling_features(df_feat, target_col=target_col)

    # Identify feature columns
    excluded_cols = {"Datetime", target_col}
    feature_names = [col for col in df_feat.columns if col not in excluded_cols]

    # 5. Drop NaN rows created by lags
    if drop_na:
        initial_len = len(df_feat)
        df_feat = df_feat.dropna().reset_index(drop=True)
        dropped_rows = initial_len - len(df_feat)
        logger.info(
            "Dropped %d initial warm-up rows containing NaN features (due to 168h lag). Remaining: %d rows.",
            dropped_rows,
            len(df_feat),
        )

    X = df_feat[feature_names].copy()
    y = df_feat[target_col].copy()

    return X, y, feature_names, df_feat
