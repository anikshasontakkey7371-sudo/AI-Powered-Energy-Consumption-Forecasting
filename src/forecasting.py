"""
Recursive Multi-Step 24-Hour Forecasting Module.

In multi-step time series forecasting, predicting multiple future steps ahead
can be achieved using a Recursive (Autoregressive) strategy:
1. For step t+1: use actual historical observations to construct features, then predict y_hat_{t+1}.
2. For step t+2: feed the newly predicted y_hat_{t+1} back into the feature pipeline
   as lag_1, while previous observations shift down (lag_2, etc.).
3. Re-calculate rolling statistics dynamically including the new forecast.
4. Repeat recursively for all 24 hours into the future.
"""

from datetime import datetime
from pathlib import Path
from typing import List
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_recursive_24h_forecast(
    model: any,
    historical_df: pd.DataFrame,
    feature_names: List[str],
    forecast_horizon: int = 24,
    unit: str = "MWh",
    is_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Generate an autoregressive recursive 24-hour ahead energy forecast.

    Args:
        model: Trained scikit-learn regressor (e.g., RandomForestRegressor).
        historical_df: Cleaned historical DataFrame containing at least 168 hours
                       of continuous data with 'Datetime' and 'Energy' columns.
        feature_names: Exact list of feature column names used to fit the model.
        forecast_horizon: Number of future hours to forecast (default: 24).
        unit: Measurement unit string ('MWh').
        is_synthetic: Whether the underlying historical dataset is synthetic.

    Returns:
        DataFrame containing 24 future hourly forecasts with timestamps and provenance.
    """
    if len(historical_df) < 168:
        raise ValueError(
            f"Historical data must contain at least 168 hours to compute lag_168 "
            f"and rolling_mean_168. Provided: {len(historical_df)} rows."
        )

    logger.info("Initiating recursive %d-hour ahead energy forecast...", forecast_horizon)

    # Work on a working copy of the series containing Datetime and Energy
    working_df = historical_df[["Datetime", "Energy"]].copy()
    working_df["Datetime"] = pd.to_datetime(working_df["Datetime"])
    working_df = working_df.sort_values(by="Datetime").reset_index(drop=True)

    # Determine starting forecast timestamp
    last_timestamp = working_df["Datetime"].iloc[-1]
    logger.info("Last known historical timestamp: %s", last_timestamp)

    forecast_records = []
    generated_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_label = (
        "Synthetic Demonstration Model (Educational Simulation)"
        if is_synthetic
        else "Trained Production Model"
    )

    for step in range(1, forecast_horizon + 1):
        target_timestamp = last_timestamp + pd.Timedelta(hours=step)

        # 1. Deterministic calendar features
        hour = target_timestamp.hour
        dayofweek = target_timestamp.dayofweek
        month = target_timestamp.month
        is_weekend = int(dayofweek >= 5)

        # 2. Extract autoregressive lag values from the tail of the working series
        # The working series includes all actual history plus any previously forecasted steps.
        # lag_1 is the observation at working_df.iloc[-1]
        lag_1 = working_df["Energy"].iloc[-1]
        lag_2 = working_df["Energy"].iloc[-2]
        lag_3 = working_df["Energy"].iloc[-3]
        lag_24 = working_df["Energy"].iloc[-24]
        lag_48 = working_df["Energy"].iloc[-48]
        lag_168 = working_df["Energy"].iloc[-168]

        # 3. Compute rolling statistics strictly on historical + already forecasted steps
        # Last 24 observations
        rolling_mean_24 = working_df["Energy"].iloc[-24:].mean()
        rolling_std_24 = working_df["Energy"].iloc[-24:].std(ddof=1)
        # Last 168 observations (1 week)
        rolling_mean_168 = working_df["Energy"].iloc[-168:].mean()

        # Assemble feature row in exact expected order
        feature_dict = {
            "hour": hour,
            "dayofweek": dayofweek,
            "month": month,
            "is_weekend": is_weekend,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_24": lag_24,
            "lag_48": lag_48,
            "lag_168": lag_168,
            "rolling_mean_24": rolling_mean_24,
            "rolling_std_24": rolling_std_24,
            "rolling_mean_168": rolling_mean_168,
        }

        # Format into single-row DataFrame matching training columns
        X_step = pd.DataFrame([feature_dict])[feature_names]

        # Predict future energy consumption
        pred_value = float(model.predict(X_step)[0])
        # Energy cannot be negative
        pred_value = max(0.0, round(pred_value, 2))

        # Append to working series so subsequent steps can use it as lag_1, lag_2, etc.
        new_row = pd.DataFrame(
            {"Datetime": [target_timestamp], "Energy": [pred_value]}
        )
        working_df = pd.concat([working_df, new_row], ignore_index=True)

        forecast_records.append(
            {
                "Datetime": target_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Forecast_Energy": pred_value,
                "Unit": unit,
                "Model": type(model).__name__,
                "Generated_At": generated_at_str,
                "Data_Source": source_label,
            }
        )

    forecast_df = pd.DataFrame(forecast_records)
    logger.info(
        "Successfully generated 24-hour forecast from %s to %s.",
        forecast_df["Datetime"].iloc[0],
        forecast_df["Datetime"].iloc[-1],
    )
    return forecast_df


def save_forecast(forecast_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the 24-hour forecast DataFrame to a CSV file.

    Args:
        forecast_df: DataFrame of 24-hour forecasts.
        output_path: Filepath destination (e.g. outputs/next_24_hours.csv).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    logger.info("Saved 24-hour forecast to '%s'.", output_path)
