"""
Data Loader & Preprocessing Module.

This module is responsible for:
1. Loading energy consumption time-series data from CSV.
2. Generating a realistic, clearly labeled synthetic dataset if the CSV is missing.
3. Cleaning timestamps, handling duplicates, and sorting in chronological order.
4. Carefully handling missing values without leaking future information.
5. Documenting measurement units explicitly.
"""

from pathlib import Path
from typing import Tuple
import logging
import numpy as np
import pandas as pd

# Set up logging for clean diagnostic feedback
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Measurement Unit Specification:
# Energy values are measured in Megawatt-hours (MWh) per hourly interval.
# For residential or microgrid scale, this could also represent Kilowatt-hours (kWh).
ENERGY_UNIT = "MWh"
DATA_SOURCE_REAL = "Actual Observed Records"
DATA_SOURCE_SYNTHETIC = "Synthetic Demonstration Dataset (Simulated for Portfolio Testing)"


def generate_synthetic_energy_data(
    output_path: Path,
    start_date: str = "2024-01-01 00:00:00",
    periods: int = 8760,  # 365 days * 24 hours = 8,760 hourly readings
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly energy consumption data for testing and demonstration.

    NOTE ON SYNTHETIC DATA:
    This dataset is procedurally generated using mathematical simulation of real-world
    power grid load dynamics. It exhibits:
    - Diurnal (daily) patterns: low at night (~3-5 AM), peaks in morning (~9 AM) and evening (~7 PM).
    - Weekly patterns: lower commercial/industrial load on weekends (Saturday, Sunday).
    - Annual seasonality: increased heating demand in winter (Jan/Feb) and cooling in summer (Jul/Aug).
    - Random Gaussian noise and realistic holiday/trough effects.

    DISCLAIMER:
    This data is strictly for educational, demonstration, and algorithmic benchmarking.
    It does NOT represent a real commercial utility customer or actual energy savings.

    Args:
        output_path: Path where the generated CSV should be saved.
        start_date: Starting ISO timestamp string.
        periods: Number of hourly timestamps to generate.
        random_seed: Seed for reproducibility.

    Returns:
        DataFrame containing 'Datetime' and 'Energy' columns.
    """
    logger.info("Generating %d synthetic hourly energy consumption records...", periods)
    np.random.seed(random_seed)

    # 1. Hourly timestamp index
    date_range = pd.date_range(start=start_date, periods=periods, freq="h")

    # Extract time components
    hour = date_range.hour.to_numpy()
    day_of_week = date_range.dayofweek.to_numpy()  # 0=Monday, 6=Sunday
    day_of_year = date_range.dayofyear.to_numpy()

    # 2. Base load (average steady-state consumption in MWh)
    base_load = 50.0

    # 3. Daily cycle (diurnal variation with two peaks: morning ~9 AM and evening ~7 PM)
    # Using dual sinusoidal waves to approximate realistic load curves
    morning_peak = 14.0 * np.exp(-((hour - 9) ** 2) / (2 * (2.2 ** 2)))
    evening_peak = 18.0 * np.exp(-((hour - 19) ** 2) / (2 * (2.5 ** 2)))
    night_trough = -8.0 * np.exp(-((hour - 4) ** 2) / (2 * (2.0 ** 2)))
    daily_component = morning_peak + evening_peak + night_trough

    # 4. Weekly cycle (weekends consume ~15% less energy due to reduced commercial activity)
    is_weekend = (day_of_week >= 5).astype(float)
    weekly_component = -7.0 * is_weekend

    # 5. Seasonal cycle (higher demand during peak winter heating and summer cooling)
    # Annual cycle: peak in winter (day 15) and summer (day 200)
    winter_heating = 8.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    summer_cooling = 6.0 * np.cos(4 * np.pi * (day_of_year - 200) / 365.25)
    seasonal_component = winter_heating + summer_cooling

    # 6. Subtle annual growth trend (slight steady secular drift)
    trend = 0.0003 * np.arange(periods)

    # 7. Stochastic noise (unpredictable human variation, weather fluctuations)
    noise = np.random.normal(loc=0.0, scale=2.5, size=periods)

    # Combine all additive components
    energy = base_load + daily_component + weekly_component + seasonal_component + trend + noise

    # Clip to ensure physical plausibility (energy consumption cannot be negative)
    energy = np.clip(energy, a_min=10.0, a_max=None)

    df = pd.DataFrame(
        {
            "Datetime": date_range.strftime("%Y-%m-%d %H:%M:%S"),
            "Energy": np.round(energy, 2),
        }
    )

    # Ensure output directory exists and write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    # Write a marker indicating synthetic generation provenance
    marker_file = output_path.parent / ".synthetic_metadata.json"
    marker_file.write_text('{"is_synthetic": true, "generator": "procedural_simulation"}\n', encoding="utf-8")
    logger.info("Successfully generated and saved synthetic data to '%s'.", output_path)

    return df


def load_energy_data(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Load energy consumption data from CSV. If the file is missing or invalid,
    synthesize a benchmark dataset automatically.

    Args:
        csv_path: Path to the energy CSV file.

    Returns:
        Tuple of (DataFrame with raw data, is_synthetic boolean flag).
    """
    is_synthetic = False
    marker_file = csv_path.parent / ".synthetic_metadata.json"

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        logger.warning(
            "Energy data file '%s' not found or empty. Generating synthetic benchmark data...",
            csv_path,
        )
        df = generate_synthetic_energy_data(csv_path)
        is_synthetic = True
    else:
        logger.info("Loading energy consumption data from '%s'...", csv_path)
        df = pd.read_csv(csv_path)
        if marker_file.exists():
            is_synthetic = True

        # Validate mandatory column presence
        required_cols = {"Datetime", "Energy"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Dataset at '{csv_path}' must contain columns {required_cols}. "
                f"Found columns: {list(df.columns)}"
            )

    return df, is_synthetic


def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, validate, and chronologically align time-series energy data.

    Steps:
    1. Parse 'Datetime' column into timezone-naive pandas datetime.
    2. Check for and eliminate duplicate timestamps (keeping the first occurrence).
    3. Sort rows chronologically ascending by 'Datetime'.
    4. Validate energy column is numeric.
    5. Handle missing values carefully: forward-fill short gaps (persistence assumption)
       to avoid future-information leakage, followed by backward-fill for any initial gap.
    6. Ensure consecutive hourly cadence without introducing lookahead bias.

    Args:
        df: Input DataFrame with 'Datetime' and 'Energy' columns.

    Returns:
        Cleaned, sorted, and complete DataFrame with Datetime index or column.
    """
    df_clean = df.copy()

    # 1. Parse Datetime
    df_clean["Datetime"] = pd.to_datetime(df_clean["Datetime"], errors="coerce")
    invalid_dates = df_clean["Datetime"].isna().sum()
    if invalid_dates > 0:
        logger.warning("Dropping %d rows with unparseable timestamps.", invalid_dates)
        df_clean = df_clean.dropna(subset=["Datetime"])

    # 2. Sort chronologically
    df_clean = df_clean.sort_values(by="Datetime").reset_index(drop=True)

    # 3. Remove duplicate timestamps
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["Datetime"], keep="first").reset_index(drop=True)
    duplicates_removed = initial_len - len(df_clean)
    if duplicates_removed > 0:
        logger.warning("Removed %d duplicate timestamp entries.", duplicates_removed)

    # 4. Validate and convert Energy to numeric float
    df_clean["Energy"] = pd.to_numeric(df_clean["Energy"], errors="coerce")

    # 5. Check and enforce regular hourly frequency (handle missing timestamp gaps)
    if len(df_clean) > 1:
        min_dt = df_clean["Datetime"].min()
        max_dt = df_clean["Datetime"].max()
        full_hourly_index = pd.date_range(start=min_dt, end=max_dt, freq="h")
        if len(full_hourly_index) > len(df_clean):
            missing_hours = len(full_hourly_index) - len(df_clean)
            logger.warning(
                "Detected %d missing hourly slots between %s and %s. "
                "Reindexing to continuous hourly cadence.",
                missing_hours,
                min_dt,
                max_dt,
            )
            df_clean = (
                df_clean.set_index("Datetime")
                .reindex(full_hourly_index)
                .rename_axis("Datetime")
                .reset_index()
            )

    # 6. Handle missing values carefully:
    # Forward fill (persistence imputation) uses the most recent known observation,
    # strictly preventing lookahead bias.
    missing_count = df_clean["Energy"].isna().sum()
    if missing_count > 0:
        logger.warning(
            "Imputing %d missing energy values using forward-fill (persistence imputation).",
            missing_count,
        )
        df_clean["Energy"] = df_clean["Energy"].ffill()
        # Fallback backward-fill only for any leading boundary NaNs
        if df_clean["Energy"].isna().any():
            df_clean["Energy"] = df_clean["Energy"].bfill()

    # Re-verify cleanliness
    if df_clean.empty:
        raise ValueError("Energy dataset is empty after cleaning!")

    logger.info(
        "Cleaned dataset: %d records spanning from %s to %s.",
        len(df_clean),
        df_clean["Datetime"].min(),
        df_clean["Datetime"].max(),
    )
    return df_clean
