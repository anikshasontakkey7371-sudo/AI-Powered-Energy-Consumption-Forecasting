"""
Tests for Data Ingestion, Synthesis, and Preprocessing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    clean_energy_data,
    generate_synthetic_energy_data,
    load_energy_data,
)


def test_generate_synthetic_energy_data(tmp_path: Path):
    """Verify synthetic data generation generates valid schema and non-negative values."""
    csv_path = tmp_path / "test_energy.csv"
    periods = 200
    df = generate_synthetic_energy_data(csv_path, periods=periods, random_seed=42)

    assert csv_path.exists()
    assert len(df) == periods
    assert list(df.columns) == ["Datetime", "Energy"]
    assert (df["Energy"] > 0).all(), "All energy values should be positive."
    assert not df["Energy"].isna().any(), "Synthetic data should contain no NaNs."


def test_load_energy_data_auto_generates_when_missing(tmp_path: Path):
    """Verify loader generates synthetic dataset when target file does not exist."""
    missing_csv = tmp_path / "non_existent.csv"
    assert not missing_csv.exists()

    df, is_synthetic = load_energy_data(missing_csv)
    assert is_synthetic is True
    assert missing_csv.exists()
    assert len(df) == 8760
    assert "Datetime" in df.columns
    assert "Energy" in df.columns


def test_clean_energy_data_sorting_and_duplicates():
    """Verify data cleaning sorts ascending and eliminates duplicate timestamps."""
    raw_data = pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 02:00:00",
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 01:00:00",  # Duplicate timestamp
            ],
            "Energy": [50.0, 45.0, 48.0, 99.0],
        }
    )

    clean_df = clean_energy_data(raw_data)

    assert len(clean_df) == 3
    # Verify chronological ordering
    assert clean_df["Datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert clean_df["Datetime"].iloc[1] == pd.Timestamp("2024-01-01 01:00:00")
    assert clean_df["Datetime"].iloc[2] == pd.Timestamp("2024-01-01 02:00:00")
    # Verify first duplicate occurrence was retained (48.0, not 99.0)
    assert clean_df["Energy"].iloc[1] == 48.0


def test_clean_energy_data_missing_values():
    """Verify missing values are handled without unhandled NaNs."""
    raw_data = pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 02:00:00",
            ],
            "Energy": [50.0, np.nan, 55.0],
        }
    )

    clean_df = clean_energy_data(raw_data)
    assert not clean_df["Energy"].isna().any()
    # Forward fill should impute 50.0 for the missing 01:00:00 slot
    assert clean_df["Energy"].iloc[1] == 50.0


def test_clean_energy_data_invalid_timestamps_and_gaps():
    """Verify unparseable dates are dropped and missing hourly slots are reindexed."""
    raw_data = pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 00:00:00",
                "not-a-valid-timestamp",  # Corrupted row
                "2024-01-01 03:00:00",  # Gaps at 01:00 and 02:00
            ],
            "Energy": [50.0, 999.0, 60.0],
        }
    )

    clean_df = clean_energy_data(raw_data)

    # Corrupted date dropped, gap reindexed to 4 continuous hours: 00:00, 01:00, 02:00, 03:00
    assert len(clean_df) == 4
    assert clean_df["Datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert clean_df["Datetime"].iloc[1] == pd.Timestamp("2024-01-01 01:00:00")
    assert clean_df["Datetime"].iloc[2] == pd.Timestamp("2024-01-01 02:00:00")
    assert clean_df["Datetime"].iloc[3] == pd.Timestamp("2024-01-01 03:00:00")
    # Forward fill should propagate 50.0 into hours 01:00 and 02:00
    assert clean_df["Energy"].iloc[1] == 50.0
    assert clean_df["Energy"].iloc[2] == 50.0
    assert clean_df["Energy"].iloc[3] == 60.0
