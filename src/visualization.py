"""
Visualization Module for Energy Consumption Forecasting.

Generates professional, publication-ready PNG charts:
1. actual_vs_predicted.png: Comparison on the chronological test set.
2. forecast_next_24_hours.png: 24-hour recursive forecast with historical context.
3. feature_importance.png: Random Forest MDI feature importance ranking.
4. residuals_distribution.png: Distribution and histogram of forecast residuals.
"""

from pathlib import Path
from typing import List
import logging
import matplotlib
# Use non-interactive Agg backend for robust headless server generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Plot styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    output_path: Path,
    zoom_hours: int = 168,  # Show last 7 days for clear visual inspection
    unit: str = "MWh",
) -> None:
    """
    Plot actual energy vs Random Forest model predictions on the test set.

    Args:
        predictions_df: DataFrame with 'Datetime', 'Actual_Energy', 'RF_Prediction'.
        output_path: PNG destination path.
        zoom_hours: Number of recent test hours to plot in detail.
        unit: Measurement unit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = predictions_df.tail(zoom_hours).copy()
    df_plot["Datetime"] = pd.to_datetime(df_plot["Datetime"])

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(
        df_plot["Datetime"],
        df_plot["Actual_Energy"],
        label="Actual Energy Consumption",
        color="#1f77b4",
        linewidth=2.2,
        alpha=0.9,
    )
    ax.plot(
        df_plot["Datetime"],
        df_plot["RF_Prediction"],
        label="Random Forest Prediction",
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
    )

    ax.set_title(
        f"Test Set Evaluation: Actual vs. Predicted Energy Consumption (Last {zoom_hours} Hours)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Date and Time", labelpad=8)
    ax.set_ylabel(f"Energy Consumption ({unit})", labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved actual vs predicted chart to '%s'.", output_path)


def plot_24h_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: Path,
    history_hours: int = 72,
    unit: str = "MWh",
) -> None:
    """
    Plot recent actual history seamlessly connected to the recursive 24-hour forecast.

    Args:
        historical_df: DataFrame with 'Datetime' and 'Energy'.
        forecast_df: DataFrame with 'Datetime' and 'Forecast_Energy'.
        output_path: Destination PNG path.
        history_hours: Number of preceding actual hours to show for context.
        unit: Measurement unit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hist_tail = historical_df.tail(history_hours).copy()
    hist_tail["Datetime"] = pd.to_datetime(hist_tail["Datetime"])

    fore_plot = forecast_df.copy()
    fore_plot["Datetime"] = pd.to_datetime(fore_plot["Datetime"])

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Historical line
    ax.plot(
        hist_tail["Datetime"],
        hist_tail["Energy"],
        label=f"Recent Actual Load (Last {history_hours}h)",
        color="#2ca02c",
        linewidth=2.2,
    )

    # Forecast line connected to last known actual point
    bridge_dt = [hist_tail["Datetime"].iloc[-1]] + list(fore_plot["Datetime"])
    bridge_vals = [hist_tail["Energy"].iloc[-1]] + list(fore_plot["Forecast_Energy"])

    ax.plot(
        bridge_dt,
        bridge_vals,
        label="Recursive 24-Hour Forecast (RF)",
        color="#d62728",
        linewidth=2.4,
        linestyle="-",
        marker="o",
        markersize=4,
    )

    # Shaded forecast horizon region
    ax.axvspan(
        fore_plot["Datetime"].iloc[0],
        fore_plot["Datetime"].iloc[-1],
        color="#d62728",
        alpha=0.08,
        label="24-Hour Forecast Horizon",
    )

    ax.set_title("AI-Powered 24-Hour Ahead Energy Consumption Forecast", fontweight="bold", pad=12)
    ax.set_xlabel("Datetime", labelpad=8)
    ax.set_ylabel(f"Forecasted Energy ({unit})", labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved 24-hour forecast chart to '%s'.", output_path)


def plot_feature_importance(
    model: any,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """
    Plot top feature importances from the trained Random Forest model.

    Args:
        model: Trained RandomForestRegressor.
        feature_names: List of feature names.
        output_path: Destination PNG path.
        top_n: Number of top features to display.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#3b82f6", edgecolor="#1d4ed8")

    # Add numeric labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#333333",
        )

    ax.set_title("Random Forest Top Feature Importances (MDI)", fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", labelpad=8)
    ax.set_xlim(0, max(feat_df["Importance"]) * 1.15)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved feature importance chart to '%s'.", output_path)


def plot_residuals_distribution(
    predictions_df: pd.DataFrame,
    output_path: Path,
    unit: str = "MWh",
) -> None:
    """
    Plot the distribution of forecast errors (residuals = Actual - Predicted).

    A well-calibrated forecasting model will have residuals symmetrically centered at zero.

    Args:
        predictions_df: DataFrame containing 'Residual' or 'Actual_Energy' and 'RF_Prediction'.
        output_path: Destination PNG path.
        unit: Measurement unit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "Residual" in predictions_df.columns:
        residuals = predictions_df["Residual"].to_numpy()
    else:
        residuals = predictions_df["Actual_Energy"] - predictions_df["RF_Prediction"]

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(
        residuals, bins=40, density=True, color="#8b5cf6", edgecolor="#6d28d9", alpha=0.75
    )

    # Plot fitted Gaussian distribution curve for reference
    x_grid = np.linspace(min(residuals), max(residuals), 200)
    pdf = (1 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mean_res) / std_res) ** 2)
    ax.plot(x_grid, pdf, color="#4c1d95", linewidth=2.0, label=f"Normal Fit (μ={mean_res:.2f}, σ={std_res:.2f})")

    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")
    ax.set_title("Test Set Residuals Distribution (Actual - Predicted)", fontweight="bold", pad=12)
    ax.set_xlabel(f"Error ({unit})", labelpad=8)
    ax.set_ylabel("Probability Density", labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved residuals distribution chart to '%s'.", output_path)
