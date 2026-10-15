"""
AI-Powered Energy Consumption Forecasting System
Main Pipeline Orchestrator.

This script executes the complete machine learning workflow:
1. Verifies/generates energy data in 'data/energy.csv'.
2. Cleans timestamps and handles missing values carefully.
3. Engineers temporal, lag, and rolling features strictly without lookahead bias.
4. Chronologically partitions historical data into training and test splits.
5. Trains a naive persistence baseline and a RandomForestRegressor.
6. Evaluates both models using MAE, RMSE, and R-squared.
7. Serializes the trained model and feature names using Joblib.
8. Generates a recursive 24-hour forecast saved to 'outputs/next_24_hours.csv'.
9. Saves evaluation metrics, prediction logs, and publication-ready charts.
"""

from pathlib import Path
import logging
import sys
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("energy_forecasting")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Modules
from src.data_loader import (
    ENERGY_UNIT,
    clean_energy_data,
    load_energy_data,
)
from src.feature_engineering import create_feature_pipeline
from src.forecasting import generate_recursive_24h_forecast, save_forecast
from src.model import (
    NaivePersistenceBaseline,
    calculate_metrics,
    chronological_split,
    save_model_artifacts,
    train_random_forest,
)
from src.visualization import (
    plot_24h_forecast,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals_distribution,
)


def ensure_directory_structure(base_dir: Path) -> dict:
    """
    Ensure all required project directories exist.
    """
    directories = {
        "data": base_dir / "data",
        "notebooks": base_dir / "notebooks",
        "src": base_dir / "src",
        "models": base_dir / "models",
        "outputs": base_dir / "outputs",
        "images": base_dir / "images",
        "docs": base_dir / "docs",
    }
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def run_pipeline() -> None:
    """
    Execute the end-to-end machine learning pipeline.
    """
    print("\n" + "=" * 70)
    print("   AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM")
    print("   End-to-End Machine Learning & Evaluation Pipeline")
    print("=" * 70 + "\n")

    # 1. Setup project directories
    dirs = ensure_directory_structure(PROJECT_ROOT)
    data_csv_path = dirs["data"] / "energy.csv"

    # 2. Load or synthesize data
    logger.info("STEP 1: Ingesting dataset...")
    df_raw, is_synthetic = load_energy_data(data_csv_path)
    if is_synthetic:
        print("\n[NOTE] No existing 'data/energy.csv' found.")
        print("       Generated synthetic hourly demonstration dataset (8,760 hours = 1 year).")
        print("       Unit: Megawatt-hours (MWh). Clearly labeled for portfolio evaluation.\n")

    # 3. Clean and validate time series
    logger.info("STEP 2: Cleaning and validating time series...")
    df_clean = clean_energy_data(df_raw)

    # 4. Feature engineering (leakage-free)
    logger.info("STEP 3: Engineering leakage-free features (calendar, lags, rolling)...")
    X, y, feature_names, df_processed = create_feature_pipeline(
        df_clean, target_col="Energy", drop_na=True
    )
    logger.info("Extracted %d features: %s", len(feature_names), feature_names)

    # 5. Chronological train/test split
    logger.info("STEP 4: Partitioning dataset chronologically (80%% train / 20%% test)...")
    train_df, test_df = chronological_split(df_processed, split_ratio=0.80)

    X_train = train_df[feature_names]
    y_train = train_df["Energy"]
    X_test = test_df[feature_names]
    y_test = test_df["Energy"]

    # 6. Evaluate Naive Baseline (Persistence: previous hour)
    logger.info("STEP 5: Evaluating Naive Persistence Baseline (lag_1)...")
    baseline = NaivePersistenceBaseline(lag_feature="lag_1")
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    metrics_baseline = calculate_metrics(
        y_test.to_numpy(),
        y_pred_baseline,
        model_name="Naive Baseline (Previous Hour)",
        unit=ENERGY_UNIT,
    )

    # 7. Train and evaluate Random Forest Regressor
    logger.info("STEP 6: Training RandomForestRegressor...")
    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = calculate_metrics(
        y_test.to_numpy(),
        y_pred_rf,
        model_name="RandomForestRegressor",
        unit=ENERGY_UNIT,
    )

    # 8. Compare Model Metrics
    metrics_summary = pd.DataFrame([metrics_baseline, metrics_rf])
    metrics_csv_path = dirs["outputs"] / "metrics.csv"
    metrics_summary.to_csv(metrics_csv_path, index=False)
    logger.info("Saved comparison metrics to '%s'.", metrics_csv_path)

    print("\n" + "-" * 65)
    print("                MODEL EVALUATION COMPARISON")
    print("-" * 65)
    print(
        f"{'Model Name':<32} | {'MAE (' + ENERGY_UNIT + ')':<10} | {'RMSE (' + ENERGY_UNIT + ')':<10} | {'R² Score':<8}"
    )
    print("-" * 65)
    for _, row in metrics_summary.iterrows():
        print(
            f"{row['model_name']:<32} | {row['mae']:<10.3f} | {row['rmse']:<10.3f} | {row['r2_score']:<8.4f}"
        )
    print("-" * 65)

    # Relative improvement
    mae_diff = (metrics_baseline["mae"] - metrics_rf["mae"]) / metrics_baseline["mae"] * 100.0
    rmse_diff = (metrics_baseline["rmse"] - metrics_rf["rmse"]) / metrics_baseline["rmse"] * 100.0
    print(
        f"Random Forest Error Reduction vs Baseline: MAE by {mae_diff:.1f}% | RMSE by {rmse_diff:.1f}%\n"
    )

    # 9. Save test set predictions
    logger.info("STEP 7: Saving test set predictions...")
    predictions_df = pd.DataFrame(
        {
            "Datetime": test_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(test_df["Datetime"], "dt")
            else test_df["Datetime"],
            "Actual_Energy": y_test.to_numpy(),
            "Baseline_Prediction": np.round(y_pred_baseline, 2),
            "RF_Prediction": np.round(y_pred_rf, 2),
            "Residual": np.round(y_test.to_numpy() - y_pred_rf, 2),
        }
    )
    predictions_csv_path = dirs["outputs"] / "predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    logger.info("Saved predictions to '%s'.", predictions_csv_path)

    # 10. Persist model and feature names using Joblib
    logger.info("STEP 8: Persisting trained model and metadata using Joblib...")
    model_artifact_path = dirs["models"] / "random_forest_model.joblib"
    save_model_artifacts(
        model=rf_model,
        feature_names=feature_names,
        metrics_summary=metrics_summary,
        output_path=model_artifact_path,
        metadata={
            "is_synthetic": is_synthetic,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "energy_unit": ENERGY_UNIT,
        },
    )

    # 11. Generate Recursive 24-hour Forecast
    logger.info("STEP 9: Generating recursive 24-hour future forecast...")
    forecast_df = generate_recursive_24h_forecast(
        model=rf_model,
        historical_df=df_clean,
        feature_names=feature_names,
        forecast_horizon=24,
        unit=ENERGY_UNIT,
        is_synthetic=is_synthetic,
    )
    forecast_csv_path = dirs["outputs"] / "next_24_hours.csv"
    save_forecast(forecast_df, forecast_csv_path)

    # 12. Generate and save diagnostic visualizations in images/ and outputs/
    logger.info("STEP 10: Generating visualization plots...")
    for target_dir in [dirs["images"], dirs["outputs"]]:
        plot_actual_vs_predicted(
            predictions_df,
            target_dir / "actual_vs_predicted.png",
            zoom_hours=168,
            unit=ENERGY_UNIT,
        )
        plot_24h_forecast(
            df_clean,
            forecast_df,
            target_dir / "forecast_next_24_hours.png",
            history_hours=72,
            unit=ENERGY_UNIT,
        )
        plot_feature_importance(
            rf_model,
            feature_names,
            target_dir / "feature_importance.png",
            top_n=10,
        )
        plot_residuals_distribution(
            predictions_df,
            target_dir / "residuals_distribution.png",
            unit=ENERGY_UNIT,
        )

    feature_names_path = dirs["models"] / "feature_names.joblib"

    print("\n" + "=" * 70)
    print("   PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f" Artifacts generated:")
    print(f"  • Trained Model    : {model_artifact_path.relative_to(PROJECT_ROOT)}")
    print(f"  • Feature Names    : {feature_names_path.relative_to(PROJECT_ROOT)}")
    print(f"  • 24-Hour Forecast : {forecast_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  • Test Predictions : {predictions_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  • Evaluation Metrics: {metrics_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  • Diagnostic Charts: images/ and outputs/ (actual_vs_predicted, 24h_forecast, feature_importance, residuals)")
    print(f"\n Next Steps:")
    print(f"  • Run Flask API    : python app.py")
    print(f"  • Run Unit Tests   : pytest -v")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
