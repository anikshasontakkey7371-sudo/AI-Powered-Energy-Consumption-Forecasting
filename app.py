"""
AI-Powered Energy Consumption Forecasting System
Flask REST API and Interactive Web Dashboard.

Provides endpoints to:
1. View model status and 24-hour forecast dashboard (GET /).
2. Health check and model metadata (GET /health).
3. Retrieve latest 24-hour ahead forecast (GET /api/forecast).
4. Retrieve model vs baseline metrics (GET /api/metrics).
5. On-demand single-step prediction (POST /api/predict).
"""

from pathlib import Path
import json
import logging
from typing import Optional
from flask import Flask, jsonify, render_template_string, request
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flask_app")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = Flask(__name__)

from src.model import load_model_artifacts

# Global model state cache
_MODEL_BUNDLE: Optional[dict] = None


def get_model_bundle() -> Optional[dict]:
    """Load model artifact from disk with memoized caching."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        model_path = MODELS_DIR / "random_forest_model.joblib"
        if model_path.exists():
            try:
                _MODEL_BUNDLE = load_model_artifacts(model_path)
                logger.info("Loaded model bundle into application cache.")
            except Exception as e:
                logger.error("Failed to load model artifact: %s", e)
    return _MODEL_BUNDLE


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI-Powered Energy Consumption Forecasting System</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.6;
      padding: 2rem 1.5rem;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent);
      border: 1px solid rgba(16, 185, 129, 0.3);
      margin-bottom: 0.75rem;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 1.05rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card:hover {
      border-color: #475569;
      transform: translateY(-2px);
    }
    .card-title {
      font-size: 0.85rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.4rem;
    }
    .card-value {
      font-size: 1.75rem;
      font-weight: 700;
      color: #ffffff;
    }
    .card-subtext {
      font-size: 0.8rem;
      color: var(--accent);
      margin-top: 0.25rem;
    }
    .section-title {
      font-size: 1.35rem;
      font-weight: 600;
      margin: 2rem 0 1rem 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .table-container {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow-x: auto;
      margin-bottom: 2rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.95rem;
    }
    th, td {
      padding: 0.85rem 1.2rem;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: rgba(15, 23, 42, 0.6);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }
    .code-box {
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1rem;
      font-family: monospace;
      font-size: 0.85rem;
      color: #38bdf8;
      overflow-x: auto;
      margin-bottom: 1rem;
    }
    .api-badge {
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 0.25rem;
      font-weight: bold;
      font-size: 0.75rem;
      margin-right: 0.5rem;
    }
    .get { background: #0284c7; color: white; }
    .post { background: #059669; color: white; }
    footer {
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.85rem;
      text-align: center;
    }
    .disclaimer {
      margin-top: 0.5rem;
      font-size: 0.8rem;
      color: #64748b;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge">● System Online</span>
      <h1>AI-Powered Energy Consumption Forecasting System</h1>
      <p class="subtitle">Machine Learning Time-Series Service with Random Forest Regressor & Naive Persistence Baseline</p>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Forecast Horizon</div>
        <div class="card-value">24 Hours</div>
        <div class="card-subtext">Autoregressive Recursive Steps</div>
      </div>
      <div class="card">
        <div class="card-title">Primary Model</div>
        <div class="card-value">Random Forest</div>
        <div class="card-subtext">100 Trees | max_depth: 15</div>
      </div>
      <div class="card">
        <div class="card-title">Test R² Score</div>
        <div class="card-value">{{ rf_r2 }}</div>
        <div class="card-subtext">Baseline: {{ baseline_r2 }}</div>
      </div>
      <div class="card">
        <div class="card-title">Test MAE (Mean Abs Error)</div>
        <div class="card-value">{{ rf_mae }} <span style="font-size:1rem;color:var(--text-muted);">{{ unit }}</span></div>
        <div class="card-subtext">Baseline: {{ baseline_mae }} {{ unit }}</div>
      </div>
    </div>

    <h2 class="section-title">24-Hour Ahead Hourly Forecast</h2>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Timestamp (Datetime)</th>
            <th>Predicted Energy ({{ unit }})</th>
            <th>Model Architecture</th>
            <th>Provenance</th>
          </tr>
        </thead>
        <tbody>
          {% for row in forecast_rows %}
          <tr>
            <td><strong>{{ row.Datetime }}</strong></td>
            <td style="color:#60a5fa; font-weight:600;">{{ row.Forecast_Energy }} {{ row.Unit }}</td>
            <td>{{ row.Model }}</td>
            <td><span style="color:#94a3b8; font-size:0.85rem;">{{ row.Data_Source }}</span></td>
          </tr>
          {% endfor %}
          {% if not forecast_rows %}
          <tr>
            <td colspan="4" style="text-align:center; color:var(--text-muted); padding:2rem;">
              No forecast found. Please execute <code>python main.py</code> to train model and generate forecast.
            </td>
          </tr>
          {% endif %}
        </tbody>
      </table>
    </div>

    <h2 class="section-title">API Endpoints (REST)</h2>
    <div class="card" style="margin-bottom: 2rem;">
      <p style="margin-bottom:0.75rem;"><span class="api-badge get">GET</span> <code>/health</code> — System and model health check</p>
      <p style="margin-bottom:0.75rem;"><span class="api-badge get">GET</span> <code>/api/forecast</code> — Fetch latest 24-hour forecast as JSON</p>
      <p style="margin-bottom:0.75rem;"><span class="api-badge get">GET</span> <code>/api/metrics</code> — Fetch model vs baseline comparative metrics</p>
      <p style="margin-bottom:0.5rem;"><span class="api-badge post">POST</span> <code>/api/predict</code> — On-demand single step prediction</p>
      <div class="code-box">curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"hour": 14, "dayofweek": 2, "month": 6, "is_weekend": 0, "lag_1": 55.2, "lag_2": 54.1, "lag_3": 52.8, "lag_24": 53.9, "lag_48": 54.0, "lag_168": 56.1, "rolling_mean_24": 51.5, "rolling_std_24": 4.2, "rolling_mean_168": 50.8}}'</div>
    </div>

    <footer>
      <p>AI-Powered Energy Consumption Forecasting Portfolio Project</p>
      <p class="disclaimer">Disclaimer: Simulated/Synthetic demonstration data is used for algorithmic benchmarking and educational evaluation. Energy values are expressed in Megawatt-hours (MWh).</p>
    </footer>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Render interactive HTML dashboard."""
    forecast_csv = OUTPUTS_DIR / "next_24_hours.csv"
    metrics_csv = OUTPUTS_DIR / "metrics.csv"

    forecast_rows = []
    if forecast_csv.exists():
        try:
            df_fore = pd.read_csv(forecast_csv)
            forecast_rows = df_fore.to_dict(orient="records")
        except Exception as e:
            logger.error("Error reading forecast CSV: %s", e)

    rf_r2 = "N/A"
    rf_mae = "N/A"
    baseline_r2 = "N/A"
    baseline_mae = "N/A"
    unit = "MWh"

    if metrics_csv.exists():
        try:
            df_m = pd.read_csv(metrics_csv)
            for _, row in df_m.iterrows():
                if "RandomForest" in str(row["model_name"]):
                    rf_r2 = str(row["r2_score"])
                    rf_mae = str(row["mae"])
                    unit = str(row.get("unit", "MWh"))
                else:
                    baseline_r2 = str(row["r2_score"])
                    baseline_mae = str(row["mae"])
        except Exception as e:
            logger.error("Error reading metrics CSV: %s", e)

    return render_template_string(
        DASHBOARD_HTML,
        forecast_rows=forecast_rows,
        rf_r2=rf_r2,
        rf_mae=rf_mae,
        baseline_r2=baseline_r2,
        baseline_mae=baseline_mae,
        unit=unit,
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint returning system status and metadata."""
    bundle = get_model_bundle()
    forecast_exists = (OUTPUTS_DIR / "next_24_hours.csv").exists()

    status = {
        "status": "healthy",
        "service": "AI-Powered Energy Consumption Forecasting System",
        "model_loaded": bundle is not None,
        "model_type": bundle["metadata"]["model_type"] if bundle else None,
        "forecast_available": forecast_exists,
        "unit": "MWh",
        "data_source_mode": "Synthetic Demonstration (Simulated)"
        if bundle and bundle.get("metadata", {}).get("is_synthetic", True)
        else "Observed",
    }
    return jsonify(status), 200


@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    """Serve the 24-hour recursive forecast from CSV."""
    forecast_csv = OUTPUTS_DIR / "next_24_hours.csv"
    if not forecast_csv.exists():
        return (
            jsonify(
                {
                    "error": "Forecast not found",
                    "message": "Please run 'python main.py' to generate the initial 24-hour forecast.",
                }
            ),
            404,
        )

    try:
        df_fore = pd.read_csv(forecast_csv)
        return (
            jsonify(
                {
                    "forecast_horizon_hours": len(df_fore),
                    "unit": "MWh",
                    "records": df_fore.to_dict(orient="records"),
                }
            ),
            200,
        )
    except Exception as e:
        logger.error("Error loading forecast: %s", e)
        return jsonify({"error": "Failed to load forecast", "details": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Serve model evaluation metrics."""
    metrics_csv = OUTPUTS_DIR / "metrics.csv"
    if not metrics_csv.exists():
        return (
            jsonify(
                {
                    "error": "Metrics not found",
                    "message": "Please run 'python main.py' to evaluate models.",
                }
            ),
            404,
        )

    try:
        df_m = pd.read_csv(metrics_csv)
        return jsonify({"metrics": df_m.to_dict(orient="records")}), 200
    except Exception as e:
        logger.error("Error reading metrics: %s", e)
        return jsonify({"error": "Failed to read metrics", "details": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    On-demand single-step energy prediction.

    Expected JSON schema:
    {
        "features": {
            "hour": int [0-23],
            "dayofweek": int [0-6],
            "month": int [1-12],
            "is_weekend": int [0 or 1],
            "lag_1": float,
            "lag_2": float,
            "lag_3": float,
            "lag_24": float,
            "lag_48": float,
            "lag_168": float,
            "rolling_mean_24": float,
            "rolling_std_24": float,
            "rolling_mean_168": float
        }
    }
    """
    bundle = get_model_bundle()
    if bundle is None:
        return (
            jsonify(
                {
                    "error": "Model not available",
                    "message": "Please train model via 'python main.py' first.",
                }
            ),
            503,
        )

    if not request.is_json:
        return jsonify({"error": "Invalid request", "message": "Payload must be JSON."}), 400

    payload = request.get_json()
    if not payload or "features" not in payload:
        return (
            jsonify(
                {
                    "error": "Missing 'features' key",
                    "required_features": bundle["feature_names"],
                }
            ),
            400,
        )

    feature_data = payload["features"]
    required_features = bundle["feature_names"]

    missing = [f for f in required_features if f not in feature_data]
    if missing:
        return (
            jsonify(
                {
                    "error": f"Missing required feature(s): {missing}",
                    "expected_features": required_features,
                }
            ),
            400,
        )

    try:
        # Convert and validate features as numeric floats
        input_df = pd.DataFrame([feature_data])[required_features].astype(float)
        if input_df.isna().any().any():
            return (
                jsonify(
                    {
                        "error": "Invalid feature value",
                        "message": "Feature values must be valid finite numbers and cannot contain null or NaN.",
                    }
                ),
                400,
            )
    except (ValueError, TypeError) as err:
        return (
            jsonify(
                {
                    "error": "Invalid feature format",
                    "message": f"All feature values must be numeric: {err}",
                }
            ),
            400,
        )

    try:
        model = bundle["model"]
        prediction = float(model.predict(input_df)[0])
        prediction = max(0.0, round(prediction, 2))

        return (
            jsonify(
                {
                    "prediction": prediction,
                    "unit": "MWh",
                    "model": type(model).__name__,
                }
            ),
            200,
        )
    except Exception as e:
        logger.error("Error making prediction: %s", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found", "message": "Endpoint does not exist"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500


if __name__ == "__main__":
    print("\nStarting Flask API Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
