# Energy Units & Evaluation Metrics Reference

## 1. Measurement Units

### Default Unit: Megawatt-hours (MWh)
Energy measurements in electric grid operations, regional transmission organizations (RTOs/ISOs), and power utility forecasting are conventionally measured in **Megawatt-hours (MWh)** or Megawatts (MW) of average hourly demand:
- **1 Megawatt-hour (MWh)** = 1,000 Kilowatt-hours (kWh) = 1,000,000 Watt-hours (Wh).
- For regional grid operations, average loads typically fluctuate between 30 and 120 MWh per substation/feeder zone.
- For residential or building microgrid installations, the identical numerical pipeline applies with the unit interpreted as **Kilowatt-hours (kWh)**.

---

## 2. Evaluation Metrics

Model accuracy is evaluated on the held-out test split using three standard regression metrics:

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
- **Interpretation:** The expected average error magnitude in the original units (MWh).
- **Properties:** Linear penalty for all errors; robust to occasional outliers.

### 2. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
- **Interpretation:** The standard deviation of the unexplained residuals in original units (MWh).
- **Properties:** Quadratic penalty; disproportionately penalizes large forecast errors (critical for grid operators avoiding blackout risks or spinning reserve penalties).

### 3. Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
- **Interpretation:** The fraction of total energy consumption variance explained by the model compared to a naive constant-mean predictor.
- **Range:** $R^2 \le 1.0$. A value closer to 1.0 indicates superior predictive fit.

---

## 3. Synthetic Data Disclaimer
When using synthetic benchmark data:
- The data is procedurally generated with diurnal cycles, weekend dips, annual seasonality, and noise.
- Results demonstrate the correct execution and validity of the machine learning algorithms, data pipeline, and forecasting architecture.
- **No real-world energy cost savings or utility-grade precision should be claimed based on simulated benchmark runs.**
