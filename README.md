# House Price Prediction — Linear Regression

End-to-end regression pipeline: synthetic housing data → EDA → cleaning →
feature engineering → Linear Regression → evaluation with RMSE / R² and
residual analysis.

## What this demonstrates
- Data cleaning (missing-value imputation, IQR-based outlier removal)
- Feature engineering (age buckets, rooms-per-sqft, lot-to-house ratio, one-hot encoding)
- Train/test split with `StandardScaler`
- Model evaluation: RMSE, R², standardized coefficient importance
- Residual analysis (residuals vs. predicted, residual distribution, actual vs. predicted)

## Results (this run)
- **RMSE:** ≈ $26.7k
- **R²:** ≈ 0.915

## Run it
```bash
pip install -r requirements.txt
python house_price_prediction.py
```
Outputs are written to `outputs/` (`residual_analysis.png`, `actual_vs_predicted.png`).

## Files
- `generate_data.py` — builds a synthetic-but-realistic housing dataset (no external download needed)
- `house_price_prediction.py` — full modeling pipeline
- `housing_data.csv` — generated dataset (created on first run)
