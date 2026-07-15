"""
House Price Prediction — Linear Regression with Feature Engineering
=====================================================================

Pipeline:
  1. Load raw data (generate_data.py creates it if missing)
  2. EDA summary
  3. Data cleaning: impute missing lot_size, remove statistical outliers (IQR)
  4. Feature engineering: price_per_sqft-adjacent features, age buckets,
     interaction term, one-hot encoding of categorical bucket
  5. Train/test split + StandardScaler
  6. Fit Linear Regression
  7. Evaluate with RMSE and R², plus residual analysis plot

Run:  python house_price_prediction.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from generate_data import generate_housing_data

DATA_PATH = "housing_data.csv"
OUT_DIR = "outputs"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        generate_housing_data().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Impute missing lot size with median
    df["lot_size_sqft"] = df["lot_size_sqft"].fillna(df["lot_size_sqft"].median())

    # Remove outliers in price using the IQR rule
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df["price"] >= lower) & (df["price"] <= upper)].reset_index(drop=True)
    print(f"Outlier removal: dropped {before - len(df)} rows ({(before - len(df)) / before:.1%})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_bucket"] = pd.cut(
        df["house_age_years"], bins=[-1, 5, 20, 50, 200],
        labels=["new", "modern", "established", "old"]
    )
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]
    df["sqft_per_room"] = df["square_footage"] / df["rooms_total"].replace(0, 1)
    df["lot_to_house_ratio"] = df["lot_size_sqft"] / df["square_footage"]
    df = pd.get_dummies(df, columns=["age_bucket"], drop_first=True)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print("\n--- EDA summary ---")
    print(df.describe().T[["mean", "std", "min", "max"]])
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

    df = clean_data(df)
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != "price"]
    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model performance (test set) ---")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²:   {r2:.4f}")

    # Feature importance via absolute standardized coefficients
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\nTop feature coefficients (standardized):")
    print(coef_df.head(8).to_string(index=False))

    # Residual analysis plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs. Predicted")

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residual_analysis.png"), dpi=130)
    print(f"\nSaved residual analysis plot to {OUT_DIR}/residual_analysis.png")

    # Actual vs predicted scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, s=15)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs. Predicted Price (R² = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "actual_vs_predicted.png"), dpi=130)
    print(f"Saved actual-vs-predicted plot to {OUT_DIR}/actual_vs_predicted.png")


if __name__ == "__main__":
    main()
