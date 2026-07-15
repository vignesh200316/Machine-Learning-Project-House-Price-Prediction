"""
Generates a synthetic but realistic housing dataset for the price
prediction project. No internet / external dataset download required.

Features are built with real relationships (bigger, newer houses closer
to the city center and in better neighborhoods cost more) plus noise,
so the downstream ML pipeline has genuine signal to learn from.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1400


def generate_housing_data(n_samples: int = N) -> pd.DataFrame:
    square_footage = RNG.normal(1800, 650, n_samples).clip(450, 6000)
    bedrooms = RNG.integers(1, 6, n_samples)
    bathrooms = np.clip(np.round(bedrooms * RNG.uniform(0.5, 1.1, n_samples)), 1, 5)
    house_age = RNG.integers(0, 80, n_samples)
    lot_size = RNG.normal(6500, 2500, n_samples).clip(1000, 20000)
    garage_spaces = RNG.integers(0, 4, n_samples)
    distance_to_city_km = RNG.exponential(8, n_samples).clip(0.5, 60)
    neighborhood_quality = RNG.integers(1, 11, n_samples)  # 1 (low) - 10 (high)
    has_renovation = RNG.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # A few missing values / raw messiness to make preprocessing meaningful
    lot_size[RNG.choice(n_samples, size=int(0.02 * n_samples), replace=False)] = np.nan

    # Ground-truth price formula (non-linear-ish + noise) that the model must recover

    price = (
        square_footage * 118
        + bedrooms * 4200
        + bathrooms * 6100
        + garage_spaces * 5200
        + np.nan_to_num(lot_size, nan=6500.0) * 2.1
        + neighborhood_quality * 9800
        + has_renovation * 15000
        - house_age * 950
        - distance_to_city_km * 1800
        + RNG.normal(0, 18000, n_samples)  # market noise
    )
    price = price.clip(35000, None)

    # Inject a handful of genuine outliers (e.g. mansion, distressed sale)
    outlier_idx = RNG.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    price[outlier_idx] *= RNG.choice([0.35, 2.8], size=len(outlier_idx))

    df = pd.DataFrame({
        "square_footage": square_footage.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_age_years": house_age,
        "lot_size_sqft": lot_size.round(0),
        "garage_spaces": garage_spaces,
        "distance_to_city_km": distance_to_city_km.round(2),
        "neighborhood_quality": neighborhood_quality,
        "has_renovation": has_renovation,
        "price": price.round(0),
    })
    return df


if __name__ == "__main__":
    data = generate_housing_data()
    data.to_csv("housing_data.csv", index=False)
    print(f"Generated {len(data)} rows -> housing_data.csv")
    print(data.head())
