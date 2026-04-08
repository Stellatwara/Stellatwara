"""
Predictive model: forecast daily STM metro ridership.

Model: Gradient Boosted Trees (sklearn GradientBoostingRegressor) with
time-series cross-validation.

Features:
    - line_id (encoded)
    - year, month, day_of_week, week_of_year
    - is_weekend, is_peak_season (Sept-Nov)
    - covid_era (Pre/COVID/Post)
    - lag_7  (ridership 7 days prior, same line)
    - lag_28 (ridership 28 days prior, same line)
    - rolling_28_mean (28-day rolling average, same line)

Outputs:
    - data/output/model_predictions.csv  (test-set predictions vs actuals)
    - data/output/feature_importance.csv
    - data/output/plots/model_performance.png
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DB_PATH, DATA_OUTPUT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PLOTS_DIR = DATA_OUTPUT / "plots"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load daily ridership and engineer features for the predictive model.
    """
    df = con.execute("""
        SELECT
            d.date,
            f.line_id,
            d.year,
            d.month,
            d.week,
            d.day_of_week,
            d.is_weekend,
            d.season,
            d.covid_era,
            SUM(f.ridership) AS daily_ridership,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        GROUP BY d.date, f.line_id, d.year, d.month, d.week,
                 d.day_of_week, d.is_weekend, d.season, d.covid_era
        ORDER BY f.line_id, d.date
    """).df()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["line_id", "date"]).reset_index(drop=True)

    # Lag features per line
    for lag in [7, 28]:
        df[f"lag_{lag}"] = (
            df.groupby("line_id")["daily_ridership"]
            .shift(lag)
        )

    # Rolling mean
    df["rolling_28_mean"] = (
        df.groupby("line_id")["daily_ridership"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=7).mean())
    )

    # Boolean flags
    df["is_peak_season"] = df["month"].isin([9, 10, 11]).astype(int)
    df["is_weekend_int"] = df["is_weekend"].astype(int)

    # Encode categoricals
    le_line = LabelEncoder()
    df["line_id_enc"] = le_line.fit_transform(df["line_id"])

    le_covid = LabelEncoder()
    df["covid_era_enc"] = le_covid.fit_transform(df["covid_era"])

    # Drop rows with NaN lags
    df = df.dropna(subset=["lag_7", "lag_28", "rolling_28_mean"])

    return df


FEATURE_COLS = [
    "line_id_enc", "year", "month", "week", "day_of_week",
    "is_weekend_int", "is_peak_season", "covid_era_enc",
    "lag_7", "lag_28", "rolling_28_mean",
]
TARGET = "daily_ridership"


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Train GBT model with time-series cross-validation.

    Returns dict with model, metrics, predictions, and feature importances.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    dates = df["date"].values

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    # Train on all but last fold; evaluate on last fold
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_pct": mape}
    logger.info("Model metrics: MAE=%.1f  RMSE=%.1f  R²=%.3f  MAPE=%.1f%%",
                mae, rmse, r2, mape)

    # Predictions DataFrame
    pred_df = df.iloc[test_idx][["date", "line_id", "daily_ridership"]].copy()
    pred_df["predicted"] = y_pred
    pred_df["residual"] = pred_df["daily_ridership"] - pred_df["predicted"]

    # Feature importance
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "predictions": pred_df,
        "feature_importance": fi_df,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_performance(results: dict, save: bool = True) -> plt.Figure:
    """Two-panel figure: actual vs predicted + feature importance."""
    pred_df = results["predictions"]
    fi_df = results["feature_importance"]
    metrics = results["metrics"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: actual vs predicted (sample first line)
    sample_line = pred_df["line_id"].iloc[0]
    sub = pred_df[pred_df["line_id"] == sample_line].sort_values("date")
    ax1.plot(sub["date"], sub["daily_ridership"], label="Actual", linewidth=1.5)
    ax1.plot(sub["date"], sub["predicted"], label="Predicted", linewidth=1.5, linestyle="--")
    ax1.set_title(f"Actual vs Predicted (Line {sample_line})", fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Ridership")
    ax1.legend()
    ax1.text(
        0.02, 0.95,
        f"MAE={metrics['MAE']:.0f}  RMSE={metrics['RMSE']:.0f}  R²={metrics['R2']:.3f}",
        transform=ax1.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    # Panel 2: feature importance
    ax2.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
    ax2.set_title("Feature Importance", fontweight="bold")
    ax2.set_xlabel("Importance Score")

    fig.suptitle("STM Daily Ridership – Gradient Boosted Model", fontweight="bold")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "model_performance.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(con: duckdb.DuckDBPyConnection | None = None) -> dict:
    """Run full predictive modelling pipeline."""
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)

    logger.info("Building features …")
    df = build_features(con)
    logger.info("Training model on %d rows …", len(df))
    results = train_and_evaluate(df)

    # Save outputs
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    results["predictions"].to_csv(DATA_OUTPUT / "model_predictions.csv", index=False)
    results["feature_importance"].to_csv(DATA_OUTPUT / "feature_importance.csv", index=False)

    plot_model_performance(results)

    print("\n=== Model Evaluation Metrics ===")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")

    return results


if __name__ == "__main__":
    run()
