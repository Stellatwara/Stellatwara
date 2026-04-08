"""
Exploratory Data Analysis: ridership trends by line, peak vs off-peak,
seasonal variation.

Produces PNG plots saved to data/output/plots/ and prints summary statistics.
"""

import logging
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_OUTPUT, DB_PATH, METRO_LINES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PLOTS_DIR = DATA_OUTPUT / "plots"

LINE_PALETTE = {
    "1": "#2ca02c",   # green
    "2": "#ff7f0e",   # orange
    "4": "#d4b400",   # yellow/gold
    "5": "#1f77b4",   # blue
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


# ---------------------------------------------------------------------------
# 1. Monthly ridership trends by line
# ---------------------------------------------------------------------------

def plot_monthly_trends(save: bool = True) -> plt.Figure:
    """Line chart of monthly ridership per line (2018-2024)."""
    con = _con()
    df = con.execute("""
        SELECT year, month, line_id, line_name,
               total_ridership / 1_000_000 AS ridership_M
        FROM agg_monthly_line
        ORDER BY year, month, line_id
    """).df()
    con.close()

    df["period"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for lid, grp in df.groupby("line_id"):
        label = f"Line {lid} – {METRO_LINES.get(lid, '')}"
        ax.plot(grp["period"], grp["ridership_M"],
                color=LINE_PALETTE.get(lid, "grey"),
                label=label, linewidth=2)

    # COVID shading
    ax.axvspan(pd.Timestamp("2020-03-13"), pd.Timestamp("2021-12-31"),
               alpha=0.15, color="red", label="COVID restrictions")

    ax.set_title("Monthly STM Metro Ridership by Line (2018-2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ridership (millions of trips)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "monthly_trends.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ---------------------------------------------------------------------------
# 2. Peak vs off-peak ridership by line
# ---------------------------------------------------------------------------

def plot_peak_vs_offpeak(save: bool = True) -> plt.Figure:
    """Grouped bar chart of annual peak vs off-peak ridership per line."""
    con = _con()
    df = con.execute("""
        SELECT
            year,
            line_id,
            line_name,
            is_peak,
            SUM(ridership) / 1_000_000 AS ridership_M
        FROM agg_peak_summary
        GROUP BY year, line_id, line_name, is_peak
        ORDER BY year, line_id, is_peak
    """).df()
    con.close()

    df["peak_label"] = df["is_peak"].map({True: "Peak", False: "Off-peak"})
    pivot = df.pivot_table(
        index=["year", "line_id"], columns="peak_label", values="ridership_M"
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    axes = axes.flatten()

    for i, (lid, grp) in enumerate(pivot.groupby("line_id")):
        ax = axes[i]
        x = grp["year"]
        w = 0.35
        ax.bar(x - w / 2, grp["Peak"], width=w,
               color=LINE_PALETTE.get(str(lid), "grey"), alpha=0.85, label="Peak")
        ax.bar(x + w / 2, grp["Off-peak"], width=w,
               color=LINE_PALETTE.get(str(lid), "grey"), alpha=0.45, label="Off-peak", hatch="//")
        ax.set_title(f"Line {lid} – {METRO_LINES.get(str(lid), '')}", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Ridership (M)")
        ax.legend()
        ax.set_xticks(x)

    fig.suptitle("Peak vs Off-Peak Ridership by Line (Annual)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "peak_vs_offpeak.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ---------------------------------------------------------------------------
# 3. Seasonal variation heat-map
# ---------------------------------------------------------------------------

def plot_seasonal_heatmap(save: bool = True) -> plt.Figure:
    """Heat-map of ridership by (season × year) for each line."""
    con = _con()
    df = con.execute("""
        SELECT year, season, line_id, line_name, avg_daily_ridership
        FROM agg_seasonal
        ORDER BY year, season, line_id
    """).df()
    con.close()

    season_order = ["Spring", "Summer", "Fall", "Winter"]
    df["season"] = pd.Categorical(df["season"], categories=season_order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, (lid, grp) in enumerate(df.groupby("line_id")):
        pivot = grp.pivot(index="season", columns="year", values="avg_daily_ridership")
        pivot.index.name = None
        pivot.columns.name = None
        ax = axes[i]
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", fmt=".0f", annot=True,
                    linewidths=0.5, cbar_kws={"label": "Avg daily ridership"})
        ax.set_title(f"Line {lid} – {METRO_LINES.get(str(lid), '')}", fontweight="bold")

    fig.suptitle("Seasonal Ridership Variation by Line and Year", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "seasonal_heatmap.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ---------------------------------------------------------------------------
# 4. COVID recovery comparison
# ---------------------------------------------------------------------------

def plot_covid_recovery(save: bool = True) -> plt.Figure:
    """
    Index ridership to pre-COVID baseline (2019 = 100) and show recovery
    trajectory per line.
    """
    con = _con()
    df = con.execute("""
        SELECT year, month, line_id, line_name, total_ridership
        FROM agg_monthly_line
        WHERE year BETWEEN 2018 AND 2024
        ORDER BY year, month, line_id
    """).df()
    con.close()

    # Compute 2019 monthly baseline per line
    baseline = (
        df[df["year"] == 2019]
        .groupby(["line_id", "month"])["total_ridership"]
        .mean()
        .rename("baseline")
    )
    df = df.merge(baseline, on=["line_id", "month"])
    df["index"] = df["total_ridership"] / df["baseline"] * 100
    df["period"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for lid, grp in df.groupby("line_id"):
        ax.plot(grp["period"], grp["index"],
                color=LINE_PALETTE.get(str(lid), "grey"),
                label=f"Line {lid} – {METRO_LINES.get(str(lid), '')}",
                linewidth=2)

    ax.axhline(100, color="grey", linestyle="--", linewidth=1, label="2019 baseline")
    ax.axvspan(pd.Timestamp("2020-03-13"), pd.Timestamp("2021-12-31"),
               alpha=0.15, color="red", label="COVID restrictions")
    ax.set_title("Ridership Recovery Index (2019 = 100)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Index (2019 = 100)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "covid_recovery.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ---------------------------------------------------------------------------
# 5. Top/bottom stations
# ---------------------------------------------------------------------------

def plot_station_bar(save: bool = True) -> plt.Figure:
    """Horizontal bar chart of top-20 stations by total ridership."""
    con = _con()
    df = con.execute("""
        SELECT station_name, line_id, total_ridership / 1_000_000 AS ridership_M
        FROM agg_station_rank
        LIMIT 20
    """).df()
    con.close()

    df = df.sort_values("ridership_M")
    colors = [LINE_PALETTE.get(str(lid), "grey") for lid in df["line_id"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["station_name"], df["ridership_M"], color=colors)
    ax.set_xlabel("Total Ridership 2018-2024 (millions)")
    ax.set_title("Top 20 STM Metro Stations by Total Ridership", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOTS_DIR / "top_stations.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved %s", path)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all(save: bool = True) -> dict[str, plt.Figure]:
    """Run all EDA plots."""
    return {
        "monthly_trends": plot_monthly_trends(save),
        "peak_vs_offpeak": plot_peak_vs_offpeak(save),
        "seasonal_heatmap": plot_seasonal_heatmap(save),
        "covid_recovery": plot_covid_recovery(save),
        "top_stations": plot_station_bar(save),
    }


if __name__ == "__main__":
    run_all()
    print("EDA complete. Plots saved to", PLOTS_DIR)
