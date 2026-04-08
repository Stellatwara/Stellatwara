"""
Hypothesis testing: Did ridership recover post-COVID differently across lines?

Tests performed:
    1. Kruskal-Wallis H-test across all four lines for 2023 ridership
       (non-parametric; does not assume normality).
    2. Pairwise Mann-Whitney U tests (Bonferroni-corrected) for each
       line pair to find which lines differ significantly.
    3. Effect-size calculation (rank-biserial correlation r).
    4. Recovery-ratio comparison: 2023 ridership / 2019 ridership by line.

Results are printed and saved to data/output/hypothesis_results.csv.
"""

import logging
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DB_PATH, DATA_OUTPUT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALPHA = 0.05


def _load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load daily ridership per line for 2019 and 2022-2024."""
    return con.execute("""
        SELECT
            d.date,
            d.year,
            f.line_id,
            l.line_name,
            SUM(f.ridership) AS daily_ridership,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        JOIN dim_lines l USING (line_id)
        WHERE d.year IN (2019, 2022, 2023, 2024)
          AND d.is_weekend = FALSE          -- weekdays only for fairness
        GROUP BY d.date, d.year, f.line_id, l.line_name
        ORDER BY d.date, f.line_id
    """).df()


def kruskal_wallis_test(df: pd.DataFrame, year: int = 2023) -> dict:
    """
    Test H0: the distribution of daily ridership is identical across all lines
    in *year*.  Uses Kruskal-Wallis H.
    """
    data_year = df[df["year"] == year]
    groups = [grp["daily_ridership"].values for _, grp in data_year.groupby("line_id")]
    h_stat, p_value = stats.kruskal(*groups)

    result = {
        "test": "Kruskal-Wallis H",
        "year": year,
        "H_statistic": round(h_stat, 4),
        "p_value": round(p_value, 6),
        "reject_H0": p_value < ALPHA,
        "conclusion": (
            f"Reject H0 at α={ALPHA}: ridership distributions differ across lines in {year}."
            if p_value < ALPHA else
            f"Fail to reject H0 at α={ALPHA}: no significant difference across lines in {year}."
        ),
    }
    logger.info("Kruskal-Wallis %s: H=%.4f, p=%.6f → %s",
                year, h_stat, p_value, "REJECT" if p_value < ALPHA else "fail to reject")
    return result


def pairwise_mannwhitney(df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests between every pair of lines in *year*.
    Applies Bonferroni correction for multiple comparisons.
    """
    data_year = df[df["year"] == year]
    lines = sorted(data_year["line_id"].unique())
    n_comparisons = len(list(combinations(lines, 2)))

    records = []
    for l1, l2 in combinations(lines, 2):
        x = data_year[data_year["line_id"] == l1]["daily_ridership"].values
        y = data_year[data_year["line_id"] == l2]["daily_ridership"].values
        u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
        p_bonf = min(p_val * n_comparisons, 1.0)

        # Rank-biserial correlation (effect size)
        n1, n2 = len(x), len(y)
        r = 1 - (2 * u_stat) / (n1 * n2)

        records.append({
            "year": year,
            "line_A": l1,
            "line_B": l2,
            "U_statistic": round(u_stat, 2),
            "p_value_raw": round(p_val, 6),
            "p_value_bonferroni": round(p_bonf, 6),
            "effect_size_r": round(r, 4),
            "significant": p_bonf < ALPHA,
        })

    return pd.DataFrame(records)


def recovery_ratio(df: pd.DataFrame, target_year: int = 2023) -> pd.DataFrame:
    """
    Compute ridership recovery ratio: mean(daily, target_year) / mean(daily, 2019).
    A value of 1.0 = full recovery; < 1.0 = still below pre-COVID.
    """
    base = (
        df[df["year"] == 2019]
        .groupby("line_id")["daily_ridership"]
        .mean()
        .rename("baseline_2019")
    )
    target = (
        df[df["year"] == target_year]
        .groupby("line_id")["daily_ridership"]
        .mean()
        .rename(f"avg_{target_year}")
    )
    ratio = pd.concat([base, target], axis=1)
    ratio["recovery_ratio"] = ratio[f"avg_{target_year}"] / ratio["baseline_2019"]
    ratio["recovery_pct"] = (ratio["recovery_ratio"] * 100).round(1)
    return ratio.reset_index()


def run_all_tests(con: duckdb.DuckDBPyConnection | None = None) -> dict:
    """Run all hypothesis tests and return results dict."""
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)

    df = _load_data(con)

    results = {}

    # 1. Kruskal-Wallis
    kw = kruskal_wallis_test(df, year=2023)
    results["kruskal_wallis"] = kw

    # 2. Pairwise Mann-Whitney
    mw_df = pairwise_mannwhitney(df, year=2023)
    results["mannwhitney_pairwise"] = mw_df

    # 3. Recovery ratios for multiple years
    ratios = []
    for yr in [2022, 2023, 2024]:
        r = recovery_ratio(df, target_year=yr)
        r["target_year"] = yr
        ratios.append(r)
    ratio_df = pd.concat(ratios, ignore_index=True)
    results["recovery_ratios"] = ratio_df

    # Save outputs
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([kw]).to_csv(DATA_OUTPUT / "hypothesis_kruskal.csv", index=False)
    mw_df.to_csv(DATA_OUTPUT / "hypothesis_mannwhitney.csv", index=False)
    ratio_df.to_csv(DATA_OUTPUT / "recovery_ratios.csv", index=False)

    # Print results
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 60)
    print(f"\n[1] Kruskal-Wallis (year=2023)")
    print(f"    H = {kw['H_statistic']},  p = {kw['p_value']}")
    print(f"    → {kw['conclusion']}")

    print(f"\n[2] Pairwise Mann-Whitney U (Bonferroni-corrected, year=2023)")
    print(mw_df[["line_A", "line_B", "p_value_bonferroni", "effect_size_r", "significant"]].to_string(index=False))

    print(f"\n[3] Recovery Ratios (weekday avg vs 2019 baseline)")
    print(ratio_df.pivot(index="line_id", columns="target_year", values="recovery_pct").to_string())
    print()

    return results


if __name__ == "__main__":
    run_all_tests()
