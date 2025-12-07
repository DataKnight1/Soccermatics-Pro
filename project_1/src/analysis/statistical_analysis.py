"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Statistical analysis functions for hypothesis testing and significance
             analysis of player performance metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import SIGNIFICANCE_LEVELS, METRICS_FOR_HYPOTHESIS_TESTING


class StatisticalAnalyzer:
    """Perform statistical analyses on player metrics."""

    def __init__(self, metrics_df: pd.DataFrame) -> None:
        """
        Initialize analyzer with metrics data.

        Parameters
        ----------
        metrics_df : DataFrame
            Contains all players' metrics.
        """
        self.metrics_df = metrics_df.copy()

    def one_sample_z_test(
        self,
        player_value: float,
        population: pd.Series,
        alternative: str = "greater",
    ) -> Tuple[float, float]:
        """
        Perform one-sample z-test against a population.

        Assumptions:
        - Population size is reasonably large (n >= 30 recommended).
        - Population distribution is approximately normal.

        Parameters
        ----------
        player_value : float
            Player's value for the metric.
        population : Series
            Population values (excluding player).
        alternative : {'greater', 'less', 'two-sided'}

        Returns
        -------
        z_score, p_value : (float, float)
            NaNs if the test is not well-defined (e.g. n < 2 or std == 0).
        """
        population = population.dropna()
        n = len(population)
        if n < 2:
            return np.nan, np.nan

        pop_mean = population.mean()
        pop_std = population.std(ddof=1)

        if pop_std == 0:
            return np.nan, np.nan

        z_score = (player_value - pop_mean) / (pop_std / np.sqrt(n))

        if alternative == "greater":
            p_value = 1.0 - stats.norm.cdf(z_score)
        elif alternative == "less":
            p_value = stats.norm.cdf(z_score)
        else:
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

        return float(z_score), float(p_value)

    def test_player_vs_population(
        self,
        player_name: str,
        metrics: List[str] | None = None,
        alternative: str = "greater",
    ) -> pd.DataFrame:
        """
        Test if player's metrics are significantly different from population.

        Parameters
        ----------
        player_name : str
            Player to test.
        metrics : list of str or None
            Metrics to test (default: METRICS_FOR_HYPOTHESIS_TESTING).
        alternative : {'greater', 'less', 'two-sided'}

        Returns
        -------
        DataFrame
            Columns: metric, player_value, population_mean, population_std,
                     z_score, p_value, significance
        """
        if metrics is None:
            metrics = METRICS_FOR_HYPOTHESIS_TESTING

        player_row = self.metrics_df[self.metrics_df["player"] == player_name]
        if len(player_row) == 0:
            raise ValueError(f"Player {player_name} not found")

        player_row = player_row.iloc[0]

        results: List[Dict] = []

        for metric in metrics:
            if metric not in self.metrics_df.columns:
                continue

            player_value = player_row[metric]
            population = self.metrics_df[self.metrics_df["player"] != player_name][metric]

            z_score, p_value = self.one_sample_z_test(
                float(player_value), population, alternative=alternative
            )

            pop_mean = float(population.mean()) if not population.empty else np.nan
            pop_std = float(population.std(ddof=1)) if not population.empty else np.nan

            if np.isnan(p_value):
                significance = "n/a"
            elif p_value < SIGNIFICANCE_LEVELS["highly_significant"]:
                significance = "***"
            elif p_value < SIGNIFICANCE_LEVELS["significant"]:
                significance = "**"
            elif p_value < SIGNIFICANCE_LEVELS["marginally_significant"]:
                significance = "*"
            else:
                significance = "ns"

            results.append(
                {
                    "metric": metric,
                    "player_value": float(player_value),
                    "population_mean": pop_mean,
                    "population_std": pop_std,
                    "z_score": z_score,
                    "p_value": p_value,
                    "significance": significance,
                }
            )

        return pd.DataFrame(results)

    def calculate_effect_size(
        self,
        player_value: float,
        population: pd.Series,
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Returns NaN if std == 0 or population empty.
        """
        population = population.dropna()
        if population.empty:
            return float("nan")

        pop_mean = population.mean()
        pop_std = population.std(ddof=1)
        if pop_std == 0:
            return float("nan")

        return float((player_value - pop_mean) / pop_std)

    def get_percentile_rank(
        self,
        player_name: str,
        metric: str,
    ) -> float:
        """
        Get percentile rank for a specific metric.

        Percentile definition: percentage of players strictly below the player's value.
        """
        if metric not in self.metrics_df.columns:
            raise ValueError(f"Metric {metric} not found in metrics_df")

        player_row = self.metrics_df[self.metrics_df["player"] == player_name]
        if len(player_row) == 0:
            raise ValueError(f"Player {player_name} not found")

        player_value = player_row.iloc[0][metric]
        series = self.metrics_df[metric].dropna()
        if series.empty:
            return float("nan")

        percentile = (series < player_value).sum() / len(series) * 100.0
        return float(percentile)

    def get_top_players(
        self,
        metric: str,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Get top N players for a metric.

        Returns only players with non-null metric values.
        """
        if metric not in self.metrics_df.columns:
            raise ValueError(f"Metric {metric} not found in metrics_df")

        df = self.metrics_df.dropna(subset=[metric])
        return df.nlargest(n, metric)[["player", "team", metric, "total_minutes"]]

    def compare_players(
        self,
        player1: str,
        player2: str,
        metrics: List[str],
    ) -> pd.DataFrame:
        """
        Compare two players across metrics.

        Returns a DataFrame with absolute and percentage differences.
        """
        p1_row = self.metrics_df[self.metrics_df["player"] == player1]
        p2_row = self.metrics_df[self.metrics_df["player"] == player2]

        if len(p1_row) == 0 or len(p2_row) == 0:
            raise ValueError("One or both players not found")

        p1_row = p1_row.iloc[0]
        p2_row = p2_row.iloc[0]

        comparisons: List[Dict] = []

        for metric in metrics:
            if metric not in self.metrics_df.columns:
                continue

            v1 = p1_row[metric]
            v2 = p2_row[metric]

            if v2 != 0:
                pct_diff = (v1 - v2) / v2 * 100.0
            else:
                pct_diff = np.nan

            comparisons.append(
                {
                    "metric": metric,
                    player1: float(v1),
                    player2: float(v2),
                    "difference": float(v1 - v2),
                    "pct_difference": float(pct_diff) if not np.isnan(pct_diff) else np.nan,
                }
            )

        return pd.DataFrame(comparisons)
