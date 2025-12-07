"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Comprehensive metrics calculator for player performance analysis including
             progression, creation, and defensive metrics normalized per 90 minutes.
"""

from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from src.core.config import (
    PROGRESSIVE_DISTANCE_THRESHOLD,
    FINAL_THIRD_X,
    PENALTY_AREA_X,
    PENALTY_AREA_Y_MIN,
    PENALTY_AREA_Y_MAX,
)


class MetricsCalculator:
    """Calculate comprehensive per-90 metrics for players."""

    def __init__(self, events_df: pd.DataFrame) -> None:
        """
        Initialize calculator with events data.

        Parameters
        ----------
        events_df : DataFrame
            StatsBomb events.
        """
        self.events_df = events_df.copy()

    @staticmethod
    def _is_progressive_pass(row: pd.Series) -> bool:
        """
        Check if a pass is progressive (≥ threshold forward OR into final third).
        """
        try:
            pass_end = row.get("pass_end_location")
            location = row.get("location")
            if (
                pass_end is not None
                and location is not None
                and isinstance(pass_end, (list, np.ndarray))
                and isinstance(location, (list, np.ndarray))
                and len(location) >= 2
                and len(pass_end) >= 2
            ):
                x_progress = pass_end[0] - location[0]
                return (
                    x_progress >= PROGRESSIVE_DISTANCE_THRESHOLD
                    or pass_end[0] >= FINAL_THIRD_X
                )
        except Exception:
            pass
        return False

    @staticmethod
    def _is_progressive_carry(row: pd.Series) -> bool:
        """
        Check if a carry is progressive (≥ threshold forward OR into final third).
        """
        try:
            carry_end = row.get("carry_end_location")
            location = row.get("location")
            if (
                carry_end is not None
                and location is not None
                and isinstance(carry_end, (list, np.ndarray))
                and isinstance(location, (list, np.ndarray))
                and len(location) >= 2
                and len(carry_end) >= 2
            ):
                x_progress = carry_end[0] - location[0]
                return (
                    x_progress >= PROGRESSIVE_DISTANCE_THRESHOLD
                    or carry_end[0] >= FINAL_THIRD_X
                )
        except Exception:
            pass
        return False

    @staticmethod
    def _is_pass_into_final_third(row: pd.Series) -> bool:
        """Check if pass ends in final third."""
        try:
            pass_end = row.get("pass_end_location")
            if pass_end is not None and isinstance(pass_end, (list, np.ndarray)):
                if len(pass_end) >= 2:
                    return pass_end[0] >= FINAL_THIRD_X
        except Exception:
            pass
        return False

    @staticmethod
    def _is_pass_into_box(row: pd.Series) -> bool:
        """Check if pass ends in penalty area."""
        try:
            pass_end = row.get("pass_end_location")
            if pass_end is not None and isinstance(pass_end, (list, np.ndarray)):
                if len(pass_end) >= 2:
                    return (
                        pass_end[0] >= PENALTY_AREA_X
                        and PENALTY_AREA_Y_MIN <= pass_end[1] <= PENALTY_AREA_Y_MAX
                    )
        except Exception:
            pass
        return False

    @staticmethod
    def _is_in_defensive_third(row: pd.Series) -> bool:
        """Check if location is in defensive third."""
        try:
            location = row.get("location")
            if location is not None and isinstance(location, (list, np.ndarray)):
                if len(location) >= 2:
                    return location[0] < 40.0
        except Exception:
            pass
        return False

    def calculate_player_metrics(
        self,
        player_name: str,
        total_minutes: float,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate comprehensive per-90 metrics for a player.

        Parameters
        ----------
        player_name : str
            Player's full name.
        total_minutes : float
            Total minutes played.

        Returns
        -------
        metrics : dict or None
            Per-90 metrics, or None if invalid (e.g. 0 minutes).
        """
        if total_minutes <= 0:
            return None

        player_events = self.events_df[self.events_df["player"] == player_name].copy()
        if player_events.empty:
            return None

        per_90 = 90.0 / total_minutes
        metrics: Dict[str, float] = {}

        passes = player_events[player_events["type"] == "Pass"].copy()
        carries = player_events[player_events["type"] == "Carry"].copy()

        if not passes.empty:
            passes["progressive"] = passes.apply(self._is_progressive_pass, axis=1)
            metrics["prog_passes_p90"] = passes["progressive"].sum() * per_90
        else:
            metrics["prog_passes_p90"] = 0.0

        # Progressive carries
        if not carries.empty:
            carries["progressive"] = carries.apply(self._is_progressive_carry, axis=1)
            metrics["prog_carries_p90"] = carries["progressive"].sum() * per_90
        else:
            metrics["prog_carries_p90"] = 0.0

        if not passes.empty:
            def _is_entry_into_final_third(row):
                try:
                    pass_end = row.get("pass_end_location")
                    loc = row.get("location")
                    if pass_end is not None and loc is not None:
                        return loc[0] < FINAL_THIRD_X and pass_end[0] >= FINAL_THIRD_X
                except: pass
                return False

            passes["final_third"] = passes.apply(_is_entry_into_final_third, axis=1)
            metrics["passes_final_third_p90"] = passes["final_third"].sum() * per_90
        else:
            metrics["passes_final_third_p90"] = 0.0

        if not carries.empty:
            def _is_carry_entry_final_third(row):
                 try:
                    end = row.get("carry_end_location")
                    loc = row.get("location")
                    if end is not None and loc is not None:
                        return loc[0] < FINAL_THIRD_X and end[0] >= FINAL_THIRD_X
                 except: pass
                 return False

            carries["final_third"] = carries.apply(_is_carry_entry_final_third, axis=1) 
            metrics["carries_final_third_p90"] = carries["final_third"].sum() * per_90
        else:
            metrics["carries_final_third_p90"] = 0.0

        metrics["deep_progressions_p90"] = metrics["passes_final_third_p90"] + metrics["carries_final_third_p90"]

        if not passes.empty:
            passes["into_box"] = passes.apply(self._is_pass_into_box, axis=1)
            metrics["passes_into_box_p90"] = passes["into_box"].sum() * per_90
        else:
            metrics["passes_into_box_p90"] = 0.0

        if "pass_shot_assist" in passes.columns:
            shot_assists = passes[passes["pass_shot_assist"] == True]
            num_shot_assists = len(shot_assists)
        else:
            num_shot_assists = 0

        metrics["key_passes_p90"] = num_shot_assists * per_90
        metrics["shot_assists_p90"] = num_shot_assists * per_90
        metrics["shot_creating_actions_p90"] = num_shot_assists * per_90

        tackles = player_events[
            player_events["type"].str.contains("Tackle", case=False, na=False)
        ]
        metrics["tackles_p90"] = len(tackles) * per_90

        interceptions = player_events[player_events["type"] == "Interception"]
        metrics["interceptions_p90"] = len(interceptions) * per_90

        metrics["tackles_interceptions_p90"] = (
            metrics["tackles_p90"] + metrics["interceptions_p90"]
        )

        pressures = player_events[player_events["type"] == "Pressure"]
        metrics["pressures_p90"] = len(pressures) * per_90

        recoveries = player_events[player_events["type"] == "Ball Recovery"].copy()
        metrics["recoveries_p90"] = len(recoveries) * per_90

        if not recoveries.empty:
            recoveries["defensive_third"] = recoveries.apply(
                self._is_in_defensive_third, axis=1
            )
            metrics["recoveries_defensive_third_p90"] = (
                recoveries["defensive_third"].sum() * per_90
            )
        else:
            metrics["recoveries_defensive_third_p90"] = 0.0

        receipts = player_events[player_events["type"].str.contains("Ball Receipt", case=False, na=False)].copy()
        if not receipts.empty:
            def _is_in_final_third(row):
                 try:
                    loc = row.get("location")
                    if loc is not None and isinstance(loc, (list, np.ndarray)) and len(loc) >= 1:
                        return loc[0] >= FINAL_THIRD_X
                 except: pass
                 return False

            receipts["final_third"] = receipts.apply(_is_in_final_third, axis=1)
            metrics["deep_receptions_p90"] = receipts["final_third"].sum() * per_90
        else:
            metrics["deep_receptions_p90"] = 0.0

        metrics["total_passes"] = float(len(passes))
        if len(passes) > 0:
            if "pass_outcome" in passes.columns:
                completed = passes["pass_outcome"].isna().sum()
            else:
                completed = len(passes)

            metrics["pass_completion_%"] = (completed / len(passes) * 100.0)
        else:
            metrics["pass_completion_%"] = 0.0

        return metrics

    def calculate_metrics_for_group(
        self,
        players_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate metrics for a group of players.

        Parameters
        ----------
        players_df : DataFrame
            Requires columns 'player' and 'total_minutes', and optionally 'team'.

        Returns
        -------
        DataFrame
            All calculated metrics.
        """
        required_cols = {"player", "total_minutes"}
        missing = required_cols.difference(players_df.columns)
        if missing:
            raise ValueError(f"players_df missing required columns: {missing}")

        all_metrics: List[Dict] = []

        for _, row in players_df.iterrows():
            metrics = self.calculate_player_metrics(
                row["player"], float(row["total_minutes"])
            )
            if metrics:
                metrics["player"] = row["player"]
                metrics["team"] = row.get("team", "Unknown")
                metrics["total_minutes"] = float(row["total_minutes"])
                all_metrics.append(metrics)

        return pd.DataFrame(all_metrics)


def calculate_percentiles(
    metrics_df: pd.DataFrame,
    player_name: str,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Calculate percentile rankings for a player.

    Percentile definition: percentage of players strictly below the player's value.

    Returns
    -------
    dict
        Metric name → percentile (0–100).
    """
    player_row = metrics_df[metrics_df["player"] == player_name]
    if len(player_row) == 0:
        raise ValueError(f"Player {player_name} not found in metrics")

    player_row = player_row.iloc[0]
    percentiles: Dict[str, float] = {}

    for metric in metrics:
        if metric not in metrics_df.columns:
            continue
        series = metrics_df[metric].dropna()
        if series.empty:
            percentiles[metric] = np.nan
            continue
        value = player_row[metric]
        pct = (series < value).sum() / len(series) * 100.0
        percentiles[metric] = pct

    return percentiles
