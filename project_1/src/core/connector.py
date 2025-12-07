"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Connector metrics calculator for quantifying a player's role in linking
             defense and attack through progressive passing and carrying actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

X_SPLITS = [0.0, 30.0, 60.0, 90.0, PITCH_LENGTH]
Y_SPLITS = [0.0, 26.67, 53.33, PITCH_WIDTH]


def get_zone(x: Optional[float], y: Optional[float]) -> Optional[int]:
    """
    Map StatsBomb coordinates (x, y) to a 12-zone grid.

    Zones are numbered left-to-right, then bottom-to-top:

    1  2  3
    4  5  6
    7  8  9
    10 11 12

    Returns
    -------
    zone : int or None
        Zone id in [1, 12], or None if coordinates are invalid.
    """
    if x is None or y is None:
        return None

    try:
        h_idx = np.searchsorted(X_SPLITS, x, side="right") - 1
        v_idx = np.searchsorted(Y_SPLITS, y, side="right") - 1
    except Exception:
        return None

    if not (0 <= h_idx <= 3 and 0 <= v_idx <= 2):
        return None

    return h_idx * 3 + v_idx + 1


def get_zone_centers() -> Dict[int, tuple[float, float]]:
    """
    Compute geometric centers of the 12 zones.

    Returns
    -------
    centers : dict[int, (x, y)]
    """
    centers: Dict[int, tuple[float, float]] = {}
    zone_id = 1
    for h in range(4):
        x_left, x_right = X_SPLITS[h], X_SPLITS[h + 1]
        for v in range(3):
            y_bottom, y_top = Y_SPLITS[v], Y_SPLITS[v + 1]
            centers[zone_id] = ((x_left + x_right) / 2.0, (y_bottom + y_top) / 2.0)
            zone_id += 1
    return centers


@dataclass(frozen=True)
class ConnectorConfig:
    """
    Configuration for connector metrics.

    Attributes
    ----------
    progressive_distance_threshold : float
        Minimum forward distance (x-direction) for an action to be counted
        as progressive.
    time_window_seconds : int
        Time window after a reception in which we search for a progressive
        action (pass or carry).
    max_receptions_x : float
        Maximum x-coordinate for a reception to be considered "deep".
    """
    progressive_distance_threshold: float = 10.0
    time_window_seconds: int = 5
    max_receptions_x: float = 80.0


class ConnectorMetricsCalculator:
    """
    Calculates deep connector metrics for a group of players.

    Definition (kept consistent with your analysis):

    - Deep reception: player receives a pass with x < max_receptions_x.
    - Deep progression: within `time_window_seconds` after reception,
      the player plays a pass or makes a carry that:
        * moves the ball forward in x by >= progressive_distance_threshold.
      Only the first qualifying action is counted.
    """

    def __init__(self, events_df: pd.DataFrame, config: ConnectorConfig | None = None) -> None:
        self.events_df = events_df.copy()

        if not pd.api.types.is_timedelta64_dtype(self.events_df["timestamp"]):
            self.events_df["timestamp"] = pd.to_timedelta(self.events_df["timestamp"])

        self.events_df.sort_values(["match_id", "timestamp"], inplace=True)
        self.passes_df = self.events_df[self.events_df["type"] == "Pass"].copy()

        self.config = config if config is not None else ConnectorConfig()

    def calculate_metrics_for_group(self, player_group_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate connector metrics for a DataFrame of players.

        Parameters
        ----------
        player_group_df : DataFrame
            Expected columns: 'player', 'total_minutes', 'team'.

        Returns
        -------
        DataFrame
            Connector metrics + rank + percentile + directional information.
        """
        required_cols = {"player", "total_minutes", "team"}
        missing = required_cols.difference(player_group_df.columns)
        if missing:
            raise ValueError(f"player_group_df missing required columns: {missing}")

        all_metrics: List[Dict] = []
        for _, row in player_group_df.iterrows():
            player_name = row["player"]
            total_minutes = float(row["total_minutes"])
            team_name = row["team"]

            metrics = self._calculate_player_metrics(player_name, total_minutes)
            metrics["team"] = team_name
            all_metrics.append(metrics)

        metrics_df = pd.DataFrame(all_metrics)

        metrics_df["rank"] = metrics_df["deep_progressions_p90"].rank(
            ascending=False, method="min"
        )
        metrics_df["percentile"] = metrics_df["deep_progressions_p90"].rank(
            pct=True
        ) * 100.0

        total_progs = (
            metrics_df["left_prog"] + metrics_df["center_prog"] + metrics_df["right_prog"]
        )
        metrics_df["right_side_pct"] = (
            (metrics_df["right_prog"] / total_progs.replace(0, np.nan)) * 100.0
        ).fillna(0.0)

        metrics_df = metrics_df.rename(
            columns={"progression_success_rate": "success_rate"}
        )

        return metrics_df

    def _calculate_player_metrics(self, player_name: str, total_minutes: float) -> Dict:
        """
        Calculate connector metrics for a single player.
        """
        player_events = self.events_df[self.events_df["player"] == player_name]
        passes_to_player = self.passes_df[self.passes_df["pass_recipient"] == player_name]

        deep_receptions = 0
        deep_progressions = 0

        successful_progressions: List[Dict[str, Optional[int]]] = []

        if passes_to_player.empty or total_minutes <= 0:
            return {
                "player": player_name,
                "minutes": total_minutes,
                "deep_receptions": 0,
                "deep_progressions": 0,
                "deep_receptions_p90": 0.0,
                "deep_progressions_p90": 0.0,
                "progression_success_rate": 0.0,
                "left_prog": 0,
                "center_prog": 0,
                "right_prog": 0,
                "primary_reception_zone": "N/A",
            }

        time_window = pd.Timedelta(seconds=self.config.time_window_seconds)

        for _, reception in passes_to_player.iterrows():
            end_loc = reception.get("pass_end_location")
            if end_loc is None or not isinstance(end_loc, (list, tuple)):
                continue

            reception_x, reception_y = end_loc[:2]
            if reception_x is None or reception_x >= self.config.max_receptions_x:
                continue

            deep_receptions += 1

            next_events = player_events[
                (player_events["match_id"] == reception["match_id"])
                & (player_events["timestamp"] > reception["timestamp"])
                & (player_events["timestamp"] <= reception["timestamp"] + time_window)
            ].head(2)

            for _, next_event in next_events.iterrows():
                prog_x: Optional[float] = None
                prog_y: Optional[float] = None
                event_type = next_event["type"]

                if event_type == "Pass" and next_event.get("pass_end_location"):
                    loc = next_event["pass_end_location"]
                    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        prog_x, prog_y = loc[:2]
                elif event_type == "Carry" and next_event.get("carry_end_location"):
                    loc = next_event["carry_end_location"]
                    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        prog_x, prog_y = loc[:2]

                if prog_x is None:
                    continue

                if (prog_x - reception_x) >= self.config.progressive_distance_threshold:
                    deep_progressions += 1
                    successful_progressions.append(
                        {
                            "reception_zone": get_zone(reception_x, reception_y),
                            "progression_zone": get_zone(prog_x, prog_y),
                        }
                    )
                    break

        metrics: Dict[str, object] = {
            "player": player_name,
            "minutes": total_minutes,
            "deep_receptions": deep_receptions,
            "deep_progressions": deep_progressions,
        }

        if total_minutes > 0:
            factor = 90.0 / total_minutes
            metrics["deep_receptions_p90"] = deep_receptions * factor
            metrics["deep_progressions_p90"] = deep_progressions * factor
        else:
            metrics["deep_receptions_p90"] = 0.0
            metrics["deep_progressions_p90"] = 0.0

        metrics["progression_success_rate"] = (
            (deep_progressions / deep_receptions) * 100.0 if deep_receptions > 0 else 0.0
        )

        prog_df = pd.DataFrame(successful_progressions)

        metrics["left_prog"] = prog_df["progression_zone"].isin([1, 4, 7, 10]).sum()
        metrics["center_prog"] = prog_df["progression_zone"].isin([2, 5, 8, 11]).sum()
        metrics["right_prog"] = prog_df["progression_zone"].isin([3, 6, 9, 12]).sum()

        if not prog_df.empty and not prog_df["reception_zone"].mode().empty:
            metrics["primary_reception_zone"] = int(prog_df["reception_zone"].mode().iloc[0])
        else:
            metrics["primary_reception_zone"] = "N/A"

        return metrics


def calculate_player_zone_data(
    events_df: pd.DataFrame,
    player_name: str,
    config: ConnectorConfig | None = None,
) -> pd.DataFrame:
    """
    Calculate zone-to-zone progressions for a given player.

    Returns
    -------
    DataFrame
        Columns: ['reception_zone', 'progression_zone']
    """
    cfg = config if config is not None else ConnectorConfig()

    events_df = events_df.copy()
    if not pd.api.types.is_timedelta64_dtype(events_df["timestamp"]):
        events_df["timestamp"] = pd.to_timedelta(events_df["timestamp"])

    player_events = events_df[events_df["player"] == player_name]
    passes_to_player = events_df[
        (events_df["type"] == "Pass") & (events_df["pass_recipient"] == player_name)
    ]

    progressions: List[Dict[str, Optional[int]]] = []

    if passes_to_player.empty:
        return pd.DataFrame(columns=["reception_zone", "progression_zone"])

    time_window = pd.Timedelta(seconds=cfg.time_window_seconds)

    for _, reception in passes_to_player.iterrows():
        end_loc = reception.get("pass_end_location")
        if end_loc is None or not isinstance(end_loc, (list, tuple)):
            continue

        rx, ry = end_loc[:2]
        if rx is None or rx >= cfg.max_receptions_x:
            continue

        next_events = player_events[
            (player_events["match_id"] == reception["match_id"])
            & (player_events["timestamp"] > reception["timestamp"])
            & (player_events["timestamp"] <= reception["timestamp"] + time_window)
        ].head(2)

        for _, next_event in next_events.iterrows():
            px: Optional[float] = None
            py: Optional[float] = None

            if next_event["type"] == "Pass" and next_event.get("pass_end_location"):
                loc = next_event["pass_end_location"]
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    px, py = loc[:2]
            elif next_event["type"] == "Carry" and next_event.get("carry_end_location"):
                loc = next_event["carry_end_location"]
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    px, py = loc[:2]

            if px is None:
                continue

            if (px - rx) >= cfg.progressive_distance_threshold:
                progressions.append(
                    {
                        "reception_zone": get_zone(rx, ry),
                        "progression_zone": get_zone(px, py),
                    }
                )
                break

    return pd.DataFrame(progressions)


def calculate_enzo_zone_data(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-compatible wrapper for Enzo Fernández zone progressions.

    This simply calls the generic `calculate_player_zone_data` with the
    hard-coded player name used in the app.
    """
    return calculate_player_zone_data(events_df, player_name="Enzo Fernandez")
