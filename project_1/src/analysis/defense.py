"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Defensive action extractor for analyzing player defensive contributions
             including recoveries, interceptions, tackles, and blocks.
"""

import pandas as pd
import numpy as np

class DefensiveActionExtractor:

    def __init__(self, events_df: pd.DataFrame):
        self.events_df = events_df

    def get_defensive_actions(self, player_name: str) -> pd.DataFrame:
        """
        Get all defensive actions for a specific player.
        Types: 'Recovery', 'Interception', 'Duel' (Tackle), 'Block'
        """
        player_events = self.events_df[self.events_df['player'] == player_name].copy()

        defensive_types = ['Recovery', 'Interception', 'Block', 'Duel', '50/50']

        def_events = player_events[player_events['type'].isin(defensive_types)].copy()

        def extract_loc(loc):
            if isinstance(loc, list) and len(loc) >= 2:
                return loc[0], loc[1]
            return np.nan, np.nan

        def_events[['x', 'y']] = def_events['location'].apply(extract_loc).apply(pd.Series)

        cols_to_keep = ['id', 'period', 'timestamp', 'type', 'x', 'y', 'under_pressure']
        def_events['type_name'] = def_events['type']

        return def_events.dropna(subset=['x', 'y'])

    def calculate_zone_counts(self, events_df: pd.DataFrame) -> pd.Series:
        """
        Assign events to 12 zones and count frequency.
        Returns a Series indexed by zone ID (1-12).
        """
        if events_df.empty:
            return pd.Series(0, index=range(1, 13))

        def get_zone(row):
            x, y = row['x'], row['y']

            if x < 30: h = 0
            elif x < 60: h = 1
            elif x < 90: h = 2
            else: h = 3

            if y < 26.67: v = 0
            elif y < 53.33: v = 1
            else: v = 2

            return h * 3 + v + 1

        events_df = events_df.copy()
        events_df['zone'] = events_df.apply(get_zone, axis=1)

        counts = events_df['zone'].value_counts()
        for z in range(1, 13):
            if z not in counts:
                counts[z] = 0

        return counts.sort_index()
