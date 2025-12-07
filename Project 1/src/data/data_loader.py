"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Data loading and caching utilities for StatsBomb data with support for
             competition data, player minutes calculation, and comparison group building.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd
from statsbombpy import sb

from src.core.config import (
    COMPETITION_NAME,
    SEASON,
    CACHE_DIR,
    TARGET_TEAM,
    MIN_MINUTES_THRESHOLD,
    CM_POSITION_KEYWORDS,
)


@dataclass(frozen=True)
class CompetitionConfig:
    competition_name: str = COMPETITION_NAME
    season_name: str = SEASON


class StatsBombDataLoader:
    """Load and cache StatsBomb World Cup 2022 data (or other competitions)."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        comp_config: Optional[CompetitionConfig] = None,
        verbose: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "wc2022_cache.pkl"
        self.verbose = verbose
        self.comp_config = comp_config or CompetitionConfig()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def load_competition_data(self, force_refresh: bool = False) -> Dict:
        """
        Load competition data with caching.
        Contains 'events', 'matches', 'lineups', 'comp_id', 'season_id'.
        """
        if not force_refresh and self.cache_file.exists():
            try:
                self._log(f"[*] Loading from cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                if not {"events", "matches", "lineups"} <= cache.keys():
                    raise ValueError("Cache file missing expected keys")
                self._log(
                    f"    Loaded {len(cache['events'])} events, "
                    f"{len(cache['matches'])} matches"
                )
                return cache
            except Exception as e:
                self._log(f"[!] Failed to load cache ({e}), reloading from API...")

        return self._load_and_cache_from_api()

    def _load_and_cache_from_api(self) -> Dict:
        """Load all necessary data from the StatsBomb API and cache it."""
        self._log("[*] Loading fresh data from StatsBomb API...")

        comps = sb.competitions()
        wc = comps[
            (comps["competition_name"] == self.comp_config.competition_name)
            & (comps["season_name"] == self.comp_config.season_name)
        ]

        if len(wc) == 0:
            raise ValueError(
                f"Could not find competition "
                f"{self.comp_config.competition_name} {self.comp_config.season_name}"
            )

        comp_id = int(wc.iloc[0]["competition_id"])
        season_id = int(wc.iloc[0]["season_id"])

        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        self._log(f"    Found {len(matches)} matches")

        all_events = []
        for i, mid in enumerate(matches["match_id"].values):
            if (i + 1) % 10 == 0 or i == 0:
                self._log(f"    Loading match {i + 1}/{len(matches)} (id={mid})...")
            events = sb.events(match_id=mid)
            all_events.append(events)

        events_df = pd.concat(all_events, ignore_index=True)
        self._log(f"    Loaded {len(events_df)} events")

        all_lineups = []
        for mid in matches["match_id"].values:
            lineups = sb.lineups(match_id=mid)
            for team, lineup in lineups.items():
                lineup = lineup.copy()
                lineup["team"] = team
                lineup["match_id"] = mid
                all_lineups.append(lineup)

        lineups_df = pd.concat(all_lineups, ignore_index=True)
        self._log(f"    Loaded {len(lineups_df)} lineup entries")

        cache = {
            "events": events_df,
            "matches": matches,
            "lineups": lineups_df,
            "comp_id": comp_id,
            "season_id": season_id,
        }

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache, f)
            self._log(f"    Saved to cache: {self.cache_file}")
        except Exception as e:
            self._log(f"[!] Failed to write cache ({e}); continuing without disk cache.")

        return cache


class PlayerDataExtractor:
    """Extract player data and minutes played."""

    def __init__(self, events_df: pd.DataFrame, matches_df: pd.DataFrame):
        self.events_df = events_df
        self.matches_df = matches_df

    def get_player_minutes(self, player_name: str, team: str) -> pd.DataFrame:
        """
        Calculate minutes played for a specific player in each match.
        Returns DataFrame with ['match_id', 'minutes'].
        """
        player_matches = []

        p_events = self.events_df[self.events_df['player'] == player_name]
        if p_events.empty:
            return pd.DataFrame(columns=['match_id', 'minutes'])

        match_ids = p_events['match_id'].unique()

        for mid in match_ids:
            match_events = self.events_df[self.events_df['match_id'] == mid]
            max_duration = match_events['minute'].max()
            if pd.isna(max_duration): max_duration = 90

            started = False
            starts = match_events[match_events['type'] == 'Starting XI']
            for _, row in starts.iterrows():
                tactics = row.get('tactics', {})
                if isinstance(tactics, dict):
                    lineup = tactics.get('lineup', [])
                    for p in lineup:
                         if p.get('player', {}).get('name') == player_name:
                             started = True
                             break

            subs_in = match_events[
                (match_events['type'] == 'Substitution') &
                (match_events['substitution_replacement'] == player_name)
            ]
            subs_out = match_events[
                (match_events['type'] == 'Substitution') &
                (match_events['player'] == player_name)
            ]

            minutes = 0.0

            if started:
                start_min = 0.0
                if not subs_out.empty:
                    end_min = subs_out.iloc[0]['minute']
                else:
                    end_min = max_duration
                minutes = end_min - start_min
            elif not subs_in.empty:
                start_min = subs_in.iloc[0]['minute']
                if not subs_out.empty:
                    end_min = subs_out.iloc[0]['minute']
                else:
                    end_min = max_duration
                minutes = max(0, end_min - start_min)

            if minutes > 0:
                player_matches.append({'match_id': mid, 'minutes': minutes})

        return pd.DataFrame(player_matches)


class ComparisonGroupBuilder:
    """Build a comparison group of players."""

    def __init__(self, lineups_df: pd.DataFrame, events_df: pd.DataFrame):
        self.lineups_df = lineups_df
        self.events_df = events_df

    def get_central_midfielders(self, min_minutes: float = 270) -> pd.DataFrame:
        """
        Get all players eligible (CM/DM) with >= min_minutes.
        """
        cm_keywords = CM_POSITION_KEYWORDS if 'CM_POSITION_KEYWORDS' in globals() else ["Central Midfield", "Defensive Midfield", "Right Defensive Midfield", "Left Defensive Midfield", "Right Center Midfield", "Left Center Midfield", "Center Midfield"]

        candidates = set()

        if 'positions' in self.lineups_df.columns:
            for _, row in self.lineups_df.iterrows():
                positions = row['positions']
                if isinstance(positions, list):
                    for pos in positions:
                        p_name = pos.get('position', '')
                        if any(k in p_name for k in cm_keywords):
                            candidates.add(row['player_name'])
                            break

        results = []

        def parse_time(t_str):
            if not isinstance(t_str, str): return 90.0
            parts = t_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) + int(parts[1])/60.0
            return 90.0

        if candidates:
             player_groups = self.lineups_df[self.lineups_df['player_name'].isin(candidates)].groupby('player_name')

             for p_name, group in player_groups:
                 total_min = 0.0
                 team = group.iloc[0]['team']

                 for _, row in group.iterrows():
                     positions = row['positions']
                     if isinstance(positions, list):
                         for pos in positions:
                             t_from = pos.get('from', '00:00')
                             t_to = pos.get('to', None)
                             start = parse_time(t_from)
                             end = parse_time(t_to) if t_to else 95.0

                             total_min += max(0, end - start)

                 if total_min >= min_minutes:
                     results.append({
                         'player': p_name,
                         'total_minutes': total_min,
                         'team': team,
                         'minutes': total_min
                     })

        return pd.DataFrame(results)
