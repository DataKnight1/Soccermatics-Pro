"""
Author: Tiago.Monteiro
Date: 2025-12-08
Description: Analyzes passing connections between teammates, clustering passes
             by recipient to identify key partnerships and play patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class TeammateAnalysis:
    """
    Analyzes passing connections between a source player and their teammates.
    """

    def __init__(self, events_df: pd.DataFrame):
        self.events_df = events_df

    def get_pass_clusters(self, player_name: str, top_n: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Retrieves passes made by `player_name` and clusters them by recipient.
        
        Args:
            player_name: The name of the passer.
            top_n: Number of top recipients to return clusters for.

        Returns:
            Dictionary where keys are teammate names and values are DataFrames
            of pass events (x, y, end_x, end_y, etc.) to that teammate.
        """
        # Filter for passes by the player
        mask = (self.events_df['player'] == player_name) & (self.events_df['type'] == 'Pass')
        player_passes = self.events_df[mask].copy()

        if player_passes.empty:
            return {}
        
        # Ensure pass_recipient column exists (StatsBomb standard)
        if 'pass_recipient' not in player_passes.columns:
            # Try to extract from related columns if needed, but standard SB has it
            return {}

        # Valid passes only (completed is usually implied for connections, but we can filter)
        # Assuming we want successful logic, or just all attempts? 
        # Usually connections imply successful passes, let's filter for incomplete if needed?
        # Standard: Look at ALL or just SUCCESSFUL. Let's do SUCCESSFUL to show actual links.
        if 'pass_outcome' in player_passes.columns:
            player_passes = player_passes[player_passes['pass_outcome'].isna()] # NaN usually means complete in SB
        
        # Count recipients
        recipient_counts = player_passes['pass_recipient'].value_counts()
        top_recipients = recipient_counts.head(top_n).index.tolist()

        clusters = {}
        for recipient in top_recipients:
            if pd.isna(recipient):
                continue
            
            # Get passes to this person
            rec_passes = player_passes[player_passes['pass_recipient'] == recipient]
            
            # Keep relevant columns for plotting
            cols = ['location', 'pass_end_location', 'minute', 'match_id']
            # Coordinates might be in 'location' list [x,y] or separate columns.
            # StatsBombPy usually gives separate x,y if flattened, or list if not.
            # Let's handle both.
            
            clean_df = self._standardize_coords(rec_passes)
            clusters[recipient] = clean_df
            
        return clusters

    def _standardize_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures x, y, end_x, end_y columns exist."""
        df = df.copy()
        
        # If x/y already exist (standard flattened)
        if {'x', 'y'}.issubset(df.columns):
            pass 
        elif 'location' in df.columns:
            # Extract from list
            locs = df['location'].apply(lambda x: pd.Series(x) if isinstance(x, (list, tuple, np.ndarray)) else pd.Series([np.nan, np.nan]))
            df['x'] = locs[0]
            df['y'] = locs[1]

        # End locations
        if {'end_x', 'end_y'}.issubset(df.columns):
            pass # assumed implicit pass_end_location split 
        # Check standard SB names
        elif 'pass_end_location' in df.columns:
             locs = df['pass_end_location'].apply(lambda x: pd.Series(x) if isinstance(x, (list, tuple, np.ndarray)) else pd.Series([np.nan, np.nan]))
             df['end_x'] = locs[0]
             df['end_y'] = locs[1]
             
        return df
