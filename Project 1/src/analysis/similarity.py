"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Player similarity calculator using Euclidean distance on Z-scored metrics
             to identify statistically similar players.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

def calculate_similarity(df: pd.DataFrame, target_player: str, metrics: list, top_n: int = 5) -> pd.DataFrame:
    """
    Calculates the most similar players to the target player using Euclidean distance on Z-scores.
    
    Args:
        df: DataFrame containing player metrics.
        target_player: Name of the player to find matches for.
        metrics: List of column names to use for the similarity calculation.
        top_n: Number of similar players to return (excluding the target).
        
    Returns:
        DataFrame of top_n most similar players with their distance scores.
    """
    # 1. Check data availability
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        raise ValueError("No valid metrics found for similarity calculation.")
    
    # 2. Prepare Data
    # Drop rows with NaN in the selected metrics
    data = df.dropna(subset=available_metrics).copy()
    
    # Check if target exists
    if target_player not in data['player'].values:
        raise ValueError(f"Player '{target_player}' not found in dataset.")
        
    # 3. Calculate Z-Scores
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[available_metrics])
    
    # Create DataFrame for scaled values
    scaled_df = pd.DataFrame(scaled_data, columns=available_metrics, index=data.index)
    
    # 4. Get Target Vector
    target_idx = data[data['player'] == target_player].index[0]
    target_vector = scaled_df.loc[[target_idx]]
    
    # 5. Calculate Euclidean Distance
    distances = cdist(scaled_df, target_vector, metric='euclidean')
    data['similarity_distance'] = distances
    
    # 6. Sort and Filter
    # Exclude the target player (distance 0)
    top_similar = data[data['player'] != target_player].sort_values('similarity_distance').head(top_n)
    
    return top_similar[['player', 'team', 'similarity_distance'] + available_metrics]
