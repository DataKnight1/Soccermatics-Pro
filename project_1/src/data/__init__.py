"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Data loading and management package for StatsBomb data including
             competition data, player statistics, and comparison group builders.
"""

from .data_loader import (
    CompetitionConfig,
    StatsBombDataLoader,
    PlayerDataExtractor,
    ComparisonGroupBuilder,
)

__all__ = [
    'CompetitionConfig',
    'StatsBombDataLoader',
    'PlayerDataExtractor',
    'ComparisonGroupBuilder',
]
