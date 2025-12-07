"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Analysis package containing modules for defensive actions, player metrics,
             similarity calculations, and statistical analysis.
"""

from .defense import DefensiveActionExtractor
from .metrics import MetricsCalculator
from .similarity import calculate_similarity
from .statistical_analysis import StatisticalAnalyzer

__all__ = [
    'DefensiveActionExtractor',
    'MetricsCalculator',
    'calculate_similarity',
    'StatisticalAnalyzer',
]
