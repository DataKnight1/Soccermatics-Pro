"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Configuration settings for the analysis including paths, thresholds,
             position filters, and metric definitions.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = DATA_DIR / "cache"

for directory in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

COMPETITION_NAME = "FIFA World Cup"
SEASON = "2022"
TARGET_PLAYER = "Enzo Fernandez"
TARGET_TEAM = "Argentina"

MIN_MINUTES_THRESHOLD = 270
PROGRESSIVE_DISTANCE_THRESHOLD = 10
FINAL_THIRD_X = 80
PENALTY_AREA_X = 102
PENALTY_AREA_Y_MIN = 18
PENALTY_AREA_Y_MAX = 62

CM_POSITION_KEYWORDS = [
    'Center Midfield',
    'Central Midfield',
    'Center Defensive Midfield',
    'Defensive Midfield'
]

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
COLOR_SCHEME = {
    'progressive_passes': '#3B7EA1',
    'progressive_carries': '#FF6B35',
    'tackles_interceptions': '#C1292E',
    'pressures': '#FDB927',
    'recoveries': '#00A650',
    'elite': '#00A650',
    'above_average': '#3B7EA1',
    'average': '#FDB927',
    'below_average': '#C1292E',
    'background': '#F5F5F5',
    'text': '#1C1C1C',
    'text_secondary': '#4A4A4A',
    'line': '#3B3B3B'
}

SIGNIFICANCE_LEVELS = {
    'highly_significant': 0.01,
    'significant': 0.05,
    'marginally_significant': 0.1
}

MAX_REPORT_PAGES = 2
MAX_FIGURES_IN_REPORT = 2
REPORT_FORMAT = 'markdown'

METRICS_CONFIG = {
    'progression': [
        'prog_passes_p90',
        'prog_carries_p90',
        'passes_final_third_p90',
        'passes_into_box_p90'
    ],
    'creation': [
        'key_passes_p90',
        'shot_assists_p90',
        'shot_creating_actions_p90'
    ],
    'defensive': [
        'tackles_p90',
        'interceptions_p90',
        'tackles_interceptions_p90',
        'pressures_p90',
        'recoveries_p90',
        'recoveries_defensive_third_p90'
    ],
    'general': [
        'total_passes',
        'pass_completion_%'
    ]
}

KEY_METRICS_FOR_COMPARISON = [
    ('prog_passes_p90', 'Progressive Passes'),
    ('prog_carries_p90', 'Progressive Carries'),
    ('passes_final_third_p90', 'Passes into Final Third'),
    ('key_passes_p90', 'Key Passes'),
    ('tackles_interceptions_p90', 'Tackles + Interceptions'),
    ('pressures_p90', 'Pressures'),
    ('recoveries_p90', 'Ball Recoveries')
]

METRICS_FOR_HYPOTHESIS_TESTING = [
    'prog_passes_p90',
    'prog_carries_p90',
    'tackles_interceptions_p90'
]
