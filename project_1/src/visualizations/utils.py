"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Utility functions for visualization modules including zone calculations,
             name normalization, text wrapping, and player name formatting.
"""

import textwrap
import unicodedata
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional
from src.core.colors import BLUES


def get_zone_centers() -> Dict[int, Tuple[float, float, float, float]]:
    """
    Centralized 12-zone layout based on a 120x80 StatsBomb pitch.
    Returns: Dict[zone_id, (center_x, center_y, width, height)]

    Zones are numbered:
        1  2  3
        4  5  6
        7  8  9
        10 11 12
    left-to-right, bottom-to-top in the standard StatsBomb coordinate system.
    """
    PITCH_LENGTH = 120.0
    PITCH_WIDTH = 80.0
    X_SPLITS = [0.0, 30.0, 60.0, 90.0, PITCH_LENGTH]
    Y_SPLITS = [0.0, 26.67, 53.33, PITCH_WIDTH]

    centers: Dict[int, Tuple[float, float, float, float]] = {}
    zone_id = 1
    for h in range(4):
        x_left, x_right = X_SPLITS[h], X_SPLITS[h + 1]
        width = x_right - x_left
        for v in range(3):
            y_bottom, y_top = Y_SPLITS[v], Y_SPLITS[v + 1]
            height = y_top - y_bottom
            centers[zone_id] = ((x_left + x_right) / 2.0, (y_bottom + y_top) / 2.0, width, height)
            zone_id += 1
    return centers


def normalize_name(name: Optional[str]) -> str:
    """
    Normalize player names for robust matching (strip accents/case).
    """
    if not isinstance(name, str):
        return ""
    cleaned = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return cleaned.strip().lower()


def wrap_label(text: str, width: int = 18) -> str:
    """
    Soft-wrap long metric labels so they don't overlap the plot area.
    """
    if not isinstance(text, str):
        return ""
    return textwrap.fill(text, width=width, break_long_words=False)


def add_orientation_guides(ax, arrow_color: str = BLUES["ice"], text_color: str = BLUES["white"]) -> None:
    """
    Add subtle left/right arrows below the pitch so viewers can quickly read orientation.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    span_x = x_max - x_min
    span_y = y_max - y_min

    y_pos = y_min + 0.015 * span_y
    arrow_len = span_x * 0.05
    gap = span_x * 0.02

    left_start = x_min + gap + arrow_len
    ax.add_patch(
        patches.FancyArrow(
            left_start,
            y_pos,
            -arrow_len,
            0,
            width=0.2,
            head_width=1.5,
            head_length=1.8,
            color=arrow_color,
            length_includes_head=True,
            zorder=6,
            alpha=0.5,
        )
    )
    ax.text(
        left_start - (arrow_len * 0.5) + (arrow_len * 0.15),
        y_pos + span_y * 0.01,
        "LEFT",
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="normal",
        color=text_color,
        alpha=0.7,
        zorder=7,
    )

    right_start = x_max - gap - arrow_len
    ax.add_patch(
        patches.FancyArrow(
            right_start,
            y_pos,
            arrow_len,
            0,
            width=0.2,
            head_width=1.5,
            head_length=1.8,
            color=arrow_color,
            length_includes_head=True,
            zorder=6,
            alpha=0.5,
        )
    )
    ax.text(
        right_start + (arrow_len * 0.5) - (arrow_len * 0.15),
        y_pos + span_y * 0.01,
        "RIGHT",
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="normal",
        color=text_color,
        alpha=0.7,
        zorder=7,
    )


def format_player_display_name(name: str) -> str:
    """
    For display purposes:
    - If name has more than 3 words, use only first and last.
      e.g., 'Bernardo Mota Pereira da Silva' -> 'Bernardo Silva'
    - Otherwise, keep name as-is.
    """
    if not isinstance(name, str):
        return ""

    parts = name.strip().split()
    # Case 1: More than 3 parts -> First Last (e.g. Bernardo Mota Veiga de Carvalho e Silva -> Bernardo Silva)
    if len(parts) > 3:
        return f"{parts[0]} {parts[-1]}"
    
    # Case 2: Exactly 3 parts and long (e.g. Thomas Teye Partey -> Thomas Partey)
    # This avoids shortening "Frenkie de Jong" (15 chars) or "Kevin De Bruyne" (15 chars), but catches "Thomas Teye Partey" (18 chars)
    if len(parts) == 3 and len(name) > 16:
        return f"{parts[0]} {parts[-1]}"
    
    return name


def calculate_percentile(value, data_series):
    """Calculate percentile rank of a value within a pandas series."""
    import pandas as pd
    if pd.isna(value):
        return 0
    return (data_series < value).mean() * 100
