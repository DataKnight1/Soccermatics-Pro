"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Color scheme configuration featuring Argentina-inspired blue gradient theme.
             Provides color palettes, gradients, and helper functions for visualizations.
"""

from typing import Dict, List
import matplotlib.colors as mcolors

ARGENTINA_BLUE = "#75AADB"
ARGENTINA_NAVY = "#0F3D5E"
ARGENTINA_WHITE = "#FFFFFF"
ARGENTINA_SKY = "#A8D8EA"

BLUES = {
    "navy": "#0B132B",
    "deep": "#0F3D5E",
    "primary": "#1D5D9B",
    "argentina": "#75AADB",
    "sky": "#A8D8EA",
    "ice": "#C8E0F4",
    "frost": "#E3F2FD",
    "cloud": "#F0F7FC",
    "white": "#FFFFFF",
}

ACCENTS = {
    "gold": "#F3B229",
    "amber": "#F39C12",
    "coral": "#FF6B6B",
    "mint": "#4ECDC4",
    "silver": "#BDC3C7",
}

GRADIENT_LIGHT_TO_DARK = [
    BLUES["white"],
    BLUES["cloud"],
    BLUES["frost"],
    BLUES["ice"],
    BLUES["sky"],
    BLUES["argentina"],
    BLUES["primary"],
    BLUES["deep"],
    BLUES["navy"],
]

GRADIENT_DARK_TO_LIGHT = list(reversed(GRADIENT_LIGHT_TO_DARK))

HEATMAP_BLUE = [
    BLUES["white"],
    BLUES["frost"],
    BLUES["ice"],
    BLUES["sky"],
    BLUES["argentina"],
    BLUES["primary"],
]

HEATMAP_BLUE_DARK_BG = [
    BLUES["navy"],
    BLUES["deep"],
    BLUES["primary"],
    BLUES["argentina"],
    BLUES["sky"],
    BLUES["ice"],
]

DIVERGING_BLUE_GOLD = [
    ACCENTS["gold"],
    "#FFE5A0",
    BLUES["white"],
    BLUES["ice"],
    BLUES["argentina"],
    BLUES["primary"],
]

CATEGORICAL_BLUES = [
    BLUES["argentina"],
    BLUES["primary"],
    BLUES["deep"],
    BLUES["sky"],
    ACCENTS["gold"],
    ACCENTS["mint"],
    BLUES["navy"],
    ACCENTS["amber"],
]

PITCH_COLORS = {
    "dark": BLUES["navy"],
    "medium": BLUES["deep"],
    "light": BLUES["frost"],
    "lines": BLUES["white"],
}

UI_COLORS = {
    "background_primary": BLUES["white"],
    "background_secondary": BLUES["cloud"],
    "background_card": BLUES["frost"],
    "text_primary": BLUES["navy"],
    "text_secondary": BLUES["deep"],
    "border_primary": BLUES["argentina"],
    "border_secondary": BLUES["ice"],
    "highlight": ACCENTS["gold"],
    "success": ACCENTS["mint"],
    "warning": ACCENTS["amber"],
    "error": ACCENTS["coral"],
}

def get_blue_cmap(name: str = "default") -> mcolors.LinearSegmentedColormap:
    """
    Get a matplotlib colormap with blue gradient.

    Args:
        name: Type of colormap ('default', 'heatmap', 'dark_bg', 'diverging')

    Returns:
        LinearSegmentedColormap object
    """
    if name == "heatmap":
        colors = HEATMAP_BLUE
    elif name == "dark_bg":
        colors = HEATMAP_BLUE_DARK_BG
    elif name == "diverging":
        colors = DIVERGING_BLUE_GOLD
    else:
        colors = GRADIENT_LIGHT_TO_DARK

    return mcolors.LinearSegmentedColormap.from_list("blue_gradient", colors)


def get_categorical_colors(n: int = None) -> List[str]:
    """
    Get categorical colors for multiple items.

    Args:
        n: Number of colors needed (if None, returns all)

    Returns:
        List of color hex codes
    """
    if n is None:
        return CATEGORICAL_BLUES

    return (CATEGORICAL_BLUES * ((n // len(CATEGORICAL_BLUES)) + 1))[:n]


PLOTLY_BLUE_SCALE = [
    [0.0, BLUES["white"]],
    [0.2, BLUES["frost"]],
    [0.4, BLUES["ice"]],
    [0.6, BLUES["argentina"]],
    [0.8, BLUES["primary"]],
    [1.0, BLUES["navy"]],
]

PLOTLY_DIVERGING_SCALE = [
    [0.0, BLUES["primary"]],
    [0.5, BLUES["white"]],
    [1.0, ACCENTS["gold"]],
]

VISUALIZATION_PRESETS = {
    "pitch_heatmap_reception": {
        "pitch_color": BLUES["navy"],
        "line_color": BLUES["white"],
        "cmap_colors": HEATMAP_BLUE_DARK_BG,
        "text_light": BLUES["white"],
        "text_dark": BLUES["navy"],
    },
    "pitch_heatmap_progression": {
        "pitch_color": BLUES["deep"],
        "line_color": BLUES["white"],
        "cmap_colors": [
            BLUES["deep"],
            BLUES["argentina"],
            BLUES["sky"],
            ACCENTS["gold"],
            "#FFE5A0",
        ],
        "text_light": BLUES["white"],
        "text_dark": BLUES["navy"],
    },
    "difference_map": {
        "pitch_color": "#E8E8E8",
        "line_color": BLUES["navy"],
        "cmap_colors": DIVERGING_BLUE_GOLD,
        "text_light": BLUES["white"],
        "text_dark": BLUES["navy"],
    },
    "defensive_activity": {
        "pitch_color": BLUES["navy"],
        "line_color": BLUES["white"],
        "recovery_color": BLUES["argentina"],
        "duel_color": ACCENTS["gold"],
        "interception_color": BLUES["white"],
        "block_color": ACCENTS["coral"],
    },
    "radar_chart": {
        "line_color": BLUES["argentina"],
        "fill_color": f"rgba(117, 170, 219, 0.4)",
        "bg_color": "rgba(240, 247, 252, 0.5)",
        "grid_color": BLUES["ice"],
    },
}

CSS_GRADIENTS = {
    "header": f"linear-gradient(135deg, {BLUES['argentina']} 0%, {BLUES['navy']} 100%)",
    "card": f"linear-gradient(135deg, {BLUES['white']} 0%, {BLUES['cloud']} 100%)",
    "highlight": f"linear-gradient(135deg, {BLUES['frost']} 0%, {BLUES['ice']} 100%)",
    "button": f"linear-gradient(135deg, {BLUES['argentina']} 0%, {BLUES['primary']} 100%)",
}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Convert hex color to RGBA string.

    Args:
        hex_color: Hex color code (e.g., '#75AADB')
        alpha: Alpha value (0.0 to 1.0)

    Returns:
        RGBA string (e.g., 'rgba(117, 170, 219, 0.5)')
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def get_intensity_color(value: float, min_val: float = 0.0, max_val: float = 1.0,
                       reverse: bool = False) -> str:
    """
    Get color based on intensity value.

    Args:
        value: Current value
        min_val: Minimum value in range
        max_val: Maximum value in range
        reverse: If True, high values get lighter colors

    Returns:
        Hex color code
    """
    if max_val == min_val:
        return BLUES["argentina"]

    intensity = (value - min_val) / (max_val - min_val)
    intensity = max(0.0, min(1.0, intensity))

    if reverse:
        intensity = 1.0 - intensity

    # Map to gradient
    gradient = GRADIENT_LIGHT_TO_DARK
    idx = int(intensity * (len(gradient) - 1))
    return gradient[idx]


def get_text_color_for_background(bg_color: str) -> str:
    """
    Determine if text should be light or dark based on background.

    Args:
        bg_color: Background hex color

    Returns:
        Appropriate text color (white or navy)
    """
    bg_color = bg_color.lstrip('#')
    r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    return BLUES["white"] if luminance < 0.5 else BLUES["navy"]
