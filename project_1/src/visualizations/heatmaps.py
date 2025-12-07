"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Pitch heatmap visualizations including zone-based heatmaps
             and difference heatmaps for comparing player activity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from mplsoccer import Pitch
from src.core.colors import BLUES, VISUALIZATION_PRESETS, get_blue_cmap
from .utils import get_zone_centers, add_orientation_guides


def create_zone_heatmap_pitch(df: pd.DataFrame, zone_col: str, title: str, cmap: str = "reception"):
    """Create a soccer pitch with a 12-zone heatmap using mplsoccer."""
    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)
    fig.patch.set_facecolor("#0B132B")

    if cmap == "progression":
        preset = VISUALIZATION_PRESETS["pitch_heatmap_progression"]
    else:
        preset = VISUALIZATION_PRESETS["pitch_heatmap_reception"]

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#0B132B",
        line_color="#FFFFFF",
        linewidth=1.5
    )
    pitch.draw(ax=ax)

    zone_centers = get_zone_centers()

    if zone_col not in df.columns:
        return fig

    zone_counts = df[zone_col].value_counts()
    max_count = zone_counts.max() if not zone_counts.empty else 0

    cmap_obj = get_blue_cmap("dark_bg")

    for zone, (x, y, w, h) in zone_centers.items():
        count = zone_counts.get(zone, 0)
        intensity = (count / max_count) ** 0.8 if max_count > 0 else 0.0

        rect_x = x - w / 2.0
        rect_y = y - h / 2.0

        rect = patches.Rectangle(
            (rect_x, rect_y),
            w,
            h,
            facecolor=cmap_obj(intensity),
            edgecolor="#FFFFFF",
            linewidth=1,
            alpha=0.7,
        )
        ax.add_patch(rect)

        if count > 0:
            text_color = preset["text_light"] if intensity > 0.5 else preset["text_dark"]
            bbox_color = BLUES["navy"] if intensity > 0.5 else BLUES["white"]

            ax.text(
                x,
                y,
                f"{count}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color,
                bbox=dict(
                    boxstyle="circle,pad=0.35",
                    facecolor=bbox_color,
                    alpha=0.5,
                    edgecolor='none'
                ),
            )

    add_orientation_guides(ax)
    ax.set_title(title, fontsize=18, fontweight="bold", color=BLUES["white"], pad=18)
    fig.subplots_adjust(top=0.9)
    return fig


def create_difference_heatmap(player_counts: pd.Series, avg_counts: pd.Series, title: str):
    """
    Difference heatmap using a single blue gradient (light = more use, dark = less use).
    """
    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#0B132B",
        line_color="#FFFFFF",
        linewidth=1.5
    )
    pitch.draw(ax=ax)

    zone_centers = get_zone_centers()

    differences = {}
    max_diff = 0.0

    for zone in range(1, 13):
        p_val = player_counts.get(zone, 0)
        a_val = avg_counts.get(zone, 0)
        diff = p_val - a_val
        differences[zone] = diff
        max_diff = max(max_diff, abs(diff))

    if max_diff == 0:
        max_diff = 1.0

    cmap = get_blue_cmap("dark_bg")
    norm = mcolors.TwoSlopeNorm(vmin=-max_diff, vcenter=0.0, vmax=max_diff)

    for zone, (x, y, w, h) in zone_centers.items():
        diff = differences[zone]

        rect_x = x - w / 2.0
        rect_y = y - h / 2.0

        rel_diff = abs(diff) / max_diff if max_diff > 0 else 0
        alpha = 0.4 + (0.5 * rel_diff)

        rect = patches.Rectangle(
            (rect_x, rect_y),
            w,
            h,
            facecolor=cmap(norm(diff)),
            edgecolor="#FFFFFF",
            linewidth=1.5,
            alpha=alpha,
        )
        ax.add_patch(rect)

        sign = "+" if diff > 0 else ""

        if abs(diff) > (max_diff * 0.6):
            text_color = BLUES["white"]
            stroke = [pe.withStroke(linewidth=3, foreground=BLUES["navy"])]
        else:
            text_color = BLUES["navy"]
            stroke = []

        ax.text(
            x, y,
            f"{sign}{diff:.1f}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=text_color,
            path_effects=stroke
        )

    add_orientation_guides(ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label('Difference vs Average CM\n(Light Blue = MORE use | Dark Blue = LESS use)',
                   fontsize=11, color=BLUES["white"])
    cbar.ax.tick_params(labelsize=10, colors=BLUES["white"])

    ax.set_title(title, fontsize=18, fontweight="bold", color=BLUES["white"], pad=18)
    fig.subplots_adjust(top=0.9)
    return fig
