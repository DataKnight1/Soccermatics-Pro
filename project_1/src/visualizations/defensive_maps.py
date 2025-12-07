"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Defensive activity maps showing recoveries, tackles, interceptions,
             and other defensive actions on the pitch using KDE and scatter plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import Pitch
from src.core.colors import BLUES, ACCENTS, VISUALIZATION_PRESETS


def create_defensive_activity_map(defensive_df: pd.DataFrame, title: str):
    """
    Plot defensive actions (Recoveries, Tackles, Interceptions) on a pitch.
    Uses KDE for density, plus scatter for specific events.
    """
    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)

    preset = VISUALIZATION_PRESETS["defensive_activity"]
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=preset["pitch_color"],
        line_color=preset["line_color"],
        linewidth=2.5
    )
    pitch.draw(ax=ax)

    if defensive_df.empty:
        return fig

    recoveries = defensive_df[defensive_df['type_name'] == 'Recovery']
    if not recoveries.empty:
        kde_colors = [BLUES["navy"], BLUES["argentina"], BLUES["sky"], ACCENTS["gold"]]
        kde_cmap = mcolors.LinearSegmentedColormap.from_list("kde_blue", kde_colors)

        pitch.kdeplot(
            recoveries.x, recoveries.y,
            ax=ax,
            levels=50,
            shade=True,
            cmap=kde_cmap,
            alpha=0.5,
            zorder=1
        )

    type_colors = {
        'Recovery': BLUES["argentina"],
        'Duel': ACCENTS["gold"],
        'Interception': BLUES["ice"],
        'Block': ACCENTS["coral"]
    }

    for action_type, color in type_colors.items():
        subset = defensive_df[defensive_df['type_name'] == action_type]
        if not subset.empty:
            pitch.scatter(
                subset.x, subset.y,
                ax=ax,
                c=color,
                s=100,
                alpha=0.85,
                label=action_type,
                zorder=2,
                edgecolors=BLUES["white"],
                linewidth=1.5
            )

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=12,
        framealpha=0.95,
        facecolor=BLUES["navy"],
        edgecolor=BLUES["argentina"],
        labelcolor=BLUES["white"]
    )
    ax.set_title(title, fontsize=18, fontweight="bold", color=BLUES["white"], pad=20)

    plt.tight_layout()
    return fig
