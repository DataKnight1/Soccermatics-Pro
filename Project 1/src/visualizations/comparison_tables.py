"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Comparison table visualizations including percentile heatmap tables
             for multi-player metric comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List
from src.core.colors import BLUES, get_blue_cmap
from .utils import normalize_name, format_player_display_name, calculate_percentile


def create_percentile_heatmap_table(
    selected_players_df: pd.DataFrame,
    all_players_df: pd.DataFrame,
    metrics: List[str],
    metric_labels: List[str],
    title: str = "Player Comparison - Percentile Rankings"
) -> plt.Figure:
    """
    Heatmap-style table of percentile rankings for selected players.
    - Blue gradient for percentile values.
    - Colorbar legend on the right.
    - Title placed above axis so it never overlaps values.
    - Enzo Fernández is always forced to be the first column when present.
    - Long player names are shortened to 'First Last' for display.
    - Header boxes have enough spacing between them and are centered over their columns.
    """
    if selected_players_df.empty or not metrics:
        return plt.figure()

    enzo_key = normalize_name("Enzo Fernández")

    df_sel = selected_players_df.copy()
    df_sel["_norm_player"] = df_sel["player"].apply(normalize_name)

    enzo_mask = df_sel["_norm_player"] == enzo_key
    if enzo_mask.any():
        enzo_df = df_sel[enzo_mask]
        rest_df = df_sel[~enzo_mask]
        df_sel = pd.concat([enzo_df, rest_df], axis=0)
    else:
        df_sel = df_sel

    percentile_data = []
    player_names = []

    for _, player_row in df_sel.iterrows():
        raw_name = player_row['player']
        display_name = format_player_display_name(raw_name)
        team_name = player_row.get('team', '')

        if team_name:
            player_names.append(f"{display_name}\n{team_name}")
        else:
            player_names.append(display_name)

        row_percentiles = []
        for metric in metrics:
            if metric in all_players_df.columns:
                percentile = calculate_percentile(player_row[metric], all_players_df[metric])
                row_percentiles.append(percentile)
            else:
                row_percentiles.append(0.0)
        percentile_data.append(row_percentiles)

    percentile_array = np.array(percentile_data)

    n_players = len(player_names)
    n_metrics = len(metric_labels)

    base_width_per_player = 2.8
    fig_width = max(16, n_players * base_width_per_player)
    fig_height = max(8, n_metrics * 0.6 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)
    fig.patch.set_facecolor(BLUES["navy"])
    ax.set_facecolor(BLUES["navy"])

    fig.subplots_adjust(left=0.25, right=0.8, top=0.82, bottom=0.15)

    cmap = get_blue_cmap("dark_bg")
    norm = plt.Normalize(vmin=0, vmax=100)

    cell_height = 0.8
    cell_width = 1.5

    if n_players <= 6:
        header_fontsize = 10
    elif n_players <= 10:
        header_fontsize = 9
    else:
        header_fontsize = 8

    header_pad = 0.25

    for i, metric_label in enumerate(metric_labels):
        row_center = -(i + 1) * cell_height

        ax.text(
            -0.5,
            row_center,
            metric_label,
            ha='right',
            va='center',
            fontsize=11,
            color=BLUES["white"],
            fontweight='bold'
        )

        for j in range(n_players):
            percentile = percentile_array[j, i]
            color = cmap(norm(percentile))

            rect = patches.Rectangle(
                (j * cell_width, row_center - cell_height / 2),
                cell_width * 0.95,
                cell_height * 0.9,
                facecolor=color,
                edgecolor=BLUES["white"],
                linewidth=1.0,
                alpha=0.9
            )
            ax.add_patch(rect)

            p_int = int(round(percentile))
            suffix = "th"
            if p_int % 10 == 1 and p_int % 100 != 11:
                suffix = "st"
            elif p_int % 10 == 2 and p_int % 100 != 12:
                suffix = "nd"
            elif p_int % 10 == 3 and p_int % 100 != 13:
                suffix = "rd"

            display_text = f"{p_int}{suffix}"

            intensity = norm(percentile)
            text_color = BLUES["navy"] if intensity > 0.6 else BLUES["white"]

            x_center = j * cell_width + cell_width / 2
            ax.text(
                x_center,
                row_center,
                display_text,
                ha='center',
                va='center',
                fontsize=13,
                fontweight='bold',
                color=text_color
            )

    header_y = cell_height * 0.9

    for j, disp_name in enumerate(player_names):
        x_center = j * cell_width + cell_width / 2
        ax.text(
            x_center,
            header_y,
            disp_name,
            ha='center',
            va='center',
            fontsize=header_fontsize,
            fontweight='bold',
            color=BLUES["white"],
            bbox=dict(
                boxstyle=f"round,pad={header_pad}",
                facecolor=BLUES["primary"],
                edgecolor=BLUES["white"],
                linewidth=1.2,
                alpha=0.95
            )
        )

    top_y = header_y + cell_height
    bottom_y = -(n_metrics + 1) * cell_height - 0.5

    ax.set_xlim(-1.5, n_players * cell_width + 0.5)
    ax.set_ylim(bottom_y, top_y)
    ax.axis('off')

    fig.suptitle(
        title,
        y=0.97,
        fontsize=18,
        fontweight='bold',
        color=BLUES["white"]
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation='vertical',
        fraction=0.04,
        pad=0.03
    )
    cbar.set_label(
        "Percentile Rank",
        fontsize=11,
        color=BLUES["white"]
    )
    cbar.ax.tick_params(labelsize=10, colors=BLUES["white"])

    return fig
