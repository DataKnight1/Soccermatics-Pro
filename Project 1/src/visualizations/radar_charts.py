"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Radar chart and pizza plot visualizations for player performance metrics,
             including single player pizzas, comparison pizzas, and percentile bars.
"""

import pandas as pd
import plotly.graph_objects as go
from mplsoccer import PyPizza
from matplotlib.font_manager import FontProperties
from typing import List, Dict
from src.core.colors import BLUES, ACCENTS
from .utils import calculate_percentile, format_player_display_name


class RobustFont:
    def __init__(self, prop):
        self.prop = prop


font_normal = RobustFont(FontProperties(family="sans-serif", weight="normal"))
font_bold = RobustFont(FontProperties(family="sans-serif", weight="bold"))
font_italic = RobustFont(FontProperties(family="sans-serif", style="italic"))


def create_single_pizza(player_row: pd.Series, all_players_df: pd.DataFrame,
                        metrics: List[str], metric_labels: List[str], player_name: str):
    """
    Create a single player Pizza Plot using mplsoccer.
    Calculates percentiles dynamically based on all_players_df.
    """
    values = []
    for metric in metrics:
        if metric in all_players_df.columns:
            player_val = player_row[metric]
            percentile = calculate_percentile(player_val, all_players_df[metric])
            values.append(round(percentile))
        else:
            values.append(0)

    values = [0 if pd.isna(x) else x for x in values]

    baker = PyPizza(
        params=metric_labels,
        background_color="#0B132B",
        straight_line_color="#1D5D9B",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=20,
        last_circle_color="#75AADB"
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(8, 8.5),
        color_blank_space="same",
        slice_colors=["#1D5D9B"] * len(values),
        value_colors=["#FFFFFF"] * len(values),
        value_bck_colors=["#1D5D9B"] * len(values),
        blank_alpha=0.1,
        kwargs_slices=dict(
            edgecolor="#0B132B", zorder=2, linewidth=1
        ),
        kwargs_params=dict(
            color="#FFFFFF", fontsize=10,
            fontproperties=font_normal.prop, va="center"
        ),
        kwargs_values=dict(
            color="#FFFFFF", fontsize=10,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#FFFFFF", facecolor="#0F3D5E",
                boxstyle="round,pad=0.15", lw=1
            )
        )
    )

    display_name = format_player_display_name(player_name)

    fig.text(
        0.515, 0.975, f"{display_name}", size=16,
        ha="center", fontproperties=font_bold.prop, color="#FFFFFF"
    )

    fig.text(
        0.515, 0.953,
        f"Percentile Rank vs Top-30 Connectors | World Cup 2022",
        size=10,
        ha="center", fontproperties=font_normal.prop, color="#75AADB"
    )

    return fig


def create_comparison_pizza(players_df: pd.DataFrame, player_names: List[str],
                            all_players: pd.DataFrame, metrics: List[str] = None,
                            labels: List[str] = None):
    """
    Create a comparison Pizza Plot for two players.
    Note: PyPizza comparison is best for 2 players. If more are selected, we strictly compare the first two.
    """
    if metrics is None or labels is None:
        metrics = ["deep_progressions_p90", "success_rate", "deep_receptions_p90"]
        labels = ["Deep Progressions", "Success Rate", "Deep Receptions"]

    if len(player_names) < 2:
        return create_single_pizza(
            players_df[players_df['player'] == player_names[0]].iloc[0],
            all_players,
            metrics,
            labels,
            player_names[0]
        )

    p1_name = player_names[0]
    p2_name = player_names[1]

    row1 = players_df[players_df['player'] == p1_name].iloc[0]
    row2 = players_df[players_df['player'] == p2_name].iloc[0]

    values1 = []
    values2 = []

    for metric in metrics:
        if metric in all_players.columns:
            v1 = calculate_percentile(row1[metric], all_players[metric])
            v2 = calculate_percentile(row2[metric], all_players[metric])
            values1.append(round(v1))
            values2.append(round(v2))
        else:
            values1.append(0)
            values2.append(0)

    baker = PyPizza(
        params=labels,
        background_color="#0B132B",
        straight_line_color="#1D5D9B",
        straight_line_lw=1,
        last_circle_lw=1,
        last_circle_color="#75AADB",
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        values1,
        compare_values=values2,
        figsize=(10, 10),
        color_blank_space="same",
        blank_alpha=0.1,
        param_location=110,
        kwargs_slices=dict(
            facecolor="#1D5D9B", edgecolor="#0B132B",
            zorder=1, linewidth=1
        ),
        kwargs_compare=dict(
            facecolor="#F3B229", edgecolor="#0B132B", zorder=3, linewidth=1,
        ),
        kwargs_params=dict(
            color="#FFFFFF", fontsize=10, zorder=5,
            fontproperties=font_normal.prop, va="center"
        ),
        kwargs_values=dict(
            color="#FFFFFF", fontsize=10,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#FFFFFF", facecolor="#1D5D9B",
                boxstyle="round,pad=0.15", lw=1
            )
        ),
        kwargs_compare_values=dict(
            color="#000000", fontsize=10,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="#F3B229",
                boxstyle="round,pad=0.15", lw=1
            )
        )
    )

    p1_display = format_player_display_name(p1_name)
    p2_display = format_player_display_name(p2_name)

    fig.text(
        0.515, 0.99, f"{p1_display} vs {p2_display}",
        size=16, ha="center", fontproperties=font_bold.prop, color="#FFFFFF"
    )

    fig.text(0.35, 0.96, p1_display, size=11, color="#75AADB", ha="center", fontproperties=font_bold.prop)
    fig.text(0.65, 0.96, p2_display, size=11, color="#F3B229", ha="center", fontproperties=font_bold.prop)

    return fig


def create_percentile_bars(player_row: pd.Series, metrics: Dict[str, str], player_name: str) -> go.Figure:
    """
    Create horizontal bar chart showing percentile rankings for multiple metrics.
    """
    percentiles = []
    labels = []

    for metric_key, metric_label in metrics.items():
        if metric_key in player_row:
            percentiles.append(player_row[metric_key])
            labels.append(metric_label)

    colors = []
    for p in percentiles:
        if p >= 90:
            colors.append(ACCENTS["gold"])
        elif p >= 75:
            colors.append(BLUES["argentina"])
        elif p >= 50:
            colors.append(BLUES["primary"])
        elif p >= 25:
            colors.append(BLUES["deep"])
        else:
            colors.append(BLUES["navy"])

    fig = go.Figure(go.Bar(
        x=percentiles,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=BLUES["white"], width=1)
        ),
        text=[f"{p:.0f}%" for p in percentiles],
        textposition="outside",
        textfont=dict(size=14, color=BLUES["white"], family="Inter")
    ))

    fig.update_layout(
        title=dict(
            text=f"{player_name} - Percentile Rankings",
            font=dict(size=18, color=BLUES["white"], family="Arial Black")
        ),
        xaxis=dict(
            title="Percentile",
            range=[0, 115],
            showgrid=True,
            gridcolor=BLUES["primary"],
            linecolor=BLUES["white"],
            tickfont=dict(color=BLUES["white"])
        ),
        yaxis=dict(
            showgrid=False,
            linecolor=BLUES["white"],
            tickfont=dict(color=BLUES["white"])
        ),
        plot_bgcolor=BLUES["navy"],
        paper_bgcolor=BLUES["navy"],
        font=dict(family="Inter", size=12, color=BLUES["white"]),
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=30)
    )

    for ref_val, ref_label in [(25, "Q1"), (50, "Median"), (75, "Q3")]:
        fig.add_vline(
            x=ref_val,
            line_dash="dot",
            line_color=BLUES["ice"],
            line_width=1,
            annotation_text=ref_label,
            annotation_position="top",
            annotation_font=dict(color=BLUES["sky"])
        )

    return fig
