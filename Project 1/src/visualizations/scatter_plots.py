"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Scatter plot visualizations for player comparisons including quadrant plots,
             dispersion plots, and metric pitch comparisons using both Plotly and Matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
from mplsoccer import Pitch
from typing import List
from src.core.colors import BLUES, ACCENTS, get_blue_cmap
from .utils import normalize_name, wrap_label, format_player_display_name


def create_quadrant_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                              x_label: str, y_label: str, title: str,
                              highlight_players: List[str] = None) -> go.Figure:
    """
    Scatter plot highlighting specific players against the population.
    """
    if df.empty:
        return go.Figure()

    x_mean = df[x_col].mean()
    y_mean = df[y_col].mean()

    if highlight_players is None:
        highlight_players = ["Enzo Fernández"]

    def get_color(player):
        if player in highlight_players:
            idx = highlight_players.index(player)
            if idx == 0:
                return "Highlight 1"
            else:
                return "Highlight 2"
        return "Population"

    df = df.copy()
    df['category'] = df['player'].apply(get_color)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        hover_data=['player', 'team'],
        color='category',
        title=title,
        labels={x_col: x_label, y_col: y_label},
        color_discrete_map={
            "Population": "#666666",
            "Highlight 1": "#75AADB",
            "Highlight 2": "#F3B229",
        },
        opacity=1.0
    )

    fig.add_vline(
        x=x_mean,
        line_width=1,
        line_dash="dash",
        line_color="#FFFFFF",
        opacity=0.3
    )
    fig.add_hline(
        y=y_mean,
        line_width=1,
        line_dash="dash",
        line_color="#FFFFFF",
        opacity=0.3
    )

    fig.update_traces(marker=dict(size=8, opacity=0.4, line=dict(width=0)))

    fig.update_traces(
        selector=dict(name="Highlight 1"),
        marker=dict(size=18, opacity=1.0, line=dict(width=2, color="#FFFFFF"))
    )
    fig.update_traces(
        selector=dict(name="Highlight 2"),
        marker=dict(size=16, opacity=1.0, line=dict(width=2, color="#FFFFFF"))
    )

    fig.update_layout(
        plot_bgcolor=BLUES["navy"],
        paper_bgcolor=BLUES["navy"],
        font=dict(family="Inter", size=13, color=BLUES["white"]),
        title_font=dict(size=20, color=BLUES["white"], family="Inter"),
        legend_title_text="",
        legend=dict(
            bgcolor=BLUES["navy"],
            bordercolor=BLUES["argentina"],
            borderwidth=1,
            font=dict(color=BLUES["white"], size=13)
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor=BLUES["primary"],
            zerolinewidth=1,
            linecolor=BLUES["white"],
            linewidth=1,
            title_font=dict(size=14, color=BLUES["white"], family="Inter")
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor=BLUES["primary"],
            zerolinewidth=1,
            linecolor=BLUES["white"],
            linewidth=1,
            title_font=dict(size=14, color=BLUES["white"], family="Inter")
        ),
        height=600
    )

    return fig


def create_quadrant_scatter_mpl(df: pd.DataFrame, x_col: str, y_col: str,
                                x_label: str, y_label: str, title: str,
                                highlight_players: List[str] = None) -> plt.Figure:
    """
    Matplotlib scatter plot with quadrant lines, highlighting Enzo separately.
    Uses name normalization so Enzo is detected regardless of accents/case.
    """
    if df.empty:
        return plt.figure()

    x_mean = df[x_col].mean()
    y_mean = df[y_col].mean()

    if highlight_players is None:
        highlight_players = []

    enzo_key = normalize_name("Enzo Fernández")

    df = df.copy()
    df["_norm_player"] = df["player"].apply(normalize_name)
    comparison_keys = {normalize_name(p) for p in highlight_players if normalize_name(p) != enzo_key}

    is_enzo = df["_norm_player"] == enzo_key
    is_comparison = df["_norm_player"].isin(comparison_keys)
    is_other = ~(is_enzo | is_comparison)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    fig.patch.set_facecolor(BLUES["navy"])
    ax.set_facecolor(BLUES["navy"])

    comp_colors = [ACCENTS["gold"], ACCENTS["coral"], ACCENTS["mint"], ACCENTS["amber"]]

    if is_other.any():
        ax.scatter(
            df.loc[is_other, x_col],
            df.loc[is_other, y_col],
            s=60,
            color="#666666",
            alpha=0.4,
            edgecolors='none',
            label="Other players",
            zorder=1
        )

    valid_highlights = [p for p in highlight_players if normalize_name(p) in df["_norm_player"].values]

    comp_idx = 0
    for player in valid_highlights:
        norm_p = normalize_name(player)
        if norm_p == enzo_key:
            continue

        p_mask = df["_norm_player"] == norm_p
        if p_mask.any():
            c = comp_colors[comp_idx % len(comp_colors)]
            comp_idx += 1

            real_name = df.loc[p_mask, "player"].iloc[0]
            display_name = format_player_display_name(real_name)

            ax.scatter(
                df.loc[p_mask, x_col],
                df.loc[p_mask, y_col],
                s=140,
                color=c,
                alpha=1.0,
                edgecolors=BLUES["white"],
                linewidth=2,
                label=display_name,
                zorder=3
            )

    if is_enzo.any():
        ax.scatter(
            df.loc[is_enzo, x_col],
            df.loc[is_enzo, y_col],
            s=180,
            color=BLUES["argentina"],
            alpha=1.0,
            edgecolors=BLUES["white"],
            linewidth=2.5,
            label="Enzo Fernández",
            zorder=4
        )

    ax.axvline(x_mean, color=BLUES["white"], linestyle='--', linewidth=1, alpha=0.3, zorder=0)
    ax.axhline(y_mean, color=BLUES["white"], linestyle='--', linewidth=1, alpha=0.3, zorder=0)

    ax.set_xlabel(x_label, fontsize=14, color=BLUES["white"], fontfamily="sans-serif")
    ax.set_ylabel(y_label, fontsize=14, color=BLUES["white"], fontfamily="sans-serif")
    ax.set_title(title, fontsize=18, color=BLUES["white"], fontfamily="sans-serif", pad=20)

    ax.tick_params(axis='both', colors=BLUES["white"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(BLUES["white"])
        spine.set_linewidth(1)

    ax.grid(False)

    legend = ax.legend(
        loc='upper right',
        fontsize=12,
        framealpha=0.9,
        edgecolor=BLUES["argentina"],
        facecolor=BLUES["navy"]
    )
    for text in legend.get_texts():
        text.set_color(BLUES["white"])

    plt.tight_layout()
    return fig


def create_dispersion_plot(df: pd.DataFrame, metric: str, metric_label: str,
                           highlight_players: List[str] = None):
    """
    1D Horizontal Dispersion Plot (Strip Plot) with dark blue background.
    Highlights specific players.
    Metric label is wrapped in the title for long names.
    """
    if df.empty or metric not in df.columns:
        return go.Figure()

    if highlight_players is None:
        highlight_players = ["Enzo Fernández"]

    df = df.copy()

    def get_category(player):
        if player in highlight_players:
            idx = highlight_players.index(player)
            if idx == 0:
                return "Highlight 1"
            else:
                return "Highlight 2"
        return "Population"

    df['category'] = df['player'].apply(get_category)

    np.random.seed(42)
    df['y'] = np.random.uniform(-0.2, 0.2, len(df))
    df.loc[df['category'] != "Population", 'y'] = 0

    wrapped_title = wrap_label(f"{metric_label} Distribution", width=28).replace("\n", "<br>")

    fig = px.strip(
        df,
        x=metric,
        y='y',
        color='category',
        hover_data=['player', 'team'],
        title=wrapped_title,
        color_discrete_map={
            "Population": BLUES["primary"],
            "Highlight 1": BLUES["argentina"],
            "Highlight 2": ACCENTS["gold"],
        }
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.5, line=dict(width=0)),
        selector=dict(name="Population")
    )
    fig.update_traces(
        marker=dict(size=16, opacity=1.0, line=dict(width=2, color=BLUES["white"])),
        selector=dict(name="Highlight 1")
    )
    fig.update_traces(
        marker=dict(size=14, opacity=1.0, line=dict(width=2, color=BLUES["white"])),
        selector=dict(name="Highlight 2")
    )

    fig.update_layout(
        plot_bgcolor=BLUES["navy"],
        paper_bgcolor=BLUES["navy"],
        font=dict(family="Inter", size=13, color=BLUES["white"]),
        title_font=dict(size=18, color=BLUES["white"], family="Inter"),
        showlegend=True,
        legend=dict(
            bgcolor=BLUES["navy"],
            bordercolor=BLUES["argentina"],
            borderwidth=1,
            font=dict(color=BLUES["white"])
        ),
        xaxis=dict(
            title=metric_label,
            showgrid=True,
            gridcolor=BLUES["primary"],
            linecolor=BLUES["white"],
            title_font=dict(size=14, color=BLUES["white"])
        ),
        yaxis=dict(
            visible=False,
            range=[-0.5, 0.5]
        ),
        height=300
    )

    return fig


def create_dispersion_plot_mpl(df: pd.DataFrame, metric: str, metric_label: str,
                                highlight_players: List[str] = None) -> plt.Figure:
    """
    Matplotlib 1D horizontal dispersion (strip plot) with jitter and robust Enzo matching.
    Long metric labels are wrapped to avoid overlapping the plot area.
    """
    if df.empty or metric not in df.columns:
        return plt.figure()

    if highlight_players is None:
        highlight_players = []

    enzo_key = normalize_name("Enzo Fernández")
    comparison_keys = {normalize_name(p) for p in highlight_players if normalize_name(p) != enzo_key}

    df = df.copy()
    df["_norm_player"] = df["player"].apply(normalize_name)

    is_enzo = df["_norm_player"] == enzo_key
    is_comparison = df["_norm_player"].isin(comparison_keys)
    is_other = ~(is_enzo | is_comparison)

    np.random.seed(42)
    df['y_jitter'] = np.random.uniform(-0.2, 0.2, len(df))
    df.loc[is_enzo | is_comparison, 'y_jitter'] = 0

    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
    fig.patch.set_facecolor(BLUES["navy"])
    ax.set_facecolor(BLUES["navy"])

    if is_other.any():
        ax.scatter(
            df.loc[is_other, metric],
            df.loc[is_other, 'y_jitter'],
            s=60,
            color=BLUES["primary"],
            alpha=0.5,
            edgecolors='none',
            label="Other players",
            zorder=1
        )

    # Use a loop for comparison players so we can label them correctly in legend
    valid_comparison = [p for p in highlight_players if normalize_name(p) in df["_norm_player"].values and normalize_name(p) != enzo_key]
    
    comp_colors = [ACCENTS["gold"], ACCENTS["coral"], ACCENTS["mint"], ACCENTS["amber"]]
    
    # If we have specific comparison players, plot them individually for legend
    if valid_comparison:
        comp_idx = 0
        for player in valid_comparison:
            norm_p = normalize_name(player)
            p_mask = df["_norm_player"] == norm_p
            if p_mask.any():
                 c = comp_colors[comp_idx % len(comp_colors)]
                 comp_idx += 1
                 
                 real_name = df.loc[p_mask, "player"].iloc[0]
                 display_name = format_player_display_name(real_name)
                 
                 ax.scatter(
                    df.loc[p_mask, metric],
                    df.loc[p_mask, 'y_jitter'],
                    s=140,
                    color=c,
                    alpha=1.0,
                    edgecolors=BLUES["white"],
                    linewidth=2,
                    label=display_name,
                    zorder=3
                )
    elif is_comparison.any(): 
        # Fallback if no specific list but key match (shouldn't happen with current logic)
        ax.scatter(
            df.loc[is_comparison, metric],
            df.loc[is_comparison, 'y_jitter'],
            s=140,
            color=ACCENTS["gold"],
            alpha=1.0,
            edgecolors=BLUES["white"],
            linewidth=2,
            label="Comparison Players",
            zorder=3
        )


    if is_enzo.any():
        ax.scatter(
            df.loc[is_enzo, metric],
            df.loc[is_enzo, 'y_jitter'],
            s=160,
            color=BLUES["argentina"],
            alpha=1.0,
            edgecolors=BLUES["white"],
            linewidth=2.5,
            label="Enzo Fernández",
            zorder=4
        )

    wrapped_label = wrap_label(metric_label, width=20)
    wrapped_title = wrap_label(f"{metric_label} Distribution", width=30)

    ax.set_xlabel(wrapped_label, fontsize=14, color=BLUES["white"], fontfamily="sans-serif")
    ax.set_title(wrapped_title, fontsize=18, color=BLUES["white"],
                 fontfamily="sans-serif", pad=20)

    ax.tick_params(axis='x', colors=BLUES["white"], labelsize=11)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(BLUES["white"])
    ax.spines['bottom'].set_linewidth(1)

    ax.grid(axis='x', color=BLUES["primary"], alpha=0.3, linestyle='-', linewidth=0.5)

    legend = ax.legend(
        loc='upper right',
        fontsize=11,
        framealpha=0.9,
        edgecolor=BLUES["argentina"],
        facecolor=BLUES["navy"]
    )
    for text in legend.get_texts():
        text.set_color(BLUES["white"])

    plt.tight_layout()
    return fig


def create_metric_pitch_comparison(comparison_df: pd.DataFrame, metric: str,
                                   metric_label: str, color_scale: str):
    """Side-by-side pitch visualization for a metric across selected players (blue theme)."""
    num_players = len(comparison_df)
    fig, axes = plt.subplots(1, num_players, figsize=(6 * num_players, 8), dpi=120)
    fig.patch.set_facecolor(BLUES["navy"])

    if num_players == 1:
        axes = [axes]

    metric_vals = comparison_df[metric].values.astype(float)
    max_val = np.nanmax(metric_vals) if len(metric_vals) else 1.0
    min_val = np.nanmin(metric_vals) if len(metric_vals) else 0.0

    cmap_obj = get_blue_cmap("dark_bg")

    for ax, (_, player_row) in zip(axes, comparison_df.iterrows()):
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color=BLUES["navy"],
            line_color=BLUES["white"],
            linewidth=1.5
        )
        pitch.draw(ax=ax)
        ax.set_facecolor(BLUES["navy"])

        val = player_row[metric]
        if pd.isna(val) or max_val == min_val:
            intensity = 0.5
        else:
            intensity = (val - min_val) / (max_val - min_val)

        color = cmap_obj(float(intensity))

        pitch_rect = patches.Rectangle(
            (0, 0),
            120,
            80,
            facecolor=color,
            zorder=0,
            alpha=0.75,
        )
        ax.add_patch(pitch_rect)

        player_name = player_row["player"]
        display_name = format_player_display_name(player_name)
        
        if metric == "success_rate":
            metric_display = f"{player_row[metric]:.1f}%"
        else:
            metric_display = f"{player_row[metric]:.2f}"

        ax.text(
            60,
            75,
            display_name,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=BLUES["white"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BLUES["primary"], alpha=0.9, edgecolor=BLUES["white"]),
        )
        ax.text(
            60,
            4,
            f"{metric_label}\n{metric_display}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=BLUES["white"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BLUES["primary"], alpha=0.9, edgecolor=BLUES["white"]),
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return fig
