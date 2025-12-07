"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Distribution plots for directional analysis and multi-metric dispersion
             visualizations using Plotly and Matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Optional
from src.core.colors import BLUES, ACCENTS
from .utils import normalize_name, format_player_display_name


def create_directional_distribution(zone_df: pd.DataFrame):
    """Donut chart of left/center/right progression counts with blue theme."""
    if "progression_zone" not in zone_df.columns:
        return None

    left_count = zone_df["progression_zone"].isin([1, 4, 7, 10]).sum()
    center_count = zone_df["progression_zone"].isin([2, 5, 8, 11]).sum()
    right_count = zone_df["progression_zone"].isin([3, 6, 9, 12]).sum()

    total = left_count + center_count + right_count
    if total == 0:
        return None

    colors_pie = [
        BLUES["sky"],
        BLUES["primary"],
        BLUES["argentina"],
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Left Wing", "Center", "Right Wing"],
                values=[left_count, center_count, right_count],
                marker=dict(
                    colors=colors_pie,
                    line=dict(color=BLUES["navy"], width=3)
                ),
                textinfo="label+percent+value",
                textfont=dict(size=15, color="#FFFFFF", family="Inter"),
                textposition="auto",
                hole=0.45,
                hovertemplate="<b>%{label}</b><br>Progressions: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Directional Distribution of Progressions",
            font=dict(size=18, color=BLUES["white"], family="Inter"),
            x=0.5,
            xanchor="center"
        ),
        height=550,
        font=dict(size=13, color=BLUES["white"], family="Inter"),
        showlegend=True,
        plot_bgcolor=BLUES["navy"],
        paper_bgcolor=BLUES["navy"],
        legend=dict(
            font=dict(color=BLUES["white"]),
            orientation="h",
            x=0.5,
            y=-0.08,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(11,19,43,0.6)",
            bordercolor=BLUES["primary"],
            borderwidth=1
        ),
        annotations=[dict(
            text=f'Total<br>{total}',
            x=0.5, y=0.5,
            font=dict(size=20, color=BLUES["white"], family="Inter"),
            showarrow=False
        )]
    )

    return fig


def create_multi_metric_dispersion(df: pd.DataFrame,
                                   metrics: List[str],
                                   metric_labels: List[str],
                                   highlight_players: List[str],
                                   player_label_map: Optional[Dict[str, str]] = None) -> plt.Figure:
    """
    Multi-metric horizontal dispersion in a single PNG-style Matplotlib figure.

    - Robust name matching ensures Enzo is always identified even without accents.
    - EXPLICITLY removes all borders, frames, and spines to prevent blue boxes.
    """
    if df.empty or not metrics:
        return plt.figure()

    bg = BLUES["navy"]
    pop_color = BLUES["primary"]
    highlight_players = highlight_players or []

    enzo_key = normalize_name("Enzo Fernández")
    comparison_keys = {normalize_name(p) for p in highlight_players if normalize_name(p) != enzo_key}

    df = df.copy()
    df["_norm_player"] = df["player"].apply(normalize_name)

    n_rows = len(metrics)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12, 2.6 * n_rows + 1.4),
        sharex=False,
        dpi=140,
        constrained_layout=False
    )
    if n_rows == 1:
        axes = [axes]

    fig.patch.set_facecolor(bg)

    def two_line_label(lbl: str) -> str:
        parts = str(lbl).split()
        if len(parts) <= 1:
            return lbl
        first = " ".join(parts[:-1])
        second = parts[-1]
        return first + "\n" + second

    for ax, metric, label in zip(axes, metrics, metric_labels):
        if metric not in df.columns:
            ax.set_axis_off()
            continue

        ax.set_frame_on(False)

        ax.set_facecolor("none")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis='both', colors=BLUES["white"], labelsize=10, length=0)

        x_vals = df[metric].astype(float)
        x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
        pad = (x_max - x_min) * 0.08 if np.isfinite(x_max - x_min) else 1.0
        ax.set_xlim(x_min - pad, x_max + pad)

        is_enzo = df["_norm_player"] == enzo_key
        is_comparison = df["_norm_player"].isin(comparison_keys)
        is_other = ~(is_enzo | is_comparison)

        if is_other.any():
            ax.scatter(
                df.loc[is_other, metric],
                np.zeros(is_other.sum()),
                s=26,
                color=pop_color,
                alpha=0.35,
                edgecolors='none',
                label="Other players" if ax is axes[0] else None,
                zorder=1
            )

        # Plot individual comparison players with distinct colors
        comp_colors = [ACCENTS["gold"], ACCENTS["coral"], BLUES["ice"], "#E056FD", "#20C997"] # Expanded palette
        
        for i, player_name in enumerate(highlight_players):
            p_key = normalize_name(player_name)
            if p_key == enzo_key: 
                continue # Skip Enzo here, handled separately
                
            p_mask = df["_norm_player"] == p_key
            if p_mask.any():
                # Get formatted name
                disp_name = format_player_display_name(player_name)
                c = comp_colors[i % len(comp_colors)]
                
                ax.scatter(
                    df.loc[p_mask, metric],
                    np.zeros(p_mask.sum()),
                    s=90,
                    color=c,
                    alpha=1.0,
                    edgecolors=BLUES["white"],
                    linewidth=2,
                    label=disp_name if ax is axes[0] else None,
                    zorder=3
                )

        if is_enzo.any():
            ax.scatter(
                df.loc[is_enzo, metric],
                np.zeros(is_enzo.sum()),
                s=130,
                color=BLUES["argentina"],
                alpha=1.0,
                edgecolors=BLUES["white"],
                linewidth=2.5,
                label="Enzo Fernández" if ax is axes[0] else None,
                zorder=4
            )

        mean_val = np.nanmean(x_vals)
        ax.axvline(mean_val, color=BLUES["primary"], linestyle='--', linewidth=1, alpha=0.7, zorder=0.5)

        lbl_text = two_line_label(label)
        ax.text(
            -0.06, 0.5,
            lbl_text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            color=BLUES["white"],
            fontsize=12,
            fontweight="bold"
        )

        ax.set_yticks([])
        ax.grid(False)

        if ax is not axes[-1]:
            ax.set_xticklabels([])

    fig.text(0.18, 0.05, "← Worse", color=BLUES["white"], fontsize=10, ha="left", va="center")
    fig.text(0.5, 0.05, "Average", color=BLUES["white"], fontsize=10, ha="center", va="center")
    fig.text(0.82, 0.05, "Better →", color=BLUES["white"], fontsize=10, ha="right", va="center")

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)
    fig.legend(
        unique_handles, unique_labels,
        loc="lower center",
        ncol=max(2, len(unique_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=10,
        labelcolor=BLUES["white"]
    )

    plt.subplots_adjust(
        left=0.30,
        right=0.98,
        top=0.97,
        bottom=0.12 + 0.02 * len(highlight_players)
    )

    return fig
