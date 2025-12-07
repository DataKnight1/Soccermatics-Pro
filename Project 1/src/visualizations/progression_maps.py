"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Progression and route map visualizations showing zone-to-zone ball movement
             and top progression routes on the pitch.
"""

import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from src.core.colors import BLUES
from .utils import get_zone_centers


def create_pitch_with_progressions(zone_df: pd.DataFrame, title: str):
    """Create pitch with top 10 zone-to-zone progression arrows (Dark Blue Theme)."""
    fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
    fig.patch.set_facecolor("#0B132B")

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#0B132B",
        line_color="#75AADB",
        linewidth=1.5
    )
    pitch.draw(ax=ax)

    zone_centers = get_zone_centers()

    if {"reception_zone", "progression_zone"} - set(zone_df.columns):
        return fig

    zone_flow = pd.crosstab(zone_df["reception_zone"], zone_df["progression_zone"]).stack()
    zone_flow = zone_flow[zone_flow > 0].sort_values(ascending=False).head(10)

    arrow_colors = ["#00F2FF", "#2CE0F4", "#58CDE9", "#85BBDE", "#B1A9D3", "#DE96C8", "#FF84BD"]

    if len(zone_flow) > len(arrow_colors):
        arrow_colors = arrow_colors * ((len(zone_flow) // len(arrow_colors)) + 1)

    colors_arrow = arrow_colors[:len(zone_flow)]

    for idx, ((from_zone, to_zone), count) in enumerate(zone_flow.items()):
        if from_zone in zone_centers and to_zone in zone_centers:
            from_pos = zone_centers[from_zone][:2]
            to_pos = zone_centers[to_zone][:2]

            ax.annotate(
                "",
                xy=to_pos,
                xytext=from_pos,
                arrowprops=dict(
                    arrowstyle="->",
                    lw=4 + (idx == 0) * 1.5,
                    color=colors_arrow[idx],
                    alpha=0.9
                ),
            )

            mid_x = (from_pos[0] + to_pos[0]) / 2.0
            mid_y = (from_pos[1] + to_pos[1]) / 2.0
            ax.text(
                mid_x,
                mid_y,
                f"{int(count)}",
                fontsize=13,
                fontweight="bold",
                color="#FFFFFF",
                bbox=dict(
                    boxstyle="circle,pad=0.4",
                    facecolor="#0B132B",
                    alpha=0.8,
                    edgecolor=colors_arrow[idx],
                    linewidth=2
                ),
            )

    ax.set_title(title, fontsize=18, fontweight="bold", color="#FFFFFF", pad=20)
    plt.tight_layout()
    return fig


def create_premium_route_map(zone_df: pd.DataFrame, title: str):
    """
    Compact visualization for Top 10 Progression Routes with dark-blue theme.
    Uses a standard legend instead of a sidebar for better space utilization.
    """
    if {"reception_zone", "progression_zone"} - set(zone_df.columns):
        return None

    zone_flow = pd.crosstab(zone_df["reception_zone"], zone_df["progression_zone"]).stack()
    zone_flow_df = zone_flow.reset_index(name='count')
    zone_flow_df = zone_flow_df[zone_flow_df["reception_zone"] != zone_flow_df["progression_zone"]]

    zone_flow_df = zone_flow_df.sort_values("count", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BLUES["navy"], dpi=120)
    ax.set_facecolor(BLUES["navy"])

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=BLUES["navy"],
        line_color=BLUES["white"],
        linewidth=1.5,
        corner_arcs=True
    )
    pitch.draw(ax=ax)

    zone_centers = get_zone_centers()
    style_zone = {"color": BLUES["primary"], "linewidth": 0.8, "linestyle": ":", "alpha": 0.25, "zorder": 1}
    for val in [26.67, 53.33]:
        ax.plot([0, 120], [val, val], **style_zone)
    for x in [30, 60, 90]:
        ax.plot([x, x], [0, 80], **style_zone)

    rank_colors = [
        BLUES["argentina"],
        BLUES["sky"],
        BLUES["primary"],
        BLUES["ice"],
        BLUES["argentina"],
    ]

    if len(zone_flow_df) > len(rank_colors):
        rank_colors = rank_colors + [BLUES["primary"]] * (len(zone_flow_df) - len(rank_colors))

    legend_handles = []

    for idx, row in zone_flow_df.reset_index(drop=True).iterrows():
        from_zone = int(row["reception_zone"])
        to_zone = int(row["progression_zone"])
        count = row["count"]

        if from_zone not in zone_centers or to_zone not in zone_centers:
            continue

        color = rank_colors[idx]
        rank = idx + 1

        from_pos = zone_centers[from_zone][:2]
        to_pos = zone_centers[to_zone][:2]

        w_line = 4.0 if idx < 3 else 2.5
        alpha = 0.95 if idx < 3 else 0.7
        m_scale = 25 if idx < 3 else 18

        arrow = FancyArrowPatch(
            from_pos,
            to_pos,
            connectionstyle="arc3,rad=0.15",
            color=color,
            arrowstyle="-|>",
            mutation_scale=m_scale,
            linewidth=w_line,
            alpha=alpha,
            zorder=3
        )
        ax.add_patch(arrow)

        marker_size = 120 if idx < 3 else 80
        ax.scatter(from_pos[0], from_pos[1], color=color, s=marker_size,
                   edgecolors=BLUES["white"], linewidth=1.5, zorder=4, alpha=0.9)

        ax.text(from_pos[0], from_pos[1], str(rank),
                color=BLUES["white"], fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=5)

        handle = Line2D([0], [0], color=color, lw=2,
                        label=f"{rank}: Z{from_zone}→Z{to_zone} ({int(count)})")
        legend_handles.append(handle)

    legend = ax.legend(handles=legend_handles,
                       loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       ncol=1,
                       facecolor=BLUES["navy"],
                       edgecolor=BLUES["argentina"],
                       labelcolor=BLUES["white"],
                       fontsize=10,
                       title="Top Routes",
                       title_fontsize=12,
                       framealpha=1.0)

    plt.setp(legend.get_title(), color=BLUES["white"], fontweight='bold')

    ax.set_title(title, fontsize=18, fontweight="bold", color=BLUES["white"], pad=15)

    plt.tight_layout()
    return fig
