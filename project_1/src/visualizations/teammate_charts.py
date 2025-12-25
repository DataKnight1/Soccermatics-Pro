"""
Author: Tiago.Monteiro
Date: 2025-12-08
Description: Visualizations for teammate connection analysis, including small multiple grids.
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from src.core.colors import BLUES, ACCENTS

def create_teammate_cluster_grid(
    clusters: dict,
    player_name: str,
    title: str = "Key Connections"
):
    """
    Creates a grid of small pitches (Small Multiples), one for each teammate.
    Each pitch shows the pass vectors from the source player to that teammate.
    """
    if not clusters:
        return None

    n_teammates = len(clusters)
    cols = min(n_teammates, 3)
    rows = math.ceil(n_teammates / cols)
    
    # Approx size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), dpi=120)
    fig.patch.set_facecolor("#0B132B")
    
    if n_teammates == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Iterate through teammates and plot
    for idx, (teammate_name, passes_df) in enumerate(clusters.items()):
        ax = axes[idx]
        
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#0B132B",
            line_color="#75AADB",
            linewidth=1.0
        )
        pitch.draw(ax=ax)
        
        # Plot passes
        # Use a distinctive color for the passes (e.g. Argentina Blue or Sky)
        pitch.arrows(
            passes_df.x, passes_df.y,
            passes_df.end_x, passes_df.end_y,
            ax=ax,
            width=2,
            headwidth=3,
            color=BLUES['sky'],
            alpha=0.6,
            label="Pass"
        )
        
        # Scatter for end locations (reception points)
        pitch.scatter(
             passes_df.end_x, passes_df.end_y,
             ax=ax,
             s=20,

             c=BLUES['ice'],
             edgecolors='white',
             linewidth=0.5,
             alpha=0.8
        )
        
        # Add Title & Takeaway
        # Simple clustering logic for "Takeaway" string could be added here or passed in.
        # For now, we use a generic statistic.
        
        count = len(passes_df)
        ax.set_title(f"{teammate_name}", fontsize=14, fontweight="bold", color="white", pad=2)
        ax.text(60, -5, f"{count} Passes", ha='center', va='top', fontsize=10, color=ACCENTS['gold'])

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(title, fontsize=20, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig
