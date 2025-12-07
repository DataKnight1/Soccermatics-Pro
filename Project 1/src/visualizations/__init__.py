"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Visualization package for soccer analytics. Contains specialized modules for
             different visualization types (pitch heatmaps, radar charts, scatter plots, etc.)
"""

from .utils import (
    get_zone_centers,
    normalize_name,
    wrap_label,
    add_orientation_guides,
    format_player_display_name,
    calculate_percentile
)
from .heatmaps import create_zone_heatmap_pitch, create_difference_heatmap
from .defensive_maps import create_defensive_activity_map
from .radar_charts import (
    create_single_pizza,
    create_comparison_pizza,
    create_percentile_bars
)
from .progression_maps import (
    create_pitch_with_progressions,
    create_premium_route_map
)
from .scatter_plots import (
    create_quadrant_scatter,
    create_quadrant_scatter_mpl,
    create_dispersion_plot,
    create_dispersion_plot_mpl,
    create_metric_pitch_comparison
)
from .distribution_plots import (
    create_directional_distribution,
    create_multi_metric_dispersion
)
from .comparison_tables import create_percentile_heatmap_table

__all__ = [
    'get_zone_centers',
    'normalize_name',
    'wrap_label',
    'add_orientation_guides',
    'format_player_display_name',
    'calculate_percentile',
    'create_zone_heatmap_pitch',
    'create_difference_heatmap',
    'create_defensive_activity_map',
    'create_single_pizza',
    'create_comparison_pizza',
    'create_percentile_bars',
    'create_pitch_with_progressions',
    'create_premium_route_map',
    'create_quadrant_scatter',
    'create_quadrant_scatter_mpl',
    'create_dispersion_plot',
    'create_dispersion_plot_mpl',
    'create_metric_pitch_comparison',
    'create_directional_distribution',
    'create_multi_metric_dispersion',
    'create_percentile_heatmap_table',
]
