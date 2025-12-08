"""
Author: Tiago.Monteiro
Date: 2025-12-07
Description: Streamlit application for Enzo Fernández World Cup 2022 deep progression analysis.
             Features tactical profiling, comparative analysis, and interactive visualizations.

Usage:
    streamlit run app.py

Requirements:
    - output/insights/connector_metrics_all_cms.csv
    - output/insights/enzo_zone_progressions.csv
    - output/insights/enzo_defensive_actions.csv
    - output/insights/population_defensive_density_p90.csv
    - output/insights/population_progression_density_p90.csv
    - output/figures/*.png
    - output/metrics/player_metrics.csv
"""

import warnings
from pathlib import Path
from typing import List, Dict, Optional
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from src.visualizations import (
    create_zone_heatmap_pitch,
    create_difference_heatmap,
    create_defensive_activity_map,
    create_single_pizza,
    create_comparison_pizza,
    create_pitch_with_progressions,
    create_premium_route_map,
    create_directional_distribution,
    create_quadrant_scatter_mpl,
    create_multi_metric_dispersion,
    create_percentile_heatmap_table,
)
from src.analysis import DefensiveActionExtractor, calculate_similarity
from src.core.colors import BLUES, ACCENTS, get_categorical_colors

warnings.filterwarnings("ignore")


DEFAULT_TARGET_PLAYER = "Enzo Fernández"
BASE_DIR = Path(__file__).parent
CONNECTOR_CSV = BASE_DIR / "output/insights/connector_metrics_all_cms.csv"
PLAYER_METRICS_CSV = BASE_DIR / "output/metrics/player_metrics.csv"
ZONE_CSV = BASE_DIR / "output/insights/enzo_zone_progressions.csv"
ENZO_DEF_CSV = BASE_DIR / "output/insights/enzo_defensive_actions.csv"
POP_DEF_DENSITY_CSV = BASE_DIR / "output/insights/population_defensive_density_p90.csv"
POP_PROG_DENSITY_CSV = BASE_DIR / "output/insights/population_progression_density_p90.csv"

FIGURES_DIR = BASE_DIR / "output/figures"

st.set_page_config(
    page_title="Enzo Fernández Deep Progression Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Minimalist Dark Blue Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
        color: #FFFFFF;
        background-color: #0B132B;
    }

    /* Clean Headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #75AADB !important; /* Argentina Blue */
    }

    /* Strong/Bold Text */
    strong {
        color: #A8D8EA !important; /* Sky Blue */
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: #75AADB !important;
    }
    [data-testid="stMetricLabel"] {
        color: #A8D8EA !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #080E21;
        border-right: 1px solid #1D5D9B;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] .stCaption {
        color: #FFFFFF !important;
    }
    .css-1d391kg {
        background-color: #080E21;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0F3D5E;
        padding: 0.5rem;
        border-radius: 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #A8D8EA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1D5D9B;
        color: #FFFFFF !important;
    }

    /* Remove white background from datatables if any */
    .stDataFrame {
        background-color: #0B132B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_connector_data() -> Optional[pd.DataFrame]:
    """Load connector metrics."""
    if not CONNECTOR_CSV.exists():
        st.error(
            f"Could not find connector metrics at `{CONNECTOR_CSV}`. "
            "Please run `python run_analysis.py` first."
        )
        return None

    try:
        df = pd.read_csv(CONNECTOR_CSV)
        return df
    except Exception as e:
        st.error(f"Error loading connector metrics: {e}")
        return None


@st.cache_data
def load_player_metrics() -> Optional[pd.DataFrame]:
    """Load general player metrics (passing, defense, creation)."""
    if not PLAYER_METRICS_CSV.exists():
        st.warning(f"Could not find player metrics at `{PLAYER_METRICS_CSV}`. Using limited data.")
        return None
    try:
        return pd.read_csv(PLAYER_METRICS_CSV)
    except Exception as e:
        st.warning(f"Error loading player metrics: {e}")
        return None


@st.cache_data
def load_zone_data() -> Optional[pd.DataFrame]:
    """Load Enzo's zone progression data."""
    if not ZONE_CSV.exists():
        return None
    try:
        print(f"Loading zone data from {ZONE_CSV}...")
        df = pd.read_csv(ZONE_CSV)
        if 'type' not in df.columns:
            st.error(f"Critical: 'type' column missing from {ZONE_CSV}")
        return df
    except Exception as e:
        st.warning(f"Could not load zone data: {e}")
        return None

@st.cache_data
def load_defensive_data() -> Optional[pd.DataFrame]:
    """Load Enzo's defensive actions."""
    if not ENZO_DEF_CSV.exists():
        return None
    try:
        return pd.read_csv(ENZO_DEF_CSV)
    except Exception as e:
        return None

@st.cache_data
def load_population_def_density() -> Optional[pd.Series]:
    """Load population defensive density (p90)."""
    if not POP_DEF_DENSITY_CSV.exists():
        return None
    try:
        df = pd.read_csv(POP_DEF_DENSITY_CSV, index_col=0)
        return df['p90']
    except Exception as e:
        return None

@st.cache_data
def load_population_prog_density() -> Optional[pd.Series]:
    """Load population progression density (p90)."""
    if not POP_PROG_DENSITY_CSV.exists():
        return None
    try:
        df = pd.read_csv(POP_PROG_DENSITY_CSV, index_col=0)
        return df['p90']
    except Exception as e:
        return None

def find_target_player(df: pd.DataFrame, default_name: str = DEFAULT_TARGET_PLAYER) -> pd.Series:
    """Robust helper to find the target player row."""
    def normalize(name: str) -> str:
        return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower().strip()

    target_norm = normalize(default_name)
    player_norm = df["player"].apply(normalize)

    exact_mask = player_norm == target_norm
    if exact_mask.any():
        return df[exact_mask].iloc[0]

    tokens = target_norm.split()
    if len(tokens) >= 2:
        contains_mask = player_norm.str.contains(tokens[0], na=False)
        contains_mask &= player_norm.str.contains(tokens[-1], na=False)
        if contains_mask.any():
            return df[contains_mask].iloc[0]

    enzo_mask = player_norm.str.contains("enzo", na=False)
    if enzo_mask.any():
        return df[enzo_mask].iloc[0]

    st.error(f"Target player '{default_name}' not found in connector metrics.")
    st.stop()


connector_df = load_connector_data()
if connector_df is None:
    st.stop()

player_metrics_df = load_player_metrics()
if player_metrics_df is not None:
    cols_to_use = player_metrics_df.columns.difference(connector_df.columns)

    cols_to_merge = [c for c in player_metrics_df.columns if c not in connector_df.columns or c == 'player']

    connector_df = pd.merge(
        connector_df,
        player_metrics_df[cols_to_merge],
        on='player',
        how='left'
    )

zone_df = load_zone_data()
def_actions_df = load_defensive_data()
pop_def_density = load_population_def_density()
pop_prog_density = load_population_prog_density()

if 'rank' not in connector_df.columns:
    if 'deep_progressions_p90' in connector_df.columns:
        connector_df['rank'] = connector_df['deep_progressions_p90'].rank(ascending=False, method='min')
    else:
        connector_df['rank'] = 0

if 'success_rate' not in connector_df.columns and 'pass_completion_%' in connector_df.columns:
    connector_df['success_rate'] = connector_df['pass_completion_%']

enzo_row = find_target_player(connector_df, DEFAULT_TARGET_PLAYER)
enzo_name = enzo_row["player"]

if zone_df is not None and not zone_df.empty:
    valid_receptions = zone_df.dropna(subset=['reception_zone'])
    if not valid_receptions.empty:
        try:
            primary_zone = int(valid_receptions['reception_zone'].mode().iloc[0])
            enzo_row['primary_reception_zone'] = primary_zone
        except Exception:
            pass

    progs = zone_df[zone_df['type'].isin(['Pass', 'Carry'])]
    if not progs.empty:
        right_count = progs['progression_zone'].isin([3, 6, 9, 12]).sum()
        total_count = len(progs)
        if total_count > 0:
            right_pct = (right_count / total_count) * 100.0
            enzo_row['right_side_pct'] = right_pct



st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "Overview",
        "Tactical Profile",
        "Comparative Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{enzo_name}**")
st.sidebar.caption(f"{enzo_row.get('team', 'Unknown')} • {enzo_row.get('total_minutes', enzo_row.get('minutes', 0)):.0f} mins")
st.sidebar.markdown(f"**Deep Progressions p90:** {enzo_row.get('deep_progressions_p90', 0.0):.2f}")
st.sidebar.markdown(f"**Pass Completion:** {enzo_row.get('success_rate', 0.0):.1f}%")
st.sidebar.markdown(f"**Prog Actions p90:** {(enzo_row.get('prog_passes_p90', 0) + enzo_row.get('prog_carries_p90', 0)):.1f}")
st.sidebar.markdown(f"**Recoveries p90:** {enzo_row.get('recoveries_p90', 0.0):.1f}")
st.sidebar.markdown(
    f"**Rank:** {enzo_row.get('rank', 0):.0f}/{connector_df['rank'].max():.0f}"
)


if page == "Overview":
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1D5D9B 0%, #0B132B 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 2px solid #75AADB;'>
        <h2 style='color: #FFFFFF; margin: 0; font-size: 24px;'>{enzo_name}</h2>
        <p style='color: #A8D8EA; margin: 5px 0 0 0; font-size: 14px;'>
            World Cup 2022 • Deep Progression Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Research Question")
    st.markdown(
        """
        *Was Enzo Fernández Argentina's primary link between defense and attack, and how did he create this connection differently from other central midfielders?*

        **Answer: Yes, uniquely.**
        """
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Global Rank", f"{enzo_row['rank']:.0f}/{connector_df['rank'].max():.0f}")
    with col2:
        st.metric("Volume (p90)", f"{enzo_row['deep_progressions_p90']:.2f}")
    with col3:
        st.metric("Right Bias", f"{enzo_row.get('right_side_pct', 0):.1f}%")
    with col4:
        st.metric("Primary Hub", f"Zone {enzo_row.get('primary_reception_zone', 'N/A')}")

    st.markdown("### Distinctive Patterns")

    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown("**Elite Global Status**")
        st.markdown(f"Ranked #{enzo_row['rank']:.0f} of {connector_df['rank'].max():.0f} midfielders. Combines high reliability with high volume.")

    with p_col2:
        st.markdown("**Unique Deep Hub**")
        st.markdown(f"Primary reception in Zone {enzo_row.get('primary_reception_zone', 'N/A')}. Operates deeper than typical playmakers.")

    with p_col3:
        st.markdown("**Right-Sided Bias**")
        st.markdown(f"{enzo_row.get('right_side_pct', 0):.1f}% directionality to the right. Facilitates play to Messi/Molina.")

    st.markdown("---")


    r_col1, r_col2 = st.columns([1, 1.2])

    with r_col1:
        st.subheader("Performance Profile")

        potential_metrics = [
            "deep_progressions_p90", "prog_passes_p90", "prog_carries_p90",
            "pass_completion_%", "passes_final_third_p90",
            "key_passes_p90", "passes_into_box_p90",
            "tackles_interceptions_p90", "recoveries_p90", "pressures_p90"
        ]

        potential_labels = [
            "Deep Progs", "Prog Passes", "Prog Carries",
            "Pass Comp %", "Final 3rd Pass",
            "Key Passes", "Into Box Pass",
            "Tkl+Int", "Recoveries", "Pressures"
        ]

        valid_metrics = []
        valid_labels = []
        for m, l in zip(potential_metrics, potential_labels):
            if m in connector_df.columns:
                valid_metrics.append(m)
                valid_labels.append(l)

        if not valid_metrics:
            valid_metrics = ["deep_progressions_p90", "success_rate", "deep_receptions_p90"]
            valid_labels = ["Deep Progressions", "Success Rate", "Deep Receptions"]

        fig_pizza = create_single_pizza(enzo_row, connector_df, valid_metrics, valid_labels, enzo_name)
        st.pyplot(fig_pizza)

    with r_col2:
        st.subheader("Top 10 Connectors")

        def clean_name(name):
            name_map = {
                "Bernardo Mota Veiga de Carvalho e Silva": "Bernardo Silva",
                "Pedro González López": "Pedri",
                "Federico Santiago Valverde Dipetta": "Federico Valverde",
                "Yunus Dimoara Musah": "Yunus Musah",
                "Frenkie de Jong": "Frenkie de Jong",
                "Mateo Kovačić": "Mateo Kovačić",
                "Joshua Kimmich": "Joshua Kimmich",
                "Luka Modrić": "Luka Modrić",
                "Weston McKennie": "Weston McKennie",
                "Enzo Fernandez": "Enzo Fernández",
                "Enzo Jeremías Fernández": "Enzo Fernández"
            }
            return name_map.get(name, name)

        top_10 = connector_df.nsmallest(10, "rank")[
            ["rank", "player", "team", "deep_progressions_p90", "success_rate"]
        ].copy()

        top_10["player"] = top_10["player"].apply(clean_name)

        top_10.columns = ["#", "Player", "Team", "Prog/90", "Success %"]

        def highlight_enzo(row):
            if "Enzo" in row["Player"]:
                return ["background-color: #E6F3FF; color: #1D5D9B; font-weight: bold; border-left: 4px solid #1D5D9B"] * len(row)
            return [""] * len(row)

        formatted_df = top_10.style.format({
            "Prog/90": "{:.2f}",
            "Success %": "{:.1f}%"
        }).apply(highlight_enzo, axis=1)

        st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(format="%d", width="small"),
                "Player": st.column_config.TextColumn(width="medium"),
                "Team": st.column_config.TextColumn(width="medium"),
                "Prog/90": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=top_10["Prog/90"].max(), width="medium"),
                "Success %": st.column_config.TextColumn(width="small")
            }
        )


elif page == "Tactical Profile":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1D5D9B 0%, #0B132B 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 2px solid #75AADB;'>
        <h2 style='color: #FFFFFF; margin: 0; font-size: 24px;'>Tactical Profile</h2>
        <p style='color: #A8D8EA; margin: 5px 0 0 0; font-size: 14px;'>
            Spatial analysis of progression and defensive actions
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_prog, tab_def = st.tabs(["Deep Progression", "Defensive Analysis"])

    with tab_prog:
        st.markdown("### Enzo's Progression on the Pitch")

        if zone_df is None or zone_df.empty:
            st.warning("No zone data available for Enzo. Please regenerate analysis.")
        else:
            subtab1, subtab2, subtab3, subtab4 = st.tabs(
                ["Reception Heatmap", "Progression Heatmap", "Distinctive Patterns (Bias)", "Progression Routes"]
            )

            with subtab1:
                st.markdown("**Where Enzo receives the ball before progressing**")
                receipts_df = zone_df[zone_df['type'].str.contains("Ball Receipt", case=False, na=False)]
                fig = create_zone_heatmap_pitch(
                    receipts_df,
                    "reception_zone",
                    "Enzo Fernández: Deep Reception Zones (WC 2022)",
                    cmap="reception",
                )
                st.pyplot(fig)

            with subtab2:
                st.markdown("**Where Enzo progresses the ball to (target zones)**")
                prog_df = zone_df[zone_df['type'].isin(['Pass', 'Carry'])]
                fig = create_zone_heatmap_pitch(
                    prog_df,
                    "progression_zone",
                    "Enzo Fernández: Deep Progression Target Zones (WC 2022)",
                    cmap="progression",
                )
                st.pyplot(fig)

            with subtab3:
                st.markdown("**Enzo's Directional Bias Compared to Average Midfielder**")
                st.info("Light Blue = Uses this zone MORE than average | Dark Blue = Uses this zone LESS than average")

                if pop_prog_density is not None:
                    mins = enzo_row.get("minutes", enzo_row.get("total_minutes", 0.0))

                    prog_df = zone_df[zone_df['type'].isin(['Pass', 'Carry'])]
                    recs = prog_df['progression_zone'].value_counts()

                    recs.index = recs.index.astype(int)

                    for z in range(1, 13):
                        if z not in recs: recs[z] = 0
                    recs = recs.sort_index()

                    if mins > 0:
                        enzo_prog_p90 = recs / (mins / 90.0)

                        pop_prog_density.index = pop_prog_density.index.astype(int)

                        fig = create_difference_heatmap(
                            enzo_prog_p90,
                            pop_prog_density,
                            "Difference Map: Enzo - Average CM (Target Zones)"
                        )
                        st.pyplot(fig)

                        st.markdown("""
                        **Interpretation:**
                        - **Right-Sided Bias**: Lighter blues on the right flank = more use than the typical midfielder; darker blues = less use.
                        - **Deep Safety**: Lighter blues in central deep zones = frequent resets/retention; darker blues = less use there.
                        """)
                    else:
                        st.warning("Enzo's minutes not found, cannot normalize.")
                else:
                    st.warning("Population progression density not available. Please run analysis.")

            with subtab4:
                st.markdown("**Top 10 most frequent zone-to-zone routes**")
                prog_df = zone_df[zone_df['type'].isin(['Pass', 'Carry'])]
                fig = create_premium_route_map(
                    prog_df, "Enzo Fernández: Top 10 Progression Routes"
                )
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("Directional Distribution")
            fig_pie = create_directional_distribution(zone_df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab_def:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1D5D9B 0%, #0B132B 100%);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border: 2px solid #75AADB;'>
            <h2 style='color: #FFFFFF; margin: 0; font-size: 24px;'>Defensive Analysis</h2>
            <p style='color: #A8D8EA; margin: 5px 0 0 0; font-size: 14px;'>
                Understanding Enzo's Defensive Contribution
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            ### Defensive Metrics
            While mainly a playmaker, Enzo's defensive recovery work was crucial to Argentina's balance.

            **Key Actions:**
            - **Interceptions**: Cutting off opponent passes
            - **Duels**: Direct physical engagement
            """)

            if def_actions_df is not None:
                n_interceptions = len(def_actions_df[def_actions_df['type_name'] == 'Interception'])
                n_duels = len(def_actions_df[def_actions_df['type_name'] == 'Duel'])
                n_total = len(def_actions_df)

                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Actions", n_total)
                with col_b:
                    st.metric("Duels", n_duels)
                with col_c:
                    st.metric("Interceptions", n_interceptions)
            else:
                st.warning("Defensive data not available")

        with col2:
            if def_actions_df is not None:
                fig = create_defensive_activity_map(
                    def_actions_df,
                    "Enzo Fernández: Defensive Activity Map"
                )
                st.pyplot(fig)
            else:
                st.warning("Defensive data not found.")

        st.markdown("---")
        st.markdown("### Comparative Recovery Patterns")

        if def_actions_df is not None and pop_def_density is not None:
            st.markdown("**Where does Enzo recover the ball compared to the average midfielder?**")
            st.info("Light Blue = MORE recoveries than average | Dark Blue = FEWER recoveries than average")

            ex = DefensiveActionExtractor(pd.DataFrame())

            rec_only = def_actions_df[def_actions_df['type_name'] == 'Ball Recovery']
            counts = ex.calculate_zone_counts(rec_only)

            mins = enzo_row.get("minutes", enzo_row.get("total_minutes", 0.0))
            if mins > 0:
                enzo_def_p90 = counts / (mins / 90.0)

                fig_diff = create_difference_heatmap(
                    enzo_def_p90,
                    pop_def_density,
                    "Difference Map: Enzo - Average CM (Defensive Actions)"
                )
                st.pyplot(fig_diff)

                st.markdown("""
                **Analysis Guide:**
                - **Central Dominance**: Look for **light blue** in central zones (5, 8). This indicates a strong engine room presence with more recoveries than average.
                - **Flank Support**: **Light blue** in wide areas suggests high work rate to cover fullbacks.
                - **Dark blue** zones show fewer recoveries than the average midfielder, indicating less defensive activity in those areas.
                """)
            else:
                st.error("Cannot calculate p90 (0 minutes).")
        else:
            st.warning("Comparison data missing. Run analysis to generate population density.")


elif page == "Comparative Analysis":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1D5D9B 0%, #0B132B 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 2px solid #75AADB;'>
        <h2 style='color: #FFFFFF; margin: 0; font-size: 24px;'>Comparative Analysis</h2>
        <p style='color: #A8D8EA; margin: 5px 0 0 0; font-size: 14px;'>
            Player comparisons, rankings, and statistical insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    if connector_df is None or connector_df.empty:
        st.error("No data available. Please run the analysis first: `python run_analysis.py`")
        st.stop()

    all_players = connector_df["player"].unique().tolist()

    if 'selected_players' not in st.session_state:
        default_selections = ["Enzo Fernandez", "Luka Modrić", "Rodrigo De Paul", "Pedri"]
        default_selections = [p for p in default_selections if p in all_players]

        if not default_selections and len(all_players) >= 4:
            default_selections = all_players[:4]
        elif not default_selections and len(all_players) > 0:
            default_selections = all_players[:min(len(all_players), 2)]

        st.session_state.selected_players = default_selections

    st.markdown("### Select Players to Compare / Highlight")
    selected_players = st.multiselect(
        "Choose players (Max 2 for Pizza Comparison, All for Scatter Highlight):",
        all_players,
        default=st.session_state.selected_players,
        key='player_multiselect'
    )

    st.session_state.selected_players = selected_players

    tab_comp, tab_rank, tab_sim = st.tabs(["Player Comparison", "Rankings", "Similar Players"])

    with tab_comp:
        st.markdown("### Compare Enzo to Other Midfielders")

        if selected_players:
            try:
                potential_metrics = [
                    "deep_progressions_p90", "prog_passes_p90", "prog_carries_p90",
                    "pass_completion_%", "passes_final_third_p90",
                    "key_passes_p90", "passes_into_box_p90",
                    "tackles_interceptions_p90", "recoveries_p90", "pressures_p90"
                ]
                potential_labels = [
                    "Deep Progs", "Prog Passes", "Prog Carries",
                    "Pass Comp %", "Final 3rd Pass",
                    "Key Passes", "Into Box Pass",
                    "Tkl+Int", "Recoveries", "Pressures"
                ]

                comp_metrics = []
                comp_labels = []
                for m, l in zip(potential_metrics, potential_labels):
                    if m in connector_df.columns:
                        comp_metrics.append(m)
                        comp_labels.append(l)

                if not comp_metrics:
                    comp_metrics = ["deep_progressions_p90", "success_rate", "deep_receptions_p90"]
                    comp_labels = ["Deep Progressions", "Success Rate", "Deep Receptions"]

                if len(selected_players) >= 2:
                    st.info(f"Comparing: {selected_players[0]} vs {selected_players[1]}")
                elif len(selected_players) == 1:
                    st.info(f"Showing: {selected_players[0]}")

                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                plt.close('all')

                current_selection = list(selected_players)

                fig_pizza = create_comparison_pizza(
                    connector_df, current_selection, connector_df,
                    metrics=comp_metrics, labels=comp_labels
                )

                chart_placeholder = st.empty()
                with chart_placeholder:
                    st.pyplot(fig_pizza)

                plt.close(fig_pizza)

                st.markdown("### Detailed Metrics Comparison")

                st.markdown("#### Percentile Rankings Heatmap")
                st.info("Compare selected players' percentile rankings across key metrics (vs all midfielders)")

                comp_df = connector_df[connector_df['player'].isin(selected_players)].copy()

                heatmap_metrics = [
                    'deep_progressions_p90',
                    'pass_completion_%',
                    'deep_receptions_p90',
                    'prog_passes_p90',
                    'prog_carries_p90',
                    'key_passes_p90',
                    'passes_final_third_p90',
                    'recoveries_p90',
                    'tackles_interceptions_p90',
                    'pressures_p90'
                ]

                heatmap_labels = [
                    'Deep Progressions (p90)',
                    'Success Rate (%)',
                    'Deep Receptions (p90)',
                    'Progressive Passes (p90)',
                    'Progressive Carries (p90)',
                    'Key Passes (p90)',
                    'Final Third Passes (p90)',
                    'Recoveries (p90)',
                    'Tackles + Interceptions (p90)',
                    'Pressures (p90)'
                ]

                available_heatmap_pairs = [
                    (m, l) for m, l in zip(heatmap_metrics, heatmap_labels)
                    if m in connector_df.columns
                ]

                if available_heatmap_pairs and not comp_df.empty:
                    avail_metrics = [m for m, _ in available_heatmap_pairs]
                    avail_labels = [l for _, l in available_heatmap_pairs]

                    plt.close('all')
                    fig_heatmap = create_percentile_heatmap_table(
                        selected_players_df=comp_df,
                        all_players_df=connector_df,
                        metrics=avail_metrics,
                        metric_labels=avail_labels,
                        title=f"Player Comparison - Percentile Rankings"
                    )
                    st.pyplot(fig_heatmap)
                    plt.close(fig_heatmap)
                else:
                    st.warning("Not enough data to generate percentile heatmap.")

                st.markdown("---")
                st.markdown("#### Raw Values Comparison")

                display_cols = ['player', 'team', 'deep_progressions_p90', 'success_rate', 'deep_receptions_p90']

                optional_metrics = ['prog_passes_p90', 'prog_carries_p90', 'key_passes_p90',
                                   'passes_final_third_p90', 'recoveries_p90',
                                   'tackles_interceptions_p90', 'pressures_p90', 'passes_into_box_p90']
                for col in optional_metrics:
                    if col in comp_df.columns:
                        display_cols.append(col)

                display_cols = [col for col in display_cols if col in comp_df.columns]
                comp_display = comp_df[display_cols].copy()

                numeric_cols = comp_display.select_dtypes(include=[np.number]).columns

                def style_dataframe(df):
                    format_dict = {}
                    for col in numeric_cols:
                        if col in df.columns:
                            if col == 'success_rate':
                                format_dict[col] = "{:.1f}%"
                            else:
                                format_dict[col] = "{:.3f}"

                    styled = df.style.background_gradient(
                        cmap='Blues',
                        subset=[col for col in numeric_cols if col in df.columns]
                    ).format(format_dict).set_properties(**{
                        'text-align': 'center',
                        'font-size': '13px',
                        'border': '1px solid #75AADB'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#1D5D9B'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('font-size', '14px')
                        ]}
                    ])
                    return styled

                st.dataframe(style_dataframe(comp_display), use_container_width=True, height=250)
            except Exception as e:
                st.error(f"Error creating comparison: {e}")
                st.info("Try selecting different players or check if all required data columns exist.")
        else:
            st.info("Please select at least one player to compare.")

    with tab_rank:
        st.markdown("### Statistical Rankings")

        try:
            st.subheader("Efficiency Quadrant: Volume vs Precision")
            st.info("Top-Right = High Volume & High Success (Elite Connectors)")

            import matplotlib
            matplotlib.use('Agg')
            plt.close('all')

            highlight_list = list(selected_players) if selected_players else []
            if "Enzo Fernández" not in highlight_list:
                highlight_list.append("Enzo Fernández")

            fig_quad = create_quadrant_scatter_mpl(
                connector_df,
                x_col='deep_progressions_p90',
                y_col='success_rate',
                x_label='Deep Progressions per 90',
                y_label='Success Rate (%)',
                title='Deep Progression Efficiency: Highlights',
                highlight_players=highlight_list
            )
            st.pyplot(fig_quad)
            plt.close(fig_quad)

            st.subheader("Metric Distribution (All Metrics)")
            st.info("Visualizing where selected players fall in the overall distribution across multiple metrics.")

            metrics_to_show = ['deep_progressions_p90', 'success_rate', 'deep_receptions_p90',
                              'prog_passes_p90', 'key_passes_p90']
            metric_labels = ['Deep Progressions p90', 'Success Rate (%)', 'Deep Receptions p90',
                           'Progressive Passes p90', 'Key Passes p90']

            available_pairs = [(m, l) for m, l in zip(metrics_to_show, metric_labels)
                              if m in connector_df.columns]

            if available_pairs:
                avail_metrics = [m for m, _ in available_pairs]
                avail_labels = [l for _, l in available_pairs]

                plt.close('all')
                fig_multi_disp = create_multi_metric_dispersion(
                    connector_df,
                    metrics=avail_metrics,
                    metric_labels=avail_labels,
                    highlight_players=highlight_list
                )
                st.pyplot(fig_multi_disp)
                plt.close(fig_multi_disp)
            else:
                st.warning("Required metrics not found in data.")

        except Exception as e:
            st.error(f"Error displaying rankings: {e}")
            st.info("Please ensure the data file contains all required columns.")

    with tab_sim:
        st.markdown("### Similar Player Analysis")
        st.info("Identifying players with the most similar statistical profile to Enzo using Z-Score Euclidean Distance.")

        try:
            sim_metrics = [
                'deep_progressions_p90', 'prog_passes_p90', 'prog_carries_p90',
                'pass_completion_%', 'passes_final_third_p90', 'recoveries_p90',
                'key_passes_p90', 'long_balls_p90'
            ]
            sim_metrics = [m for m in sim_metrics if m in connector_df.columns]

            similar_df = calculate_similarity(connector_df, enzo_name, sim_metrics, top_n=5)

            st.subheader(f"Top 5 Players Most Similar to {enzo_name}")

            disp_sim = similar_df[['player', 'team', 'similarity_distance']].copy()
            disp_sim.columns = ['Player', 'Team', 'Euclidean Dist']

            st.dataframe(
                disp_sim.style.background_gradient(cmap='Greens_r', subset=['Euclidean Dist']).format({'Euclidean Dist': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )

            top_3_sim = similar_df.head(3)['player'].tolist()
            sim_highlight = [enzo_name] + top_3_sim

            st.markdown("---")
            st.subheader("Visual Comparison: Enzo vs Top 3 Twins")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Metric Dispersion**")
                disp_metrics_sim = ['deep_progressions_p90', 'pass_completion_%', 'prog_passes_p90', 'recoveries_p90']
                disp_labels_sim = ['Deep Progs', 'Success %', 'Prog Passes', 'Recoveries']

                plt.close('all')
                fig_sim_disp = create_multi_metric_dispersion(
                    connector_df, disp_metrics_sim, disp_labels_sim,
                    highlight_players=sim_highlight
                )
                st.pyplot(fig_sim_disp)
                plt.close(fig_sim_disp)

            with c2:
                st.markdown("**Percentile Heatmap**")
                hm_metrics_sim = [
                    'deep_progressions_p90', 'pass_completion_%', 'deep_receptions_p90',
                    'prog_passes_p90', 'prog_carries_p90', 'key_passes_p90',
                    'passes_final_third_p90', 'recoveries_p90', 'tackles_interceptions_p90', 'pressures_p90'
                ]
                hm_labels_sim = [
                    'Deep Progressions (p90)', 'Success Rate (%)', 'Deep Receptions (p90)',
                    'Prog Passes (p90)', 'Prog Carries (p90)', 'Key Passes (p90)',
                    'Final Third Passes (p90)', 'Recoveries (p90)', 'Tkl + Int (p90)', 'Pressures (p90)'
                ]

                sim_comp_df = connector_df[connector_df['player'].isin(sim_highlight)].copy()

                sim_comp_df['order'] = pd.Categorical(sim_comp_df['player'], categories=sim_highlight, ordered=True)
                sim_comp_df = sim_comp_df.sort_values('order')

                plt.close('all')
                fig_sim_hm = create_percentile_heatmap_table(
                    sim_comp_df, connector_df, hm_metrics_sim, hm_labels_sim,
                    title="Similarity Profile: Percentiles"
                )
                st.pyplot(fig_sim_hm)
                plt.close(fig_sim_hm)

        except Exception as e:
            st.error(f"Error calculating similarity: {e}")
