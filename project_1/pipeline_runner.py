import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import unicodedata

from src.data import StatsBombDataLoader, ComparisonGroupBuilder
from src.analysis import MetricsCalculator, StatisticalAnalyzer, calculate_similarity, DefensiveActionExtractor
from src.visualizations import (
    create_single_pizza,
    create_comparison_pizza,
    create_zone_heatmap_pitch,
    create_difference_heatmap,
    create_premium_route_map,
    create_defensive_activity_map,
    create_percentile_heatmap_table,
    create_quadrant_scatter_mpl,
    create_multi_metric_dispersion,
    get_zone_centers,
    normalize_name
)
from src.core.colors import BLUES, ACCENTS

TARGET_PLAYER = "Enzo Fernandez"
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
INSIGHTS_DIR = OUTPUT_DIR / "insights"
METRICS_DIR = OUTPUT_DIR / "metrics"

for d in [FIGURES_DIR, INSIGHTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def main():
    print(f"[*] Starting Gallery Generation Pipeline for {TARGET_PLAYER}...")

    print("\n[1] Loading Data...")
    loader = StatsBombDataLoader()
    data = loader.load_competition_data()
    events = data['events']
    lineups = data['lineups']
    print(f"    Loaded {len(events)} events.")

    all_players = events['player'].dropna().unique()
    enzo_name = None
    target_norm = normalize_name(TARGET_PLAYER)
    for p in all_players:
        if normalize_name(p) == target_norm:
            enzo_name = p
            break

    if not enzo_name:
        print(f"    [!] Error: Could not resolve name for {TARGET_PLAYER}")
        return
    print(f"    Resolved Target Player: {enzo_name}")

    print("\n[2] Building Comparison Group...")
    builder = ComparisonGroupBuilder(lineups, events)
    cms_df = builder.get_central_midfielders(min_minutes=270)
    print(f"    Found {len(cms_df)} qualified central midfielders.")

    print("\n[3] Calculating Metrics...")
    calculator = MetricsCalculator(events)
    metrics_df = calculator.calculate_metrics_for_group(cms_df)

    metrics_df.to_csv(METRICS_DIR / "player_metrics.csv", index=False)
    metrics_df.to_csv(INSIGHTS_DIR / "connector_metrics_all_cms.csv", index=False)

    enzo_row = metrics_df[metrics_df['player'] == enzo_name].iloc[0]

    print("\n[4] Generating Gallery Assets (13 Steps)...")

    print("   1. Performance Profile...")
    pizza_metrics = [
        "deep_progressions_p90", "prog_passes_p90", "prog_carries_p90",
        "pass_completion_%", "passes_final_third_p90",
        "key_passes_p90", "passes_into_box_p90",
        "tackles_interceptions_p90", "recoveries_p90", "pressures_p90"
    ]
    pizza_labels = [
        "Deep Progs", "Prog Passes", "Prog Carries",
        "Pass Comp %", "Final 3rd Pass",
        "Key Passes", "Into Box Pass",
        "Tkl+Int", "Recoveries", "Pressures"
    ]

    fig1 = create_single_pizza(enzo_row, metrics_df, pizza_metrics, pizza_labels, enzo_name)
    fig1.savefig(FIGURES_DIR / "1_performance_profile.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig1)

    enzo_events = events[events['player'] == enzo_name].dropna(subset=['location']).copy()

    def assign_zone(loc):
        if not isinstance(loc, (list, tuple)) or len(loc) < 2: return None
        x, y = loc[0], loc[1]
        if x < 30: col = 0
        elif x < 60: col = 1
        elif x < 90: col = 2
        else: col = 3
        if y < 26.67: row_idx = 0
        elif y < 53.33: row_idx = 1
        else: row_idx = 2
        return col * 3 + row_idx + 1

    enzo_events['reception_zone'] = enzo_events['location'].apply(assign_zone)
    rec_types = ['Ball Receipt*', 'Pressure', 'Ball Recovery', 'Interception', 'Block']
    enzo_receipts = enzo_events[enzo_events['type'].str.contains("Receipt", case=False, na=False)].copy()

    print("   2. Reception Heatmap...")
    fig2 = create_zone_heatmap_pitch(enzo_receipts, 'reception_zone', f"{enzo_name}: Reception Zones", cmap="reception")
    fig2.savefig(FIGURES_DIR / "2_reception_heatmap.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig2)

    def get_prog_zone(row):
        end_loc = None
        if row['type'] == 'Pass' and 'pass_end_location' in row:
            end_loc = row['pass_end_location']
        elif row['type'] == 'Carry' and 'carry_end_location' in row:
            end_loc = row['carry_end_location']

        if end_loc and isinstance(end_loc, (list, tuple)):
            return assign_zone(end_loc)
        return None

    prog_events = enzo_events[enzo_events['type'].isin(['Pass', 'Carry'])].copy()
    prog_events['progression_zone'] = prog_events.apply(get_prog_zone, axis=1)

    # Combine progressions with receipts for the app
    receipts_export = enzo_receipts[['type', 'location', 'reception_zone']].copy()
    receipts_export['progression_zone'] = None  # Receipts don't have a progression zone yet

    prog_events_export = prog_events[['type', 'location', 'progression_zone', 'reception_zone']].copy()
    
    all_events_export = pd.concat([prog_events_export, receipts_export], ignore_index=True)
    all_events_export.to_csv(INSIGHTS_DIR / "enzo_zone_progressions.csv", index=False)

    print("   3. Progression Heatmap...")
    fig3 = create_zone_heatmap_pitch(prog_events, 'progression_zone', f"{enzo_name}: Progression Target Zones", cmap="progression")
    fig3.savefig(FIGURES_DIR / "3_progression_heatmap.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig3)

    print("   4. Distinctive Patterns (Bias)...")
    print("      Processing population progression events (this may take a moment)...")
    cm_players = cms_df['player'].tolist()
    cm_events = events[(events['player'].isin(cm_players)) & (events['type'].isin(['Pass', 'Carry']))].copy()

    cm_events['progression_zone'] = cm_events.apply(get_prog_zone, axis=1)

    enzo_mins = enzo_row['total_minutes']
    enzo_prog_counts = prog_events['progression_zone'].value_counts()
    enzo_prog_p90 = enzo_prog_counts / (enzo_mins / 90)

    player_zone_counts = cm_events.groupby(['player', 'progression_zone']).size().reset_index(name='count')

    mins_map = metrics_df.set_index('player')['total_minutes'].to_dict()
    player_zone_counts['minutes'] = player_zone_counts['player'].map(mins_map)
    player_zone_counts['p90'] = player_zone_counts['count'] / (player_zone_counts['minutes'] / 90)

    pop_prog_p90 = player_zone_counts.groupby('progression_zone')['p90'].mean()

    pop_prog_p90.to_csv(INSIGHTS_DIR / "population_progression_density_p90.csv")

    fig4 = create_difference_heatmap(enzo_prog_p90, pop_prog_p90, f"{enzo_name} vs Average CM: Progression Bias")
    fig4.savefig(FIGURES_DIR / "4_distinctive_patterns_bias.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig4)

    print("   5. Progression Routes...")
    fig5 = create_premium_route_map(prog_events, f"{enzo_name}: Top 10 Progression Routes")
    fig5.savefig(FIGURES_DIR / "5_progression_routes.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig5)

    print("   6. Defensive Actions...")
    def_extractor = DefensiveActionExtractor(events)
    enzo_def_events = def_extractor.get_defensive_actions(enzo_name)
    enzo_def_events['x'] = enzo_def_events['location'].apply(lambda loc: loc[0])
    enzo_def_events['y'] = enzo_def_events['location'].apply(lambda loc: loc[1])

    enzo_def_events.to_csv(INSIGHTS_DIR / "enzo_defensive_actions.csv", index=False)

    fig6 = create_defensive_activity_map(enzo_def_events, f"{enzo_name}: Defensive Activity")
    fig6.savefig(FIGURES_DIR / "6_defensive_actions.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig6)

    print("   7. Recovery Patterns (Bias)...")
    rec_type = 'Ball Recovery'
    enzo_rec = enzo_def_events[enzo_def_events['type'] == rec_type].copy()
    enzo_rec['zone'] = enzo_rec['location'].apply(assign_zone)

    enzo_rec_counts = enzo_rec['zone'].value_counts()
    enzo_rec_p90 = enzo_rec_counts / (enzo_mins / 90)

    cm_def_events = events[(events['player'].isin(cm_players)) & (events['type'] == rec_type)].copy()
    cm_def_events['zone'] = cm_def_events['location'].apply(assign_zone)

    p_rec_counts = cm_def_events.groupby(['player', 'zone']).size().reset_index(name='count')
    p_rec_counts['minutes'] = p_rec_counts['player'].map(mins_map)
    p_rec_counts['p90'] = p_rec_counts['count'] / (p_rec_counts['minutes'] / 90)

    pop_rec_p90 = p_rec_counts.groupby('zone')['p90'].mean()

    pop_rec_p90.to_csv(INSIGHTS_DIR / "population_defensive_density_p90.csv")

    fig7 = create_difference_heatmap(enzo_rec_p90, pop_rec_p90, f"{enzo_name} vs Average CM: Recovery Bias")
    fig7.savefig(FIGURES_DIR / "7_recovery_patterns.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig7)

    print("   8. Comparison Pizza...")
    rival_name = "Bernardo Mota Veiga de Carvalho e Silva"

    if rival_name not in metrics_df['player'].values:
        print(f"      [!] Warning: {rival_name} not found in metrics. Searching for alternate match...")
        rival_norm = normalize_name(rival_name)
        found = False
        for p in metrics_df['player'].values:
            if normalize_name(p) == rival_norm:
                rival_name = p
                found = True
                break

        if not found:
            print("      [!] Could not find Bernardo Silva. Falling back to Pedri.")
            rival_name = "Pedro González López"

    print(f"      Comparing vs {rival_name}")

    comp_metrics = ["prog_passes_p90", "prog_carries_p90", "key_passes_p90", "passes_final_third_p90", "recoveries_p90", "tackles_interceptions_p90", "pressures_p90", "pass_completion_%", "deep_progressions_p90", "passes_into_box_p90"]
    comp_labels = ["Prog Passes", "Prog Carries", "Key Passes", "Final 3rd", "Recoveries", "Tkl + Int", "Pressures", "Pass %", "Deep Progs", "Box Entries"]

    comp_df = metrics_df[metrics_df['player'].isin([enzo_name, rival_name])]
    comp_df['order'] = pd.Categorical(comp_df['player'], categories=[enzo_name, rival_name], ordered=True)
    comp_df = comp_df.sort_values('order')

    fig8 = create_comparison_pizza(comp_df, [enzo_name, rival_name], metrics_df, comp_metrics, comp_labels)
    fig8.savefig(FIGURES_DIR / "8_comparison_pizza.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig8)

    sim_highlight_names = [
        enzo_name,
        "Bernardo Mota Veiga de Carvalho e Silva",
        "Frenkie de Jong",
        "Luka Modrić"
    ]

    valid_sim_highlights = []
    for raw_target in sim_highlight_names:
        t_norm = normalize_name(raw_target)
        matched = False
        for p in metrics_df['player'].unique():
            if normalize_name(p) == t_norm:
                valid_sim_highlights.append(p)
                matched = True
                break
        if not matched:
             print(f"      [!] Warning: Could not find {raw_target} for highlight group/quadrant.")

    if not valid_sim_highlights:
        valid_sim_highlights = [enzo_name]

    print("   9. Percentile Heatmap (Specific Group)...")
    comp_group_df = metrics_df[metrics_df['player'].isin(valid_sim_highlights)].copy()

    comp_group_df['order'] = pd.Categorical(comp_group_df['player'], categories=valid_sim_highlights, ordered=True)
    comp_group_df = comp_group_df.sort_values('order')

    hm_metrics = ['deep_progressions_p90', 'pass_completion_%', 'deep_receptions_p90', 'prog_passes_p90', 'prog_carries_p90', 'key_passes_p90', 'passes_final_third_p90', 'recoveries_p90', 'tackles_interceptions_p90', 'pressures_p90']
    hm_labels = ['Deep Progressions (p90)', 'Success Rate (%)', 'Deep Receptions (p90)', 'Progressive Passes (p90)', 'Progressive Carries (p90)', 'Key Passes (p90)', 'Final Third Passes (p90)', 'Recoveries (p90)', 'Tackles + Interceptions (p90)', 'Pressures (p90)']

    fig9 = create_percentile_heatmap_table(comp_group_df, metrics_df, hm_metrics, hm_labels, title="Comparison: Enzo vs Elite Connectors")
    fig9.savefig(FIGURES_DIR / "9_percentile_heatmap.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig9)

    print("   10. Efficiency Quadrant...")
    fig10 = create_quadrant_scatter_mpl(metrics_df, 'deep_progressions_p90', 'pass_completion_%', 'Deep Progressions (p90)', 'Pass Completion %', 'Deep Progression Efficiency', highlight_players=valid_sim_highlights)
    fig10.savefig(FIGURES_DIR / "10_efficiency_quadrant.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig10)

    print("   11. Metric Dispersion...")
    disp_metrics = ['deep_progressions_p90', 'pass_completion_%', 'deep_receptions_p90', 'prog_passes_p90', 'key_passes_p90']
    disp_labels = ['Deep Progs (p90)', 'Success %', 'Deep Rcpts (p90)', 'Prog Passes (p90)', 'Key Passes (p90)']
    fig11 = create_multi_metric_dispersion(metrics_df, disp_metrics, disp_labels, highlight_players=valid_sim_highlights)
    fig11.savefig(FIGURES_DIR / "11_metric_dispersion.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
    plt.close(fig11)

    print("   12/13. Similarity Analysis (Z-Score Algorithm)...")
    sim_metrics = ['deep_progressions_p90', 'prog_passes_p90', 'prog_carries_p90', 'pass_completion_%', 'passes_final_third_p90', 'recoveries_p90', 'key_passes_p90', 'long_balls_p90']
    sim_metrics = [m for m in sim_metrics if m in metrics_df.columns]

    try:
        similar_df = calculate_similarity(metrics_df, enzo_name, sim_metrics, top_n=5)
        similar_df.to_csv(INSIGHTS_DIR / "similarity_results.csv", index=False)

        z_score_group = similar_df['player'].tolist()

        fig12 = create_multi_metric_dispersion(metrics_df, ['deep_progressions_p90', 'pass_completion_%', 'prog_passes_p90', 'recoveries_p90'], ['Deep Progs (p90)', 'Success %', 'Prog Passes (p90)', 'Recoveries (p90)'], highlight_players=z_score_group)
        fig12.savefig(FIGURES_DIR / "12_similarity_dispersion.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
        plt.close(fig12)

        sim_comp_df = metrics_df[metrics_df['player'].isin(z_score_group)].copy()
        sim_comp_df['order'] = pd.Categorical(sim_comp_df['player'], categories=z_score_group, ordered=True)
        sim_comp_df = sim_comp_df.sort_values('order')

        fig13 = create_percentile_heatmap_table(sim_comp_df, metrics_df, hm_metrics, hm_labels, title=f"Statistical Similarity (Z-Score)")
        fig13.savefig(FIGURES_DIR / "13_similarity_heatmap.png", dpi=300, bbox_inches='tight', facecolor='#0B132B')
        plt.close(fig13)

    except Exception as e:
        print(f"    [!] Error in similarity/comparison section: {e}")

    print("\n   14. Generating Report with Dynamic Insights...")

    right_zones = [3, 6, 9, 12]
    total_prog = len(prog_events)
    right_prog = prog_events['progression_zone'].isin(right_zones).sum()
    right_side_pct = (right_prog / total_prog * 100) if total_prog > 0 else 0

    report_data = {}

    report_data['total_minutes'] = int(enzo_row.get('minutes', enzo_row.get('total_minutes', 0)))
    report_data['deep_progressions_p90'] = f"{enzo_row.get('deep_progressions_p90', 0):.2f}"
    report_data['rank'] = int(enzo_row.get('rank', 0))
    report_data['prog_passes_p90'] = f"{enzo_row.get('prog_passes_p90', 0):.2f}"
    report_data['pass_completion_pct'] = f"{enzo_row.get('pass_completion_%', 0):.1f}"

    report_data['deep_receptions_p90'] = f"{enzo_row.get('deep_receptions_p90', 0):.2f}"
    report_data['right_side_pct'] = f"{right_side_pct:.1f}"
    report_data['passes_final_third_p90'] = f"{enzo_row.get('passes_final_third_p90', 0):.2f}"

    report_data['pressures_p90'] = f"{enzo_row.get('pressures_p90', 0):.2f}"
    report_data['recoveries_p90'] = f"{enzo_row.get('recoveries_p90', 0):.2f}"
    report_data['tackles_interceptions_p90'] = f"{enzo_row.get('tackles_interceptions_p90', 0):.2f}"

    try:
        sim_names = similar_df['player'].tolist()
        report_data['sim_1_name'] = sim_names[1] if len(sim_names) > 1 else "Unknown"
        report_data['sim_2_name'] = sim_names[2] if len(sim_names) > 2 else "Unknown"
        report_data['sim_3_name'] = sim_names[3] if len(sim_names) > 3 else "Unknown"

        rival_row = metrics_df[metrics_df['player'] == rival_name].iloc[0] if rival_name in metrics_df['player'].values else None

        if rival_row is not None:
             report_data['rival_name'] = rival_name
             report_data['rival_prog_passes_p90'] = f"{rival_row.get('prog_passes_p90', 0):.2f}"
             report_data['rival_recoveries_p90'] = f"{rival_row.get('recoveries_p90', 0):.2f}"
             report_data['rival_deep_progs_p90'] = f"{rival_row.get('deep_progressions_p90', 0):.2f}"
        else:
             report_data['rival_name'] = "Rival"
             report_data['rival_prog_passes_p90'] = "N/A"
             report_data['rival_recoveries_p90'] = "N/A"
             report_data['rival_deep_progs_p90'] = "N/A"

    except Exception as e:
        print(f"      [!] Error extracting similarity/rival metrics: {e}")
        report_data['sim_1_name'] = "Player A"
        report_data['sim_2_name'] = "Player B"
        report_data['sim_3_name'] = "Player C"
        report_data['rival_name'] = "Rival"
        report_data['rival_prog_passes_p90'] = "0"
        report_data['rival_recoveries_p90'] = "0"
        report_data['rival_deep_progs_p90'] = "0"

    print("\n[*] Gallery Generation Complete! All assets stored in output/figures/")

if __name__ == "__main__":
    main()
