# analysis_module.py
print("Loaded analysis_module.py from Scriptit folder")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Radar
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import re

GOAL_X, GOAL_Y = 120, 40

# === Time Utilities ===
def parse_time(t):
    if pd.isna(t): return None
    t = str(t).strip()
    if t in ['', '--', 'nan', 'None']: return None
    if '+' in t:
        try:
            base, extra = t.split('+')
            base = base.strip()
            extra = extra.strip()
            base_min, base_sec = map(int, re.split('[:.]', base))
            extra_min, extra_sec = map(int, re.split('[:.]', extra))
            return (base_min + extra_min) * 60 + base_sec + extra_sec
        except Exception:
            return None
    parts = re.split('[:.]', t)
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return None
    elif len(parts) == 3:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return None
    else:
        try:
            return float(t)
        except Exception:
            return None

def format_seconds(seconds):
    minutes = int(seconds) // 60
    sec = int(seconds) % 60
    return f"{minutes:02}:{sec:02}"

# === Feature Engineering ===
def calculate_angle(x, y):
    dx = GOAL_X - x
    dy_top = y - 36
    dy_bottom = y - 44
    return np.abs(np.arctan2(dy_top, dx) - np.arctan2(dy_bottom, dx))

def process_shots(shots_df, expected_cols, xgb_model):
    shots_df = shots_df.copy()
    shots_df = shots_df.dropna(subset=['X', 'Y'])
    shots_df['location_x'] = shots_df['X'] * 1.2
    shots_df['location_y'] = shots_df['Y'] * 0.8

    for col in expected_cols:
        shots_df[col] = 0

    play_pattern_map = {
        'From Goal Kick': 'play_pattern_From Goal Kick',
        'From Free Kick': 'play_pattern_From Free Kick',
        'From Corner': 'play_pattern_From Corner',
        'From Throw In': 'play_pattern_From Throw In',
        'From Counter': 'play_pattern_From Counter',
        'Regular Play': 'play_pattern_Regular Play'
    }
    for pattern, col_name in play_pattern_map.items():
        shots_df.loc[shots_df['Secondary'].str.contains(pattern, na=False), col_name] = 1
    pattern_cols = [col for col in expected_cols if col.startswith("play_pattern_")]
    shots_df['play_pattern_Regular Play'] = (shots_df[pattern_cols].sum(axis=1) == 0).astype(int)
    shots_df['shot_distance'] = np.sqrt((shots_df['location_x'] - GOAL_X)**2 + (shots_df['location_y'] - GOAL_Y)**2)
    shots_df['shot_angle'] = shots_df.apply(lambda row: calculate_angle(row['location_x'], row['location_y']), axis=1)
    body_part_map = {'Left Foot': 0, 'Right Foot': 1, 'Head': 2}
    shots_df['body_part_code'] = shots_df['Body Part'].map(body_part_map).fillna(1)
    # Penalty override after prediction
    is_penalty = shots_df['Secondary'].str.contains("Penalty", na=False)
    shots_df['xG'] = xgb_model.predict_proba(shots_df[expected_cols])[:, 1]
    shots_df.loc[is_penalty, 'xG'] = 0.78
    return shots_df

# === KPI Extraction ===
def extract_full_match_stats(df, xgb_model, expected_cols, team_name):
    # Helper functions
    def count_goals(df): return len(df[(df["Name"] == "Shot") & (df["Primary"] == "Goal")])
    def count_shots(df): return len(df[df["Name"] == "Shot"])
    def count_box_entries(df): return len(df[df["Name"] == "Box Entry"])
    def count_box_entries_dribble(df): return len(df[(df["Name"] == "Box Entry") & (df["Primary"] == "Dribble")])
    def count_passes_behind(df): return len(df[df["Name"] == "Behind"])
    def count_successful_passes_behind(df): return len(df[(df["Name"] == "Behind") & (df["Primary"] == "Successful")])
    def count_high_recoveries(df): return len(df[(df["Name"] == "Defensive Action") & (df["X"] > 66)])
    def count_fouls(df): return len(df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Foul")])
    def count_opp_goals(df): return len(df[(df["Name"] == "Shot OPP") & (df["Primary"] == "Goal")])
    def count_opp_shots(df): return len(df[df["Name"] == "Shot OPP"])
    def count_opp_box_entries(df): return len(df[df["Name"] == "Box Entry OPP"])
    def count_opp_box_entries_dribble(df): return len(df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Dribble")])
    def count_opp_passes_behind(df): return len(df[df["Name"] == "Behind OPP"])
    def count_opp_successful_passes_behind(df): return len(df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Successful")])
    def count_opp_high_recoveries(df): return len(df[(df["Name"] == "Defensive Action OPP") & (df["X"] < 33)])
    def count_opp_fouls(df): return len(df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Foul")])

    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    xG = shots['xG'].sum()
    xG_opp = shots_opp['xG'].sum()
    goals = count_goals(df)
    num_shots = count_shots(df)
    conversion = goals / num_shots if num_shots > 0 else 0
    box_entries = count_box_entries(df)
    box_entries_dribble = count_box_entries_dribble(df)
    xG_per_shot = xG / num_shots if num_shots > 0 else 0
    passes_behind = count_passes_behind(df)
    successful_passes_behind = count_successful_passes_behind(df)
    high_recoveries = count_high_recoveries(df)
    fouls = count_fouls(df)
    goals_opp = count_opp_goals(df)
    num_shots_opp = count_opp_shots(df)
    conversion_opp = goals_opp / num_shots_opp if num_shots_opp > 0 else 0
    box_entries_opp = count_opp_box_entries(df)
    box_entries_dribble_opp = count_opp_box_entries_dribble(df)
    xG_per_shot_opp = xG_opp / num_shots_opp if num_shots_opp > 0 else 0
    passes_behind_opp = count_opp_passes_behind(df)
    successful_passes_behind_opp = count_opp_successful_passes_behind(df)
    high_recoveries_opp = count_opp_high_recoveries(df)
    fouls_opp = count_opp_fouls(df)
    total_time = df[df["Name"] == "Total Time"]["Duration"].sum()
    attacking_time = df[df["Name"] == "Attacking Time"]["Duration"].sum()
    attacking_time_opp = df[df["Name"] == "Attacking Time OPP"]["Duration"].sum()
    att_percent = (attacking_time / total_time * 100) if total_time > 0 else 0
    opp_att_percent = (attacking_time_opp / total_time * 100) if total_time > 0 else 0

    jjk_stats = [
        goals,
        f"{xG:.2f}",
        num_shots,
        f"{xG_per_shot:.2f}" if num_shots > 0 else "N/A",
        f"{conversion:.2f}",
        box_entries,
        format_seconds(attacking_time),
        f"{att_percent:.1f}%",
        f"{attacking_time / box_entries:.2f}" if box_entries > 0 else "N/A",
        passes_behind,
        successful_passes_behind,
        high_recoveries,
        fouls
    ]
    opp_stats = [
        goals_opp,
        f"{xG_opp:.2f}",
        num_shots_opp,
        f"{xG_per_shot_opp:.2f}" if num_shots_opp > 0 else "N/A",
        f"{conversion_opp:.2f}",
        box_entries_opp,
        format_seconds(attacking_time_opp),
        f"{opp_att_percent:.1f}%",
        f"{attacking_time_opp / box_entries_opp:.2f}" if box_entries_opp > 0 else "N/A",
        passes_behind_opp,
        successful_passes_behind_opp,
        high_recoveries_opp,
        fouls_opp
    ]
    row_labels = [
        "Goals", "xG", "Shots", "xG / Shot", "Conversion", "Box Entries",
        "Attacking Time", "AT %", "AT / BE (s)", "Passes Behind", "Succ. Passes Behind", "High Recoveries", "Fouls"
    ]
    return row_labels, jjk_stats, opp_stats

def generate_kpi_table(row_labels, jjk_stats, opp_stats, team_name):
    cell_text = []
    for i in range(len(row_labels)):
        cell_text.append([jjk_stats[i], opp_stats[i]])
    columns = ["JJK", team_name]
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#122c3d')
    ax.set_facecolor('#122c3d')
    ax.axis('off')
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=columns,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#122c3d')
        cell.set_linewidth(0)
        cell.set_text_props(color='white')
        cell.set_facecolor('#122c3d')
    for cell in table._cells:
        if cell[0] == 0:
            table._cells[cell].set_fontsize(15)
    plt.tight_layout()
    return fig

# === Radar Chart ===
def generate_radar_chart(jjk_vals, opp_vals, avg_jjk, avg_opp, team_name):
    params = ['Shots', 'xG', 'xG/Shot', 'AT%', 'AT/BE (s)', 'High Recoveries', 'Succ. Passes Behind', 'Box Entries']
    # Set min/max for normalization
    low = [0, 0, 0, 0, 20, 0, 0, 0]
    high = [
        max(jjk_vals[0], opp_vals[0], avg_jjk[0], avg_opp[0]) + 5,
        max(jjk_vals[1], opp_vals[1], avg_jjk[1], avg_opp[1]) + 0.5,
        max(jjk_vals[2], opp_vals[2], avg_jjk[2], avg_opp[2]) + 0.05,
        max(jjk_vals[3], opp_vals[3], avg_jjk[3], avg_opp[3]) + 5,
        max(jjk_vals[4], opp_vals[4], avg_jjk[4], avg_opp[4]) + 5,
        max(jjk_vals[5], opp_vals[5], avg_jjk[5], avg_opp[5]) + 5,
        max(jjk_vals[6], opp_vals[6], avg_jjk[6], avg_opp[6]) + 5,
        max(jjk_vals[7], opp_vals[7], avg_jjk[7], avg_opp[7]) + 5
    ]
    lower_is_better = ['AT/BE (s)']
    radar = Radar(params, low, high, lower_is_better=lower_is_better,
                  round_int = [True, False, False, False, False, True, True, True],
                  num_rings=5, ring_width=1,
                  center_circle_radius=1)
    fig, ax = radar.setup_axis(figsize=(8,8))
    radar.draw_circles(ax=ax, facecolor="#28252c", edgecolor="#39353f", lw=1.5)
    radar.draw_radar_solid(jjk_vals, ax=ax, kwargs={'facecolor': '#A6192E', 'alpha': 0.1, 'edgecolor': '#A6192E', 'lw': 3})
    radar.draw_radar_solid(avg_jjk, ax=ax, kwargs={'facecolor': "#A6192E00", 'alpha': 0.99, 'edgecolor': "#470D15", 'lw': 3, 'linestyle': '--'})
    radar.draw_radar_solid(opp_vals, ax=ax, kwargs={'facecolor': '#FFD100', 'alpha': 0.1, 'edgecolor': '#FFD100', 'lw': 3})
    radar.draw_radar_solid(avg_opp, ax=ax, kwargs={'facecolor': "#FFD00000", 'alpha': 0.55, 'edgecolor': "#000000FF", 'lw': 3, 'linestyle': '--'})
    radar.draw_range_labels(ax=ax, fontsize=15, color='#fcfcfc')
    radar.draw_param_labels(ax=ax, fontsize=15, color='#fcfcfc')
    jjk_patch = mpatches.Patch(color='#A6192E', label='JJK')
    opp_patch = mpatches.Patch(color='#FFD100', label=team_name)
    jjk_avg_line = mlines.Line2D([], [], color='#470D15', linestyle='--', linewidth=3, label='JJK avg.')
    opp_avg_line = mlines.Line2D([], [], color='#000000FF', linestyle='--', linewidth=3, label='Opp avg.')
    ax.legend(
        handles=[jjk_patch, opp_patch, jjk_avg_line, opp_avg_line],
        loc='upper right',
        facecolor='#122c3d',
        edgecolor='white',
        labelcolor='white',
        fontsize=12
    )
    fig.patch.set_facecolor('#122c3d')
    ax.set_facecolor('#122c3d')
    plt.tight_layout()
    return fig

# === Cumulative xG Plot ===
def plot_cumulative_xg(df, xgb_model, expected_cols, team_name):
    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    shots['time'] = shots['End'].apply(parse_time)
    shots = shots.sort_values('time')
    shots['cum_xG'] = shots['xG'].cumsum()
    shots_opp['time'] = shots_opp['End'].apply(parse_time)
    shots_opp = shots_opp.sort_values('time')
    shots_opp['cum_xG'] = shots_opp['xG'].cumsum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#122c3d')
    fig.patch.set_facecolor('#122c3d')
    ax.step(shots['time'], shots['cum_xG'], where='post', label='JJK', color='#A6192E', linewidth=2)
    ax.step(shots_opp['time'], shots_opp['cum_xG'], where='post', label=team_name, color='#FFD100', linewidth=2)
    # Vertical lines for goals
    for t in shots[(shots['Primary'] == 'Goal')]['time']:
        ax.axvline(x=t, color='#A6192E', linestyle=':', linewidth=2, alpha=0.9)
    for t in shots_opp[(shots_opp['Primary'] == 'Goal')]['time']:
        ax.axvline(x=t, color='#FFD100', linestyle=':', linewidth=2, alpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)//60:02}:{int(x)%60:02}"))
    ax.set_xlabel('Match Time (mm:ss)', color='white', fontsize=12)
    ax.set_ylabel('Cumulative xG', color='white', fontsize=12)
    ax.set_title('Cumulative xG Progression', color='white', fontsize=15)
    ax.legend(facecolor='#122c3d', edgecolor='white', labelcolor='white')
    ax.grid(True, axis='y', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.tight_layout()
    return fig

# === Momentum Chart ===
def plot_momentum_chart(df, team_name):
    interval = 300
    att_df = df[df['Name'] == 'Attacking Time'].copy()
    att_df['start_sec'] = att_df['Start'].apply(parse_time)
    att_df['end_sec'] = att_df['End'].apply(parse_time)
    att_opp_df = df[df['Name'] == 'Attacking Time OPP'].copy()
    att_opp_df['start_sec'] = att_opp_df['Start'].apply(parse_time)
    att_opp_df['end_sec'] = att_opp_df['End'].apply(parse_time)
    total_df = df[df['Name'] == 'Total Time'].copy()
    total_df['start_sec'] = total_df['Start'].apply(parse_time)
    total_df['end_sec'] = total_df['End'].apply(parse_time)
    match_end = max(
        total_df['end_sec'].max() if not total_df.empty else 0,
        att_df['end_sec'].max() if not att_df.empty else 0,
        att_opp_df['end_sec'].max() if not att_opp_df.empty else 0
    )
    bins = np.arange(0, match_end + interval, interval)
    att_percent_list = []
    opp_att_percent_list = []
    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i+1]
        total_time = total_df.apply(lambda row: max(0, min(row['end_sec'], end) - max(row['start_sec'], start)), axis=1).sum()
        att_time = att_df.apply(lambda row: max(0, min(row['end_sec'], end) - max(row['start_sec'], start)), axis=1).sum()
        att_opp_time = att_opp_df.apply(lambda row: max(0, min(row['end_sec'], end) - max(row['start_sec'], start)), axis=1).sum()
        att_percent = (att_time / total_time * 100) if total_time > 0 else 0
        opp_att_percent = (att_opp_time / total_time * 100) if total_time > 0 else 0
        att_percent_list.append(att_percent)
        opp_att_percent_list.append(opp_att_percent)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#122c3d')
    fig.patch.set_facecolor('#122c3d')
    momentum = np.array(att_percent_list) - np.array(opp_att_percent_list)
    x_vals = bins[1:]
    ax.bar(x_vals, momentum, width=interval*0.9, color=np.where(momentum >= 0, '#A6192E', '#FFD100'), align='center', alpha=0.85, edgecolor='none')
    ax.axhline(0, color='white', linewidth=1.5, linestyle='-')
    # Add vertical lines for goals
    shots = df[df['Name'] == 'Shot'].copy()
    shots['time'] = shots['End'].apply(parse_time)
    shots_opp = df[df['Name'] == 'Shot OPP'].copy()
    shots_opp['time'] = shots_opp['End'].apply(parse_time)
    for t in shots[(shots['Primary'] == 'Goal')]['time']:
        ax.axvline(x=t, color='#A6192E', linestyle=':', linewidth=2, alpha=0.9)
    for t in shots_opp[(shots_opp['Primary'] == 'Goal')]['time']:
        ax.axvline(x=t, color='#FFD100', linestyle=':', linewidth=2, alpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)//60:02}:{int(x)%60:02}"))
    ax.set_xlabel('Match Time (mm:ss)', color='white')
    ax.set_title('Momentum', color='white', fontsize=16)
    ax.grid(True, axis='y', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_ylim(-80, 80)
    ax.set_yticklabels([])
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A6192E', edgecolor='none', label='JJK'),
        Patch(facecolor='#FFD100', edgecolor='none', label=team_name)
    ]
    ax.legend(handles=legend_elements, facecolor='#122c3d', edgecolor='white', labelcolor='white', loc='upper left', fontsize=9)
    plt.tight_layout()
    return fig

# === Combined Pitch Plot ===
def plot_combined_pitch(df, xgb_model, expected_cols, selected_events, team_name):
    pitch = Pitch(pitch_color='#122c3d', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))
    # Shots
    if 'Shots' in selected_events:
        shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
        shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
        shots['plot_x'] = shots['location_x']
        shots['plot_y'] = shots['location_y']
        shots_opp['plot_x'] = 120 - shots_opp['location_x']
        shots_opp['plot_y'] = 80 - shots_opp['location_y']
        shots_no_pen = shots[~shots['Secondary'].str.contains("Penalty", na=False)]
        shots_opp_no_pen = shots_opp[~shots_opp['Secondary'].str.contains("Penalty", na=False)]
        pitch.scatter(
            shots_no_pen['plot_x'], shots_no_pen['plot_y'],
            s=shots_no_pen['xG'] * 500, color='#A6192E', edgecolors='black', alpha=0.9, ax=ax, label='JJK Shots'
        )
        pitch.scatter(
            shots_opp_no_pen['plot_x'], shots_opp_no_pen['plot_y'],
            s=shots_opp_no_pen['xG'] * 500, color='#FFD100', edgecolors='black', alpha=0.9, ax=ax, label=f'{team_name} Shots'
        )
    # Box Entry Passes
    if 'Box Entry Passes' in selected_events:
        box_entry_passes = df[(df['Name'] == 'Box Entry') & (df['Primary'] == 'Pass')].dropna(subset=['X', 'Y', 'End X', 'End Y'])
        box_entry_passes_opp = df[(df['Name'] == 'Box Entry OPP') & (df['Primary'] == 'Pass')].dropna(subset=['X', 'Y', 'End X', 'End Y'])
        box_entry_passes['plot_x'] = box_entry_passes['X'] * 1.2
        box_entry_passes['plot_y'] = box_entry_passes['Y'] * 0.8
        box_entry_passes['plot_end_x'] = box_entry_passes['End X'] * 1.2
        box_entry_passes['plot_end_y'] = box_entry_passes['End Y'] * 0.8
        box_entry_passes_opp['plot_x'] = 120 - box_entry_passes_opp['X'] * 1.2
        box_entry_passes_opp['plot_y'] = 80 - box_entry_passes_opp['Y'] * 0.8
        box_entry_passes_opp['plot_end_x'] = 120 - box_entry_passes_opp['End X'] * 1.2
        box_entry_passes_opp['plot_end_y'] = 80 - box_entry_passes_opp['End Y'] * 0.8
        pitch.arrows(
            box_entry_passes['plot_x'], box_entry_passes['plot_y'],
            box_entry_passes['plot_end_x'], box_entry_passes['plot_end_y'],
            ax=ax, color='#A6192E', width=1, headwidth=5, headlength=5, alpha=0.9, zorder=1, label='Box Entry Pass'
        )
        pitch.arrows(
            box_entry_passes_opp['plot_x'], box_entry_passes_opp['plot_y'],
            box_entry_passes_opp['plot_end_x'], box_entry_passes_opp['plot_end_y'],
            ax=ax, color='#FFD100', width=1, headwidth=5, headlength=5, alpha=0.9, zorder=1, label=f'Box Entry Pass {team_name}'
        )
    # Defensive Actions
    if 'Defensive Actions' in selected_events:
        # JJK Defensive Actions
        fouls = df[(df['Name'] == 'Defensive Action') & (df['Primary'] == 'Foul')].dropna(subset=['X', 'Y'])
        interceptions = df[(df['Name'] == 'Defensive Action') & (df['Primary'] == 'Interception')].dropna(subset=['X', 'Y'])
        duels = df[(df['Name'] == 'Defensive Action') & (df['Primary'] == 'Defensive Duel')].dropna(subset=['X', 'Y'])
        # OPP Defensive Actions
        fouls_opp = df[(df['Name'] == 'Defensive Action OPP') & (df['Primary'] == 'Foul')].dropna(subset=['X', 'Y'])
        interceptions_opp = df[(df['Name'] == 'Defensive Action OPP') & (df['Primary'] == 'Interception')].dropna(subset=['X', 'Y'])
        duels_opp = df[(df['Name'] == 'Defensive Action OPP') & (df['Primary'] == 'Defensive Duel')].dropna(subset=['X', 'Y'])

        # JJK
        pitch.scatter(
            fouls['X'] * 1.2, fouls['Y'] * 0.8,
            marker='x', color='#A6192E', s=80, ax=ax, label='JJK Foul'
        )
        pitch.scatter(
            interceptions['X'] * 1.2, interceptions['Y'] * 0.8,
            marker='s', color='#A6192E', s=80, ax=ax, label='JJK Interception'
        )
        pitch.scatter(
            duels['X'] * 1.2, duels['Y'] * 0.8,
            marker='^', color='#A6192E', s=80, ax=ax, label='JJK Defensive Duel'
        )
        # OPP
        pitch.scatter(
            120 - fouls_opp['X'] * 1.2, fouls_opp['Y'] * 0.8,
            marker='x', color='#FFD100', s=80, ax=ax, label=f'{team_name} Foul'
        )
        pitch.scatter(
            120 - interceptions_opp['X'] * 1.2, interceptions_opp['Y'] * 0.8,
            marker='s', color='#FFD100', s=80, ax=ax, label=f'{team_name} Interception'
        )
        pitch.scatter(
            120 - duels_opp['X'] * 1.2, duels_opp['Y'] * 0.8,
            marker='^', color='#FFD100', s=80, ax=ax, label=f'{team_name} Defensive Duel'
        )
    # Defensive Actions (individual overlays)
    if 'Fouls' in selected_events:
        fouls = df[(df['Name'] == 'Defensive Action') & (df['Primary'] == 'Foul')].dropna(subset=['X', 'Y'])
        fouls_opp = df[(df['Name'] == 'Defensive Action OPP') & (df['Primary'] == 'Foul')].dropna(subset=['X', 'Y'])
        pitch.scatter(
            fouls['X'] * 1.2, fouls['Y'] * 0.8,
            marker='x', color='#A6192E', s=80, ax=ax, edgecolors='black', label='JJK Foul'
        )
        pitch.scatter(
            120 - fouls_opp['X'] * 1.2, 80 - fouls_opp['Y'] * 0.8,
            marker='x', color='#FFD100', s=80, ax=ax, edgecolors='black', label=f'{team_name} Foul'
        )
    if 'Interceptions' in selected_events:
        interceptions = df[(df['Name'] == 'Defensive Action') & (df['Primary'] == 'Interception')].dropna(subset=['X', 'Y'])
        interceptions_opp = df[(df['Name'] == 'Defensive Action OPP') & (df['Primary'] == 'Interception')].dropna(subset=['X', 'Y'])
        pitch.scatter(
            interceptions['X'] * 1.2, interceptions['Y'] * 0.8,
            marker='s', color='#A6192E', s=80, ax=ax, edgecolors='black', label='JJK Interception'
        )
        pitch.scatter(
            120 - interceptions_opp['X'] * 1.2, 80 - interceptions_opp['Y'] * 0.8,
            marker='s', color='#FFD100', s=80, ax=ax, edgecolors='black', label=f'{team_name} Interception'
        )
    if 'Defensive Duels' in selected_events:
        duels = df[(df['Name'] == 'Defensive Action') & (df['Primary'] == 'Defensive Duel')].dropna(subset=['X', 'Y'])
        duels_opp = df[(df['Name'] == 'Defensive Action OPP') & (df['Primary'] == 'Defensive Duel')].dropna(subset=['X', 'Y'])
        pitch.scatter(
            duels['X'] * 1.2, duels['Y'] * 0.8,
            marker='^', color='#A6192E', s=100, ax=ax, edgecolors='black', label='JJK Defensive Duel'
        )
        pitch.scatter(
            120 - duels_opp['X'] * 1.2, 80 - duels_opp['Y'] * 0.8,
            marker='^', color='#FFD100', s=100, ax=ax, edgecolors='black', label=f'{team_name} Defensive Duel'
        )
   
    return fig

def get_avg_kpis(folder_path, xgb_model, expected_cols):
    jjk_list = []
    opp_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(folder_path, filename))
                jjk_kpis, opp_kpis = extract_radar_kpis(df, xgb_model, expected_cols)
                jjk_list.append(jjk_kpis)
                opp_list.append(opp_kpis)
            except Exception:
                pass
    if jjk_list and opp_list:
        avg_jjk = np.nanmean(jjk_list, axis=0)
        avg_opp = np.nanmean(opp_list, axis=0)
        return avg_jjk, avg_opp
    else:
        return None, None

def extract_radar_kpis(df, xgb_model, expected_cols):
    # JJK
    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    xG = shots['xG'].sum()
    num_shots = len(shots)
    xG_per_shot = xG / num_shots if num_shots > 0 else 0
    box_entries = len(df[df['Name'] == 'Box Entry'])
    high_recoveries = len(df[(df['Name'] == 'Defensive Action') & (df['X'] > 66)])
    successful_passes_behind = len(df[(df['Name'] == 'Behind') & (df['Primary'] == 'Successful')])
    attacking_time = df[df['Name'] == 'Attacking Time']['Duration'].sum()
    total_time = df[df['Name'] == 'Total Time']['Duration'].sum()
    att_percent = (attacking_time / total_time * 100) if total_time > 0 else 0
    att_per_box = attacking_time / box_entries if box_entries > 0 else 0

    jjk_kpis = [
        num_shots,
        xG,
        xG_per_shot,
        att_percent,
        att_per_box,
        high_recoveries,
        successful_passes_behind,
        box_entries
    ]
    # OPP
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    xG_opp = shots_opp['xG'].sum()
    num_shots_opp = len(shots_opp)
    xG_per_shot_opp = xG_opp / num_shots_opp if num_shots_opp > 0 else 0
    box_entries_opp = len(df[df['Name'] == 'Box Entry OPP'])
    high_recoveries_opp = len(df[(df['Name'] == 'Defensive Action OPP') & (df['X'] < 33)])
    successful_passes_behind_opp = len(df[(df['Name'] == 'Behind OPP') & (df['Primary'] == 'Successful')])
    attacking_time_opp = df[df['Name'] == 'Attacking Time OPP']['Duration'].sum()
    att_percent_opp = (attacking_time_opp / total_time * 100) if total_time > 0 else 0
    att_per_box_opp = attacking_time_opp / box_entries_opp if box_entries_opp > 0 else 0

    opp_kpis = [
        num_shots_opp,
        xG_opp,
        xG_per_shot_opp,
        att_percent_opp,
        att_per_box_opp,
        high_recoveries_opp,
        successful_passes_behind_opp,
        box_entries_opp
    ]
    return jjk_kpis, opp_kpis
