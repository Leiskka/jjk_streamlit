
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
    def count_high_recoveries(df): return len(df[(df["Name"] == "Defensive Action") & (df["X"] > 66) & (df["Primary"] != "Foul")])
    def count_fouls(df): return len(df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Foul")])
    def count_opp_goals(df): return len(df[(df["Name"] == "Shot OPP") & (df["Primary"] == "Goal")])
    def count_opp_shots(df): return len(df[df["Name"] == "Shot OPP"])
    def count_opp_box_entries(df): return len(df[df["Name"] == "Box Entry OPP"])
    def count_opp_box_entries_dribble(df): return len(df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Dribble")])
    def count_opp_passes_behind(df): return len(df[df["Name"] == "Behind OPP"])
    def count_opp_successful_passes_behind(df): return len(df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Successful")])
    def count_opp_high_recoveries(df): return len(df[(df["Name"] == "Defensive Action OPP") & (df["X"] < 33) & (df["Primary"] != "Foul")])
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
    passes_behind_success_rate = (successful_passes_behind / passes_behind * 100) if passes_behind > 0 else 0

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
    passes_behind_success_rate_opp = (successful_passes_behind_opp / passes_behind_opp * 100) if passes_behind_opp > 0 else 0
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
        f"{attacking_time / box_entries:.1f}" if box_entries > 0 else "N/A",
        passes_behind,
        successful_passes_behind,
        f"{passes_behind_success_rate:.1f}%",  # <-- Add here
        high_recoveries
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
        f"{attacking_time_opp / box_entries_opp:.1f}" if box_entries_opp > 0 else "N/A",
        passes_behind_opp,
        successful_passes_behind_opp,
        f"{passes_behind_success_rate_opp:.1f}%", 
        high_recoveries_opp
    ]
    row_labels = [
        "Goals", "xG", "Shots", "xG / Shot", "Conversion", "Box Entries",
        "Attacking Time", "AT %", "AT / BE (s)", "Passes Behind", "Succ. Passes Behind",
        "Succ. Passes Behind %",
        "High Recoveries"
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
    low = [0, 0, 0.07, 0, 20, 0, 0, 0]
    high = [
        max(jjk_vals[0], opp_vals[0], avg_jjk[0], avg_opp[0]) + 5,
        max(jjk_vals[1], opp_vals[1], avg_jjk[1], avg_opp[1]) + 0.5,
        max(jjk_vals[2], opp_vals[2], avg_jjk[2], avg_opp[2]) + 0.03,
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
    radar.draw_radar_solid(jjk_vals, ax=ax, kwargs={'facecolor': '#A6192E', 'alpha': 0.6, 'edgecolor': '#A6192E', 'lw': 3})
    radar.draw_radar_solid(avg_jjk, ax=ax, kwargs={'facecolor': 'none', 'alpha': 0.9, 'edgecolor': "#A6192E", 'lw': 3, 'linestyle': '--'})
    radar.draw_radar_solid(opp_vals, ax=ax, kwargs={'facecolor': '#FFD100', 'alpha': 0.6, 'edgecolor': '#FFD100', 'lw': 3})
    radar.draw_radar_solid(avg_opp, ax=ax, kwargs={'facecolor': 'none', 'alpha': 0.9, 'edgecolor': "#FFD100", 'lw': 3, 'linestyle': '--'})
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
def plot_cumulative_xg(df, xgb_model, expected_cols, team_name, selected_half="Full Match"):
    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    shots['time'] = shots['End'].apply(parse_time)
    shots = shots.sort_values('time')
    shots['cum_xG'] = shots['xG'].cumsum()
    shots_opp['time'] = shots_opp['End'].apply(parse_time)
    shots_opp = shots_opp.sort_values('time')
    shots_opp['cum_xG'] = shots_opp['xG'].cumsum()

    # Add 0:00 point for both teams
    shots = pd.concat([
        pd.DataFrame({'time': [0], 'cum_xG': [0]}),
        shots[['time', 'cum_xG']]
    ], ignore_index=True)
    shots_opp = pd.concat([
        pd.DataFrame({'time': [0], 'cum_xG': [0]}),
        shots_opp[['time', 'cum_xG']]
    ], ignore_index=True)

    # Set x-axis range based on selected_half
    if selected_half == "1st Half":
        x_min, x_max = 0, 2700
    elif selected_half == "2nd Half":
        x_min, x_max = 2700, 5400
    else:
        x_min, x_max = 0, 5400

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#122c3d')
    fig.patch.set_facecolor('#122c3d')
    ax.step(shots['time'], shots['cum_xG'], where='post', label='JJK', color='#A6192E', linewidth=2)
    ax.step(shots_opp['time'], shots_opp['cum_xG'], where='post', label=team_name, color='#FFD100', linewidth=2)
    # Vertical lines for goals only
    goal_times = df[(df['Name'] == 'Shot') & (df['Primary'] == 'Goal')]['End'].apply(parse_time)
    for t in goal_times:
        if x_min <= t <= x_max:
            ax.axvline(x=t, color='#A6192E', linestyle=':', linewidth=2, alpha=0.9)
    goal_times_opp = df[(df['Name'] == 'Shot OPP') & (df['Primary'] == 'Goal')]['End'].apply(parse_time)
    for t in goal_times_opp:
        if x_min <= t <= x_max:
            ax.axvline(x=t, color='#FFD100', linestyle=':', linewidth=2, alpha=0.9)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    if selected_half == "1st Half":
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
        )
    elif selected_half == "2nd Half":
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
        )
    else:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
        )
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
def plot_momentum_chart(df, team_name, selected_half="Full Match"):
    # Ensure 'Start' is in seconds (int)
    if not np.issubdtype(df['Start'].dtype, np.number):
        df = df.copy()
        df['Start_sec'] = df['Start'].apply(parse_time)
    else:
        df['Start_sec'] = df['Start']

    # Determine time window based on selected_half
    if selected_half == "1st Half":
        start_time = 0
        end_time = 2700  # 45*60
    elif selected_half == "2nd Half":
        start_time = 2700
        end_time = 5400  # 90*60
    else:
        start_time = 0
        end_time = 5400

    interval = 180  # 1 minute bins
    bins = np.arange(start_time, end_time + interval, interval)
    att_percent_list = []
    opp_att_percent_list = []

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        total_time = df[(df['Start_sec'] >= bin_start) & (df['Start_sec'] < bin_end) & (df['Name'] == 'Total Time')]['Duration'].sum()
        att_time = df[(df['Start_sec'] >= bin_start) & (df['Start_sec'] < bin_end) & (df['Name'] == 'Attacking Time')]['Duration'].sum()
        att_time_opp = df[(df['Start_sec'] >= bin_start) & (df['Start_sec'] < bin_end) & (df['Name'] == 'Attacking Time OPP')]['Duration'].sum()
        att_percent = (att_time / total_time * 100) if total_time > 0 else 0
        opp_att_percent = (att_time_opp / total_time * 100) if total_time > 0 else 0
        att_percent_list.append(att_percent)
        opp_att_percent_list.append(opp_att_percent)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#122c3d')
    fig.patch.set_facecolor('#122c3d')
    momentum = np.array(att_percent_list) - np.array(opp_att_percent_list)
    x_vals = bins[1:] - 90
    ax.bar(x_vals, momentum, width=interval*0.9, color=np.where(momentum >= 0, '#A6192E', '#FFD100'), align='center', alpha=0.85, edgecolor='none')
    ax.axhline(0, color='white', linewidth=1.5, linestyle='-')

    # Add vertical lines for goals
    shots = df[df['Name'] == 'Shot'].copy()
    shots['time'] = shots['End'].apply(parse_time)
    shots_opp = df[df['Name'] == 'Shot OPP'].copy()
    shots_opp['time'] = shots_opp['End'].apply(parse_time)
    for t in shots[(shots['Primary'] == 'Goal')]['time']:
        if start_time <= t < end_time:
            ax.axvline(t, color='#A6192E', linestyle='--', linewidth=1.5, alpha=0.7)
    for t in shots_opp[(shots_opp['Primary'] == 'Goal')]['time']:
        if start_time <= t < end_time:
            ax.axvline(t, color='#FFD100', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlim(start_time, end_time)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(900))
    if selected_half == "1st Half":
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
        )
    elif selected_half == "2nd Half":
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
        )
    else:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
        )

    ax.set_xlim(start_time, end_time)
    ax.set_xlabel('Match Time (mm:ss)', color='white')
    ax.set_title('Momentum', color='white', fontsize=16)
    ax.grid(True, axis='y', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_ylim(-100, 100)
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
        box_entry_passes_opp['plot_x'] = box_entry_passes_opp['X'] * 1.2
        box_entry_passes_opp['plot_y'] = box_entry_passes_opp['Y'] * 0.8
        box_entry_passes_opp['plot_end_x'] = box_entry_passes_opp['End X'] * 1.2
        box_entry_passes_opp['plot_end_y'] = box_entry_passes_opp['End Y'] * 0.8
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
            fouls_opp['X'] * 1.2, fouls_opp['Y'] * 0.8,
            marker='x', color='#FFD100', s=80, ax=ax, label=f'{team_name} Foul'
        )
        pitch.scatter(
            interceptions_opp['X'] * 1.2, interceptions_opp['Y'] * 0.8,
            marker='s', color='#FFD100', s=80, ax=ax, label=f'{team_name} Interception'
        )
        pitch.scatter(
            duels_opp['X'] * 1.2, duels_opp['Y'] * 0.8,
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
            fouls_opp['X'] * 1.2, fouls_opp['Y'] * 0.8,
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
            interceptions_opp['X'] * 1.2, interceptions_opp['Y'] * 0.8,
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
            duels_opp['X'] * 1.2, duels_opp['Y'] * 0.8,
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
    high_recoveries = len(df[(df['Name'] == 'Defensive Action') & (df['X'] > 66) & (df['Primary'] != "Foul")])
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
    high_recoveries_opp = len(df[(df['Name'] == 'Defensive Action OPP') & (df['X'] < 33) & (df['Primary'] != "Foul")])
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

def plot_kpi_bars(jjk_vals, opp_vals, avg_jjk, avg_opp, team_name):
    import matplotlib.pyplot as plt
    import numpy as np

    params = [
        'Shots', 'xG', 'xG/Shot', 'AT%', 'AT/BE (s)', 
        'High Recoveries', 'Succ. Passes Behind', 'Box Entries'
    ]
    colors = ['#A6192E', '#FFD100']  # JJK, OPP
    avg_colors = ['#7A0F20', '#B89E00']  # Darker shades for averages
    n_params = len(params)
    bg_color = '#122c3d'
    bar_width = 0.01

    def format_val(val, idx):
        if idx in [0, 5, 6, 7]:
            return f"{int(round(val))}"
        elif idx in [1, 2, 4]:
            return f"{val:.1f}"
        elif idx == 3:
            return f"{val:.1f}"
        else:
            return f"{val}"

    fig, axes = plt.subplots(1, n_params, figsize=(7, 3), sharey=False)
    fig.patch.set_facecolor(bg_color)

    for i, ax in enumerate(axes):
        ax.set_facecolor(bg_color)
        max_val = max(jjk_vals[i], opp_vals[i], avg_jjk[i], avg_opp[i])
        ylim = max_val * 1.2 if max_val > 0 else 1
        ax.set_ylim(0, ylim)

        # Bars
        positions = [-bar_width / 2, bar_width / 2]
        ax.bar(positions[0], jjk_vals[i], width=bar_width, color=colors[0], zorder=3)
        ax.bar(positions[1], opp_vals[i], width=bar_width, color=colors[1], zorder=3)

        # Labels
        ax.text(positions[0], jjk_vals[i] + ylim * 0.03, format_val(jjk_vals[i], i),
                ha='center', va='bottom', color='white', fontsize=6, zorder=5)
        ax.text(positions[1], opp_vals[i] + ylim * 0.03, format_val(opp_vals[i], i),
                ha='center', va='bottom', color='white', fontsize=6, zorder=5)

        ax.plot(positions[0], avg_jjk[i], marker='v', color='#7A0F20', markeredgecolor='black', markersize=6, zorder=4)
        ax.plot(positions[1], avg_opp[i], marker='v', color='#B89E00', markeredgecolor='black', markersize=6, zorder=4)

        # Axis & labels
        ax.set_xticks([])
        ax.set_xlabel(params[i], color='white', fontsize=6, fontweight='normal', labelpad=6)
        ax.tick_params(axis='y', left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Custom legend
    handles = [
        plt.Line2D([0], [0], color=colors[0], lw=6, label='JJK'),
        plt.Line2D([0], [0], color=colors[1], lw=6, label=team_name),
        plt.Line2D([0], [0], marker='v', color=avg_colors[0], linestyle='None',
               markersize=7, markeredgecolor='black', label='JJK avg.'),
        plt.Line2D([0], [0], marker='v', color=avg_colors[1], linestyle='None',
               markersize=7, markeredgecolor='black', label=f'{team_name} avg.')
    ]
    leg = fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.05))
    for text in leg.get_texts():
        text.set_color('white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

import cmasher as cmr

def plot_event_heatmap(uploaded_dfs, event_name, selected_half="Full Match", team="JJK"):
    pitch = Pitch(pitch_color='#122c3d', line_color='white', line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 7))

    all_x = []
    all_y = []

    for df in uploaded_dfs:
        # Filter by half
        if selected_half == "1st Half":
            df = df[df["Half"] == "1st Half"]
        elif selected_half == "2nd Half":
            df = df[df["Half"] == "2nd Half"]

         # Event filtering
        if team == "JJK":
            if event_name == "Shots":
                filtered = df[df["Name"] == "Shot"]
            elif event_name == "Box Entries (Pass)":
                filtered = df[(df["Name"] == "Box Entry") & (df["Primary"] == "Pass")]
            elif event_name == "Defensive Duels":
                filtered = df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Defensive Duel")]
            elif event_name == "Interceptions":
                filtered = df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Interception")]
            elif event_name == "Fouls":
                filtered = df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Foul")]
            else:
                continue
        elif team == "OPP":
            if event_name == "Shots":
                filtered = df[df["Name"] == "Shot OPP"]
            elif event_name == "Box Entries (Pass)":
                filtered = df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Pass")]
            elif event_name == "Defensive Duels":
                filtered = df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Defensive Duel")]
            elif event_name == "Interceptions":
                filtered = df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Interception")]
            elif event_name == "Fouls":
                filtered = df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Foul")]
            else:
                continue
        else:
            continue

        if 'X' in filtered.columns and 'Y' in filtered.columns:
            x = filtered['X'].dropna() * 1.2
            y = filtered['Y'].dropna() * 0.8

            # Only mirror OPP shots, not other events
            if team == "OPP" and event_name == "Shots":
                x = 120 - x
                y = 80 - y
            all_x.extend(x)
            all_y.extend(y)

    if all_x and all_y:
        pitch.kdeplot(
            x=all_x,
            y=all_y,
            ax=ax,
            cmap=cmr.amber if team == "OPP" else cmr.sunburst,
            fill=True,
            levels=125,
            shade_lowest=False,
            alpha=0.85,
            bw_adjust=0.3,
        )
    else:
        ax.text(60, 40, "No Data", ha='center', va='center', fontsize=20, color='white')

    fig.patch.set_facecolor('#122c3d')
    return fig

def get_kpi_development(uploaded_files, xgb_model, expected_cols, selected_kpi, team, selected_half="Full Match"):
    # Extract date from filename like "TeamName 10.5.csv"
    def extract_date(file):
        match = re.search(r'(\d{1,2})\.(\d{1,2})(?=\D*$)', file.name)
        if match:
            day, month = match.groups()
            return (int(month), int(day))
        return (0, 0)

    sorted_files = sorted(uploaded_files, key=extract_date)

    kpi_names = [
        'Shots', 'xG', 'xG/Shot', 'AT%', 'AT/BE (s)',
        'High Recoveries', 'Succ. Passes Behind', 'Box Entries'
    ]

    if selected_kpi not in kpi_names:
        return [], [], []

    values = []
    labels = []

    for f in sorted_files:
        f.seek(0)
        df = pd.read_csv(f)
        # Filter by half if not full match
        if selected_half == "1st Half":
            df = df[df["Half"] == "1st Half"]
        elif selected_half == "2nd Half":
            df = df[df["Half"] == "2nd Half"]
        jjk_vals, opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
        val = jjk_vals if team == "JJK" else opp_vals
        if val is not None:
            idx = kpi_names.index(selected_kpi)
            values.append(val[idx])
            match = re.search(r'^(.*?)\s*(\d{1,2})\.(\d{1,2})(?=\D*$)', f.name)
            if match:
                team_name, day, month = match.groups()
                label = f"{team_name.strip()} {int(day):02}.{int(month):02}"
            else:
                label = f.name.split('.')[0]
            labels.append(label)
    return labels, values, selected_kpi
