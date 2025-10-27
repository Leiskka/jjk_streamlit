import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Radar, VerticalPitch
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cmasher as cmr
import re

# Translation function for Finnish to English data
def translate_finnish_to_english_data(df):
    """Translate Finnish column values back to English for processing"""
    # Create reverse translation dictionary (Finnish -> English)
    finnish_to_english = {
        "Laukaus": "Shot",
        "Laukaus VAS": "Shot OPP",
        "Vapaapotku VAS": "Free Kick OPP",
        "Kulmapotku VAS": "Corner Kick OPP",
        "Taakse VAS": "Behind OPP",
        "Maalipotku VAS": "Goal Kick OPP",
        "Maalipotku": "Goal Kick",
        "Murtautuminen boksiin VAS": "Box Entry OPP",
        "Box Entry VAS": "Box Entry OPP",
        "Hyökkäysaika VAS": "Attacking Time OPP",
        "Pallonriisto VAS": "Defensive Action OPP",
        "Vapaapotku": "Free Kick",
        "Kulmapotku": "Corner Kick",
        "Taakse": "Behind",
        "Murtautuminen boksiin": "Box Entry",
        "Hyökkäysaika": "Attacking Time",
        "Pallonriisto": "Defensive Action",
        "Aktiivinen peliaika": "Total Time",
        "1. puoliaika": "1. puoliaika",
        "2. puoliaika": "2. puoliaika",
        "Maali": "Goal",
        "Kohti maalia": "On Target",
        "Ohi": "Off Target",
        "Blokattu": "Blocked",
        "Oikea jalka": "Right Foot",
        "Vasen jalka": "Left Foot",
        "Pää": "Head",
        "Normaali pelitilanne": "Regular Play",
        "Vastahyökkäyksestä": "From Counter",
        "Vapaapotkusta": "From Free Kick",
        "Rajaheitosta": "From Throw In",
        "Kulmapotkusta": "From Corner",
        "Maalipotkusta": "From Goal Kick",
        "Vapaapotku (suora laukaus)": "Direct Free Kick",
        "Rangaistuspotku": "Penalty",
        "Onnistunut": "Successful",
        "Epäonnistunut": "Unsuccessful",
        "Paitsio": "Offside",
        "Puolustuskaksinkamppailu": "Defensive Duel",
        "Syötönkatko": "Interception",
        "Kuljetus": "Dribble",
        "Syöttö": "Pass",
        "Vasemmalta laidalta": "Left Flank",
        "Oikealta laidalta": "Right Flank",
        "Oikealta": "Right",
        "Vasemmalta": "Left",
        "Keskeltä": "Central"
    }
    
    # Apply translation to relevant columns
    if 'Name' in df.columns:
        df['Name'] = df['Name'].map(finnish_to_english).fillna(df['Name'])
    if 'Primary' in df.columns:
        df['Primary'] = df['Primary'].map(finnish_to_english).fillna(df['Primary'])
    if 'Secondary' in df.columns:
        df['Secondary'] = df['Secondary'].map(finnish_to_english).fillna(df['Secondary'])
    if 'Detail' in df.columns:
        df['Detail'] = df['Detail'].map(finnish_to_english).fillna(df['Detail'])
    
    return df

# Helper function to load and translate data from uploaded file
def load_data_from_upload_analysis(uploaded_file):
    """Load and translate data from uploaded file - analysis module version"""
    uploaded_file.seek(0)  # Ensure pointer is at the start
    df = pd.read_csv(uploaded_file)
    # Translate Finnish data to English for processing
    df = translate_finnish_to_english_data(df)
    filename = uploaded_file.name
    team_name = os.path.splitext(filename)[0]
    team_name = team_name.split(" ")[0]
    return df, team_name

GOAL_X, GOAL_Y = 120, 40

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
    shots_df['xG'] = xgb_model.predict_proba(shots_df[expected_cols])[:, 1] * 1.5
    shots_df.loc[is_penalty, 'xG'] = 0.78 * 1.5
    return shots_df

def format_seconds(seconds):
    minutes = int(seconds) // 60
    sec = int(seconds) % 60
    return f"{minutes:02}:{sec:02}"

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

# === KPI Extraction ===
def extract_full_match_stats(df, xgb_model, expected_cols, team_name):
    # Helper functions
    def count_goals(df): return len(df[(df["Name"] == "Shot") & (df["Primary"] == "Goal")])
    def count_shots(df): return len(df[df["Name"] == "Shot"])
    def count_box_entries(df): return len(df[df["Name"] == "Box Entry"])
    def count_box_entries_pass(df): return len(df[(df["Name"] == "Box Entry") & (df["Primary"] == "Pass")])
    def count_box_entries_dribble(df): return len(df[(df["Name"] == "Box Entry") & (df["Primary"] == "Dribble")])
    def count_passes_behind(df): return len(df[df["Name"] == "Behind"])
    def count_successful_passes_behind(df): return len(df[(df["Name"] == "Behind") & (df["Primary"] == "Successful")])
    def count_offsides(df): return len(df[(df["Name"] == "Behind") & (df["Primary"] == "Offside")])
    def count_high_recoveries(df): return len(df[(df["Name"] == "Defensive Action") & (df["X"] > 66) & (df["Primary"] != "Foul")])
    def count_fouls(df): return len(df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Foul")])
    def count_opp_goals(df): return len(df[(df["Name"] == "Shot OPP") & (df["Primary"] == "Goal")])
    def count_opp_shots(df): return len(df[df["Name"] == "Shot OPP"])
    def count_opp_box_entries(df): return len(df[df["Name"] == "Box Entry OPP"])
    def count_opp_box_entries_pass(df): return len(df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Pass")])
    def count_opp_box_entries_dribble(df): return len(df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Dribble")])
    def count_opp_passes_behind(df): return len(df[df["Name"] == "Behind OPP"])
    def count_opp_successful_passes_behind(df): return len(df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Successful")])
    def count_opp_offsides(df): return len(df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Offside")])
    def count_opp_high_recoveries(df): return len(df[(df["Name"] == "Defensive Action OPP") & (df["X"] < 33) & (df["Primary"] != "Foul")])
    def count_opp_fouls(df): return len(df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Foul")])

    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    xG = shots['xG'].sum()
    xG_opp = shots_opp['xG'].sum()
    goals = count_goals(df)
    num_shots = count_shots(df)
    conversion = goals / num_shots * 100 if num_shots > 0 else 0  # Convert to percentage
    box_entries = count_box_entries(df)
    box_entries_pass = count_box_entries_pass(df)
    box_entries_dribble = count_box_entries_dribble(df)
    xG_per_shot = xG / num_shots if num_shots > 0 else 0
    passes_behind = count_passes_behind(df)
    successful_passes_behind = count_successful_passes_behind(df)
    passes_behind_success_rate = (successful_passes_behind / passes_behind * 100) if passes_behind > 0 else 0
    offsides = count_offsides(df)

    high_recoveries = count_high_recoveries(df)
    fouls = count_fouls(df)
    goals_opp = count_opp_goals(df)
    num_shots_opp = count_opp_shots(df)
    conversion_opp = goals_opp / num_shots_opp * 100 if num_shots_opp > 0 else 0  # Convert to percentage
    box_entries_opp = count_opp_box_entries(df)
    box_entries_pass_opp = count_opp_box_entries_pass(df)
    box_entries_dribble_opp = count_opp_box_entries_dribble(df)
    xG_per_shot_opp = xG_opp / num_shots_opp if num_shots_opp > 0 else 0
    passes_behind_opp = count_opp_passes_behind(df)
    successful_passes_behind_opp = count_opp_successful_passes_behind(df)
    passes_behind_success_rate_opp = (successful_passes_behind_opp / passes_behind_opp * 100) if passes_behind_opp > 0 else 0
    offsides_opp = count_opp_offsides(df)
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
        f"{conversion:.1f}%" if num_shots > 0 else "N/A",
        box_entries,
        box_entries_pass,
        box_entries_dribble,
        format_seconds(attacking_time),
        f"{att_percent:.1f}%",
        f"{attacking_time / box_entries:.1f}" if box_entries > 0 else "N/A",
        passes_behind,
        successful_passes_behind,
        f"{passes_behind_success_rate:.1f}%",
        offsides,
        high_recoveries
    ]
    opp_stats = [
        goals_opp,
        f"{xG_opp:.2f}",
        num_shots_opp,
        f"{xG_per_shot_opp:.2f}" if num_shots_opp > 0 else "N/A",
        f"{conversion_opp:.1f}%" if num_shots_opp > 0 else "N/A",
        box_entries_opp,
        box_entries_pass_opp,
        box_entries_dribble_opp,
        format_seconds(attacking_time_opp),
        f"{opp_att_percent:.1f}%",
        f"{attacking_time_opp / box_entries_opp:.1f}" if box_entries_opp > 0 else "N/A",
        passes_behind_opp,
        successful_passes_behind_opp,
        f"{passes_behind_success_rate_opp:.1f}%",
        offsides_opp,
        high_recoveries_opp
    ]
    row_labels = [
        "Maalit", "xG", "Laukaukset", "xG / Laukaus", "Tehokkuus", "Box Entryt",
        "→ Syöttäen", "→ Kuljettaen", "Hyökkäysaika", "HA %", "HA / BE (s)", 
        "Syötöt taakse", "Onn. syötöt taakse", "Onn. syötöt taakse %", "Paitsiot", "Korkeat riistot"
    ]
    return row_labels, jjk_stats, opp_stats

def generate_kpi_table(row_labels, jjk_stats, opp_stats, team_name):
    import matplotlib.pyplot as plt

    def fmt(val):
        try:
            import re
            # If it's already a formatted string (contains % or N/A or time), return as is
            if isinstance(val, str) and ('%' in val or 'N/A' in val or ':' in val):
                return str(val)
            # Preserve strings that are already numeric with exactly 2 decimals (e.g. "1.23")
            if isinstance(val, str) and re.match(r'^-?\d+\.\d{2}$', val):
                return val
            # If it's a whole number, return as integer
            float_val = float(val)
            if float_val == int(float_val):
                return str(int(float_val))
            # Otherwise return with default 1 decimal
            return f"{float_val:.1f}"
        except (ValueError, TypeError):
            return str(val)

    fig, ax = plt.subplots(figsize=(11, 12), dpi=100)
    fig.patch.set_facecolor('#122c3d')
    ax.set_facecolor('#122c3d')
    
    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    table_data = [[label, fmt(jjk), fmt(opp)] for label, jjk, opp in zip(row_labels, jjk_stats, opp_stats)]
    table = ax.table(
        cellText=table_data,
        colLabels=[" ", "JJK", team_name],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(25)
    table.scale(1, 3)
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(22)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#A6192E')
        else:
            cell.set_fontsize(18)
            cell.set_facecolor('#122c3d' if row % 2 == 0 else "#0b1c26")
            cell.set_text_props(color='#FFFFFF')
    fig.tight_layout(pad=1)
    return fig

# === Radar Chart ===
def generate_radar_chart(jjk_vals, opp_vals):
    params = [
        'Laukaukset', 'xG', 'xG/Laukaus', 'Box Entryt', 'HA%', 'HA/BE (s)',
        'Korkeat riistot', 'Onn. syötöt taakse', 'Onn. syötöt taakse %'
    ]
    # Set min/max for normalization - only using match values
    low = [0, 0, 0.07, 0, 0, 20, 0, 0, 0]
    high = [
        max(jjk_vals[0], opp_vals[0]) + 5,
        max(jjk_vals[1], opp_vals[1]) + 0.5,
        max(jjk_vals[2], opp_vals[2]) + 0.03,
        max(jjk_vals[3], opp_vals[3]) + 5,
        max(jjk_vals[4], opp_vals[4]) + 5,
        max(jjk_vals[5], opp_vals[5]) + 5,
        max(jjk_vals[6], opp_vals[6]) + 5,
        max(jjk_vals[7], opp_vals[7]) + 5,
        max(jjk_vals[8], opp_vals[8]) + 5
    ]
    lower_is_better = ['HA/BE (s)']
    radar = Radar(params, low, high, lower_is_better=lower_is_better,
                  round_int = [True, False, False, True, False, False, True, True, False],
                  num_rings=5, ring_width=1,
                  center_circle_radius=1)
    fig, ax = radar.setup_axis(figsize=(9,9))
    radar.draw_circles(ax=ax, facecolor="#28252c", edgecolor="#39353f", lw=1.5)
    
    # Draw only match values as solid lines without fill
    radar.draw_radar_solid(jjk_vals, ax=ax, kwargs={'facecolor': 'none', 'alpha': 1.0, 'edgecolor': '#A6192E', 'lw': 3})
    radar.draw_radar_solid(opp_vals, ax=ax, kwargs={'facecolor': 'none', 'alpha': 1.0, 'edgecolor': '#FFD100', 'lw': 3})
    
    radar.draw_range_labels(ax=ax, fontsize=15, color='#fcfcfc')
    radar.draw_param_labels(ax=ax, fontsize=15, color='#fcfcfc')
    
    fig.patch.set_facecolor('#122c3d')
    ax.set_facecolor('#122c3d')
    plt.tight_layout()
    return fig

# === Cumulative xG Plot ===
def plot_cumulative_xg(df, xgb_model, expected_cols, team_name, selected_half="Täysi ottelu"):
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

    # Set x-axis range based on selected_half and actual data
    if selected_half in ["1st Half", "1. puoliaika"]:
        x_min, x_max = 0, 2700
    elif selected_half in ["2nd Half", "2. puoliaika"]:
        x_min, x_max = 2700, 5400
    else:
        # For full match, make x-axis flexible to handle extra time
        x_min = 0
        # Find the maximum time from all events in the match
        max_event_time = 0
        if not shots.empty and len(shots) > 1:  # Check if there are shots beyond 0:00
            max_event_time = max(max_event_time, shots['time'].max())
        if not shots_opp.empty and len(shots_opp) > 1:  # Check if there are shots beyond 0:00
            max_event_time = max(max_event_time, shots_opp['time'].max())
        # Also check goal times to ensure all goals are visible
        goal_times = df[(df['Name'] == 'Shot') & (df['Primary'] == 'Goal')]['End'].apply(parse_time)
        goal_times_opp = df[(df['Name'] == 'Shot OPP') & (df['Primary'] == 'Goal')]['End'].apply(parse_time)
        if not goal_times.empty:
            max_event_time = max(max_event_time, goal_times.max())
        if not goal_times_opp.empty:
            max_event_time = max(max_event_time, goal_times_opp.max())
        # Set x_max to at least 90 minutes, but extend if there's extra time
        x_max = max(5400, max_event_time + 300)  # Add 5 minutes buffer

    # Add final points at x_max to extend lines to the end of the time axis
    if not shots.empty and len(shots) > 1:
        final_cum_xg = shots['cum_xG'].iloc[-1]  # Get the last cumulative xG value
        shots = pd.concat([
            shots,
            pd.DataFrame({'time': [x_max], 'cum_xG': [final_cum_xg]})
        ], ignore_index=True)
    else:
        # If no shots, just add a point at x_max with 0 cumulative xG
        shots = pd.concat([
            shots,
            pd.DataFrame({'time': [x_max], 'cum_xG': [0]})
        ], ignore_index=True)

    if not shots_opp.empty and len(shots_opp) > 1:
        final_cum_xg_opp = shots_opp['cum_xG'].iloc[-1]  # Get the last cumulative xG value
        shots_opp = pd.concat([
            shots_opp,
            pd.DataFrame({'time': [x_max], 'cum_xG': [final_cum_xg_opp]})
        ], ignore_index=True)
    else:
        # If no shots, just add a point at x_max with 0 cumulative xG
        shots_opp = pd.concat([
            shots_opp,
            pd.DataFrame({'time': [x_max], 'cum_xG': [0]})
        ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=100)
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
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x//60):02}:{int(x%60):02}")
    )
    ax.step(shots['time'], shots['cum_xG'], where='post', label='JJK', color='#A6192E', linewidth=2)
    ax.step(shots_opp['time'], shots_opp['cum_xG'], where='post', label=team_name, color='#FFD100', linewidth=2)
    # Set y-axis limit to +1 above the highest cumulative xG value
    max_cum_xg = max(shots['cum_xG'].max(), shots_opp['cum_xG'].max())
    ax.set_ylim(0, max_cum_xg + 1)
    
    # Set custom y-axis ticks excluding 0.0
    y_max = max_cum_xg + 1
    if y_max <= 1.0:
        y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif y_max <= 2.0:
        y_ticks = [0.5, 1.0, 1.5, 2.0]
    elif y_max <= 3.0:
        y_ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    else:
        y_ticks = [i for i in range(1, int(y_max) + 1)]
    
    # Filter ticks to only show those within the y-axis range and exclude 0
    y_ticks = [tick for tick in y_ticks if 0 < tick <= y_max]
    ax.set_yticks(y_ticks)
    
    ax.grid(True, axis='y', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white', labelsize=18)
    ax.tick_params(axis='y', colors='white', labelsize=22)
    ax.set_xlabel(' ', color='white', fontsize=5)
    
    plt.tight_layout(pad=3)
    return fig

def plot_combined_pitch_vertical(df, xgb_model, expected_cols, team_name):
    """Vertical pitch view showing only shots for each team."""
    # Optimize figsize for better space usage - taller and narrower
    pitch = VerticalPitch(pitch_color='#122c3d', line_color='white')
    fig, ax = pitch.draw(figsize=(4.8, 15))
    fig.patch.set_facecolor('#122c3d')

    # JJK Shots
    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    shots['plot_x'] = shots['location_x']
    shots['plot_y'] = shots['location_y']
    shots_no_pen = shots[~shots['Secondary'].str.contains("Penalty", na=False)]
    
    # Separate goals and non-goals for different colors
    goals_jjk = shots_no_pen[shots_no_pen['Primary'] == 'Goal']
    non_goals_jjk = shots_no_pen[shots_no_pen['Primary'] != 'Goal']
    
    # Plot non-goals in team color
    if not non_goals_jjk.empty:
        pitch.scatter(
            non_goals_jjk['plot_x'], non_goals_jjk['plot_y'],
            s=non_goals_jjk['xG'] * 400, color='#A6192E', edgecolors='black', alpha=0.9, ax=ax
        )
    
    # Plot goals in green
    if not goals_jjk.empty:
        pitch.scatter(
            goals_jjk['plot_x'], goals_jjk['plot_y'],
            s=goals_jjk['xG'] * 400, color='green', edgecolors='black', alpha=0.9, ax=ax
        )

    # Opponent Shots
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    shots_opp['plot_x'] = 120 - shots_opp['location_x']
    shots_opp['plot_y'] = 80 - shots_opp['location_y']
    shots_opp_no_pen = shots_opp[~shots_opp['Secondary'].str.contains("Penalty", na=False)]
    
    # Separate goals and non-goals for different colors
    goals_opp = shots_opp_no_pen[shots_opp_no_pen['Primary'] == 'Goal']
    non_goals_opp = shots_opp_no_pen[shots_opp_no_pen['Primary'] != 'Goal']
    
    # Plot non-goals in team color
    if not non_goals_opp.empty:
        pitch.scatter(
            non_goals_opp['plot_x'], non_goals_opp['plot_y'],
            s=non_goals_opp['xG'] * 400, color='#FFD100', edgecolors='black', alpha=0.9, ax=ax
        )
    
    # Plot goals in green
    if not goals_opp.empty:
        pitch.scatter(
            goals_opp['plot_x'], goals_opp['plot_y'],
            s=goals_opp['xG'] * 400, color='green', edgecolors='black', alpha=0.9, ax=ax
        )

    # Optimize layout to minimize empty space around the pitch
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02)
    
    return fig

def debug_data_columns(df):
    """Debug function to print available column names and unique values in 'Name' column"""
    print("Available columns:", df.columns.tolist())
    if 'Name' in df.columns:
        unique_names = df['Name'].unique()
        print("Unique 'Name' values:", unique_names)
        # Check for specific patterns
        for name in unique_names:
            if any(keyword in name.lower() for keyword in ['behind', 'taakse', 'attack', 'hyökk', 'defensive', 'puolust']):
                print(f"Found potential match: {name}")
    return df

def extract_radar_kpis_robust(df, xgb_model, expected_cols):
    """Extract radar KPIs with support for both English and Finnish column names"""
    
    # Define mapping for English and Finnish terms
    name_mappings = {
        'shot': ['Shot', 'Laukaus'],
        'shot_opp': ['Shot OPP', 'Laukaus VAS'],
        'box_entry': ['Box Entry', 'Murtautuminen boksiin'],
        'box_entry_opp': ['Box Entry OPP', 'Murtautuminen boksiin VAS'],
        'behind': ['Behind', 'Taakse'],
        'behind_opp': ['Behind OPP', 'Taakse VAS'],
        'defensive_action': ['Defensive Action', 'Puolustustoiminto'],
        'defensive_action_opp': ['Defensive Action OPP', 'Puolustustoiminto VAS'],
        'attacking_time': ['Attacking Time', 'Hyökkäysaika'],
        'attacking_time_opp': ['Attacking Time OPP', 'Hyökkäysaika VAS'],
        'total_time': ['Total Time', 'Kokonaisaika']
    }
    
    def find_matching_rows(df, name_options):
        """Find rows that match any of the name options"""
        for name in name_options:
            matching_rows = df[df['Name'] == name]
            if not matching_rows.empty:
                return matching_rows
        return df[df['Name'] == name_options[0]]  # Return empty df with correct structure
    
    # JJK
    shots = process_shots(find_matching_rows(df, name_mappings['shot']), expected_cols, xgb_model)
    xG = shots['xG'].sum()
    num_shots = len(shots)
    xG_per_shot = xG / num_shots if num_shots > 0 else 0
    box_entries = len(find_matching_rows(df, name_mappings['box_entry']))
    
    defensive_actions = find_matching_rows(df, name_mappings['defensive_action'])
    high_recoveries = len(defensive_actions[(defensive_actions['X'] > 66) & (defensive_actions['Primary'] != "Foul")])
    
    behind_df = find_matching_rows(df, name_mappings['behind'])
    passes_behind = len(behind_df)
    successful_passes_behind = len(behind_df[behind_df['Primary'] == 'Successful'])
    passes_behind_success_rate = (successful_passes_behind / passes_behind * 100) if passes_behind > 0 else 0
    
    attacking_time_df = find_matching_rows(df, name_mappings['attacking_time'])
    attacking_time = attacking_time_df['Duration'].sum()
    
    total_time_df = find_matching_rows(df, name_mappings['total_time'])
    total_time = total_time_df['Duration'].sum()
    
    att_percent = (attacking_time / total_time * 100) if total_time > 0 else 0
    att_per_box = attacking_time / box_entries if box_entries > 0 else 0

    jjk_kpis = [
        num_shots,
        xG,
        xG_per_shot,
        box_entries,
        att_percent,
        att_per_box,
        high_recoveries,
        successful_passes_behind,
        passes_behind_success_rate
    ]
    
    # OPP
    shots_opp = process_shots(find_matching_rows(df, name_mappings['shot_opp']), expected_cols, xgb_model)
    xG_opp = shots_opp['xG'].sum()
    num_shots_opp = len(shots_opp)
    xG_per_shot_opp = xG_opp / num_shots_opp if num_shots_opp > 0 else 0
    box_entries_opp = len(find_matching_rows(df, name_mappings['box_entry_opp']))
    
    defensive_actions_opp = find_matching_rows(df, name_mappings['defensive_action_opp'])
    high_recoveries_opp = len(defensive_actions_opp[(defensive_actions_opp['X'] < 33) & (defensive_actions_opp['Primary'] != "Foul")])
    
    behind_opp_df = find_matching_rows(df, name_mappings['behind_opp'])
    passes_behind_opp = len(behind_opp_df)
    successful_passes_behind_opp = len(behind_opp_df[behind_opp_df['Primary'] == 'Successful'])
    passes_behind_success_rate_opp = (successful_passes_behind_opp / passes_behind_opp * 100) if passes_behind_opp > 0 else 0
    
    attacking_time_opp_df = find_matching_rows(df, name_mappings['attacking_time_opp'])
    attacking_time_opp = attacking_time_opp_df['Duration'].sum()
    
    att_percent_opp = (attacking_time_opp / total_time * 100) if total_time > 0 else 0
    att_per_box_opp = attacking_time_opp / box_entries_opp if box_entries_opp > 0 else 0

    opp_kpis = [
        num_shots_opp,
        xG_opp,
        xG_per_shot_opp,
        box_entries_opp,
        att_percent_opp,
        att_per_box_opp,
        high_recoveries_opp,
        successful_passes_behind_opp,
        passes_behind_success_rate_opp
    ]
    return jjk_kpis, opp_kpis

def detect_data_language(df):
    """Detect if data uses English or Finnish column names"""
    if 'Name' not in df.columns:
        return 'english'  # Default fallback
    
    unique_names = df['Name'].unique()
    english_indicators = ['Shot', 'Behind', 'Box Entry', 'Defensive Action', 'Attacking Time']
    finnish_indicators = ['Laukaus', 'Taakse', 'Murtautuminen', 'Puolustus', 'Hyökkäys']
    
    english_count = sum(1 for name in unique_names if any(eng in str(name) for eng in english_indicators))
    finnish_count = sum(1 for name in unique_names if any(fin in str(name) for fin in finnish_indicators))
    
    return 'finnish' if finnish_count > english_count else 'english'

def extract_radar_kpis_smart(df, xgb_model, expected_cols):
    """Smart extraction that automatically detects language and uses appropriate logic"""
    language = detect_data_language(df)
    
    if language == 'finnish':
        return extract_radar_kpis_robust(df, xgb_model, expected_cols)
    else:
        return extract_radar_kpis(df, xgb_model, expected_cols)

def extract_radar_kpis(df, xgb_model, expected_cols):
    # JJK
    shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
    xG = shots['xG'].sum()
    num_shots = len(shots)
    xG_per_shot = xG / num_shots if num_shots > 0 else 0
    box_entries = len(df[df['Name'] == 'Box Entry'])
    high_recoveries = len(df[(df['Name'] == 'Defensive Action') & (df['X'] > 66) & (df['Primary'] != "Foul")])
    passes_behind = len(df[df['Name'] == 'Behind'])
    successful_passes_behind = len(df[(df['Name'] == 'Behind') & (df['Primary'] == 'Successful')])
    passes_behind_success_rate = (successful_passes_behind / passes_behind * 100) if passes_behind > 0 else 0
    attacking_time = df[df['Name'] == 'Attacking Time']['Duration'].sum()
    total_time = df[df['Name'] == 'Total Time']['Duration'].sum()
    att_percent = (attacking_time / total_time * 100) if total_time > 0 else 0
    att_per_box = attacking_time / box_entries if box_entries > 0 else 0

    jjk_kpis = [
        num_shots,
        xG,
        xG_per_shot,
        box_entries,
        att_percent,
        att_per_box,
        high_recoveries,
        successful_passes_behind,
        passes_behind_success_rate
    ]
    # OPP
    shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
    xG_opp = shots_opp['xG'].sum()
    num_shots_opp = len(shots_opp)
    xG_per_shot_opp = xG_opp / num_shots_opp if num_shots_opp > 0 else 0
    box_entries_opp = len(df[df['Name'] == 'Box Entry OPP'])
    high_recoveries_opp = len(df[(df['Name'] == 'Defensive Action OPP') & (df['X'] < 33) & (df['Primary'] != "Foul")])
    passes_behind_opp = len(df[df['Name'] == 'Behind OPP'])
    successful_passes_behind_opp = len(df[(df['Name'] == 'Behind OPP') & (df['Primary'] == 'Successful')])
    passes_behind_success_rate_opp = (successful_passes_behind_opp / passes_behind_opp * 100) if passes_behind_opp > 0 else 0
    attacking_time_opp = df[df['Name'] == 'Attacking Time OPP']['Duration'].sum()
    att_percent_opp = (attacking_time_opp / total_time * 100) if total_time > 0 else 0
    att_per_box_opp = attacking_time_opp / box_entries_opp if box_entries_opp > 0 else 0

    opp_kpis = [
        num_shots_opp,
        xG_opp,
        xG_per_shot_opp,
        box_entries_opp,
        att_percent_opp,
        att_per_box_opp,
        high_recoveries_opp,
        successful_passes_behind_opp,
        passes_behind_success_rate_opp
    ]
    return jjk_kpis, opp_kpis

def display_jjk_kpi_stats_vertical(jjk_vals, opp_vals, avg_jjk, avg_opp, std_jjk, std_opp, team_name):
    """Display JJK and opponent KPI z-scores as vertical side-by-side arrows"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    # Update parameter names to match KPI table order and add centered/multi-line names
    params = [
        'Laukaukset', 'xG', 'xG/Laukaus', 'Box Entryt', 'HA%', 'HA/BE (s)', 
        'Korkeat\nriistot', 'Onn. syötöt\ntaakse', 'Onn. syötöt\ntaakse %'
    ]
    
    colors = ['#A6192E', '#FFD100']  # JJK, OPP
    bg_color = '#122c3d'
    
    # Convert all inputs to float arrays
    jjk_vals = np.array(jjk_vals, dtype=float)
    opp_vals = np.array(opp_vals, dtype=float)
    avg_jjk = np.array(avg_jjk, dtype=float)
    avg_opp = np.array(avg_opp, dtype=float)
    std_jjk = np.array(std_jjk, dtype=float)
    std_opp = np.array(std_opp, dtype=float)
    
    # Avoid division by zero for std
    std_jjk = np.where(std_jjk == 0, 1e-8, std_jjk)
    std_opp = np.where(std_opp == 0, 1e-8, std_opp)
    
    # Calculate z-scores for both teams
    z_jjk = (jjk_vals - avg_jjk) / std_jjk
    z_opp = (opp_vals - avg_opp) / std_opp
    
    # DEBUG: Create detailed calculation info for shots (index 0)
    debug_text = f"""🔍 DEBUG: Match Report Z-Score Calculation
Match: {team_name}
Shots - Raw value: {jjk_vals[0]:.3f}
Shots - Average: {avg_jjk[0]:.3f}
Shots - Std Dev: {std_jjk[0]:.3f}
Shots - Z-score: {z_jjk[0]:.3f}
Calculation: ({jjk_vals[0]:.3f} - {avg_jjk[0]:.3f}) / {std_jjk[0]:.3f} = {z_jjk[0]:.3f}"""
    
    # Create vertical figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(9, 11), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Set x-axis limits to fixed range
    ax.set_xlim(-2, 2)
    
    # Create y positions for arrows with more space between metrics
    n_params = len(params)
    y_spacing = 1.2  # Increased spacing between metrics
    y = np.arange(n_params * y_spacing, 0, -y_spacing)  # Count down from top with more space
    
    # Larger arrows and elements for better readability
    height_offset = 0.18  # Slightly increased
    arrow_height = 0.08   # Increased arrow height
    arrow_head_width = 0.25  # Increased arrow head width
    
    # Draw horizontal arrows for JJK (top position for each KPI)
    for i, z_val in enumerate(z_jjk):
        z_clamped = np.clip(z_val, -2.0, 2.0)
        ypos = y[i] + height_offset

        if abs(z_val) < 0.25:
            # Draw a vertical dash for near-average performance
            ax.plot(
                z_clamped, ypos,
                marker='o', markersize=12,  # Increased marker size
                color=colors[0],
                markeredgewidth=1.5,  # Increased edge width
                markeredgecolor='black',
                alpha=1, zorder=10
            )
        else:
            # Draw horizontal arrow
            ax.arrow(
                0, ypos, z_clamped, 0,
                width=arrow_height, head_width=arrow_head_width, head_length=0.2,
                fc=colors[0], ec='black', alpha=1, linewidth=1, zorder=9, length_includes_head=True
            )

    # Draw horizontal arrows for OPP (bottom position for each KPI)
    for i, z_val in enumerate(z_opp):
        z_clamped = np.clip(z_val, -2.0, 2.0)
        ypos = y[i] - height_offset

        if abs(z_val) < 0.5:
            # Draw a vertical dash for near-average performance
            ax.plot(
                z_clamped, ypos,
                marker='o', markersize=12,  # Increased marker size
                color=colors[1],
                markeredgewidth=1.5,  # Increased edge width
                markeredgecolor='black',
                alpha=1, zorder=10
            )
        else:
            ax.arrow(
                0, ypos, z_clamped, 0,
                width=arrow_height, head_width=arrow_head_width, head_length=0.2,
                fc=colors[1], ec='black', alpha=1, linewidth=1, zorder=9, length_includes_head=True
            )

    # Add vertical reference lines for z-score interpretation
    ax.axvline(x=1, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=0, color='white', linestyle='-', alpha=0.6, linewidth=1.5)
    ax.axvline(x=-1, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Set y-axis limits first to establish the plot area
    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
    
    # Position y-axis labels outside the plot area on the left with larger font
    ax.set_yticks(y)
    ax.set_yticklabels(params, color='white', fontsize=14, ha='right', fontweight='bold')  # Increased font size and weight
    ax.tick_params(axis='y', pad=30)  # Increased padding
    
    ax.set_xlabel('Z-arvo', color='white', fontsize=16, fontweight='bold')  # Increased font size
    ax.tick_params(axis='x', colors='white', labelsize=13)  # Increased font size
    ax.tick_params(axis='y', colors='white', labelsize=13)  # Increased font size
    
    # Custom gridlines only at 0, 1, 2, -1, -2
    gridline_positions = [-2, -1, 0, 1, 2]
    for pos in gridline_positions:
        ax.axvline(x=pos, color='white', alpha=0.4, linestyle='-', linewidth=0.7)
    
    # Add bottom x-axis legend for z-score interpretation
    ax2 = ax.twiny()  # Create a second x-axis at the top
    ax2.set_xlim(ax.get_xlim())  # Match the main axis limits
    
    # Set custom ticks for z-score values
    z_ticks = [-2, -1, 0, 1, 2]
    z_labels = [
        'Reilusti\n alle keskiarvon(-2)',
        'Alle keskiarvon\n(-1)', 
        'Keskiarvo\n(0)',
        'Yli keskiarvon\n(+1)',
        'Reilusti\n yli keskiarvon(+2)'
    ]
    
    ax2.set_xticks(z_ticks)
    ax2.set_xticklabels(z_labels, color='white', fontsize=13, ha='center')  # Increased font size
    ax2.tick_params(axis='x', colors='white', labelsize=13, pad=15)  # Increased font size and padding

    # Create compact legend
    plt.tight_layout()
    # Adjust layout to leave minimal space for better use of figure area
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.15, right=0.95)

    return fig, debug_text

def plot_event_heatmap(uploaded_dfs, event_name, selected_half="Täysi ottelu", team="JJK"):
    pitch = Pitch(pitch_color='#122c3d', line_color='white', line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 7))

    all_x = []
    all_y = []
    
    # For dribble box entries, collect zone data
    dribble_zones = {'left flank': 0, 'left': 0, 'center': 0, 'right': 0, 'right flank': 0}

    for df in uploaded_dfs:
        # Filter by half
        if selected_half in ["1st Half", "1. puoliaika"]:
            df = df[df["Half"].isin(["1st Half", "1. puoliaika"])]
        elif selected_half in ["2nd Half", "2. puoliaika"]:
            df = df[df["Half"].isin(["2nd Half", "2. puoliaika"])]

        # Event filtering
        if team == "JJK":
            if event_name == "Laukaukset":
                filtered = df[df["Name"] == "Shot"]
            elif event_name == "Box Entryt (syötön lähtöpiste)":
                filtered = df[(df["Name"] == "Box Entry") & (df["Primary"] == "Pass")]
            elif event_name == "Box Entryt (syötön loppupiste)":
                filtered = df[(df["Name"] == "Box Entry") & (df["Primary"] == "Pass")]
            elif event_name == "Box Entryt (kuljettaen)":
                filtered = df[(df["Name"] == "Box Entry") & (df["Primary"] == "Dribble")]
            elif event_name == "Onnistuneet syötöt taakse (syötön lähtöpiste)":
                filtered = df[(df["Name"] == "Behind") & (df["Primary"] == "Successful")]
            elif event_name == "Epäonnistuneet syötöt taakse (syötön lähtöpiste)":
                filtered = df[(df["Name"] == "Behind") & (df["Primary"] == "Unsuccessful")]
            elif event_name == "Onnistuneet syötöt taakse (syötön loppupiste)":
                filtered = df[(df["Name"] == "Behind") & (df["Primary"] == "Successful")]
            elif event_name == "Epäonnistuneet syötöt taakse (syötön loppupiste)":
                filtered = df[(df["Name"] == "Behind") & (df["Primary"] == "Unsuccessful")]
            elif event_name == "Pallonriistot":
                filtered = df[(df["Name"] == "Defensive Action")]
            elif event_name == "Pallonriistot (Puolustuskaksinkamppailut)":
                filtered = df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Defensive Duel")]
            elif event_name == "Pallonriistot (Syötönkatkot)":
                filtered = df[(df["Name"] == "Defensive Action") & (df["Primary"] == "Interception")]
            else:
                continue
        elif team == "VAS":
            if event_name == "Laukaukset":
                filtered = df[df["Name"] == "Shot OPP"]
            elif event_name == "Box Entryt (syötön lähtöpiste)":
                filtered = df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Pass")]
            elif event_name == "Box Entryt (syötön loppupiste)":
                filtered = df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Pass")]
            elif event_name == "Box Entryt (kuljettaen)":
                filtered = df[(df["Name"] == "Box Entry OPP") & (df["Primary"] == "Dribble")]
            elif event_name == "Onnistuneet syötöt taakse (syötön lähtöpiste)":
                filtered = df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Successful")]
            elif event_name == "Epäonnistuneet syötöt taakse (syötön lähtöpiste)":
                filtered = df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Unsuccessful")]
            elif event_name == "Onnistuneet syötöt taakse (syötön loppupiste)":
                filtered = df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Successful")]
            elif event_name == "Epäonnistuneet syötöt taakse (syötön loppupiste)":
                filtered = df[(df["Name"] == "Behind OPP") & (df["Primary"] == "Unsuccessful")]
            elif event_name == "Pallonriistot":
                filtered = df[(df["Name"] == "Defensive Action OPP")]
            elif event_name == "Pallonriistot (Puolustuskaksinkamppailut)":
                filtered = df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Defensive Duel")]
            elif event_name == "Pallonriistot (Syötönkatkot)":
                filtered = df[(df["Name"] == "Defensive Action OPP") & (df["Primary"] == "Interception")]
            else:
                continue
        else:
            continue


        # Handle different coordinate sources for box entries
        if event_name == "Box Entryt (syötön lähtöpiste)":
            # Use start coordinates for box entry passes
            if 'X' in filtered.columns and 'Y' in filtered.columns:
                x = filtered['X'].dropna() * 1.2
                y = filtered['Y'].dropna() * 0.8
                all_x.extend(x)
                all_y.extend(y)
        elif event_name == "Box Entryt (syötön loppupiste)":
            # Use end coordinates for box entry passes
            if 'End X' in filtered.columns and 'End Y' in filtered.columns:
                x = filtered['End X'].dropna() * 1.2
                y = filtered['End Y'].dropna() * 0.8
                all_x.extend(x)
                all_y.extend(y)
            elif 'X' in filtered.columns and 'Y' in filtered.columns:
                # Fallback to start coordinates if end coordinates not available
                x = filtered['X'].dropna() * 1.2
                y = filtered['Y'].dropna() * 0.8
                all_x.extend(x)
                all_y.extend(y)
        elif event_name in ["Onnistuneet syötöt taakse (syötön lähtöpiste)", "Epäonnistuneet syötöt taakse (syötön lähtöpiste)"]:
            # Use start coordinates for behind events
            if 'X' in filtered.columns and 'Y' in filtered.columns:
                x = filtered['X'].dropna() * 1.2
                y = filtered['Y'].dropna() * 0.8
                all_x.extend(x)
                all_y.extend(y)
        elif event_name in ["Onnistuneet syötöt taakse (syötön loppupiste)", "Epäonnistuneet syötöt taakse (syötön loppupiste)"]:
            # Use end coordinates for behind events
            if 'End X' in filtered.columns and 'End Y' in filtered.columns:
                x = filtered['End X'].dropna() * 1.2
                y = filtered['End Y'].dropna() * 0.8
                all_x.extend(x)
                all_y.extend(y)
            elif 'X' in filtered.columns and 'Y' in filtered.columns:
                # Fallback to start coordinates if end coordinates not available
                x = filtered['X'].dropna() * 1.2
                y = filtered['Y'].dropna() * 0.8
                all_x.extend(x)
                all_y.extend(y)
        elif event_name == "Box Entryt (kuljettaen)":
            # Count dribble box entries by zone
            if 'Secondary' in filtered.columns:
                for zone in filtered['Secondary'].dropna():
                    zone_lower = zone.lower()
                    if zone_lower in dribble_zones:
                        dribble_zones[zone_lower] += 1
        else:
            # Use regular coordinates for other events
            if 'X' in filtered.columns and 'Y' in filtered.columns:
                x = filtered['X'].dropna() * 1.2
                y = filtered['Y'].dropna() * 0.8

                # Only mirror VAS shots, not other events since data is already sorted
                if team == "VAS" and event_name == "Laukaukset":
                    x = 120 - x
                    y = 80 - y
                all_x.extend(x)
                all_y.extend(y)

    # Handle dribble box entries with zonal rectangles
    if event_name == "Box Entryt (kuljettaen)":
        import matplotlib.patches as patches
        
        # Define exact zone boundaries with rectangles positioned around the box
        # Outer edges face center line (x=60) and sidelines (y=0, y=80)
        if team == "JJK":
            # JJK zones - rectangles extend outward from penalty area
            zone_coords = {
                'right flank': {'x': 102, 'y': 62, 'width': 18, 'height': 6},   # Top horizontal - extends to y=80 (right sideline)
                'right': {'x': 96, 'y': 48, 'width': 6, 'height': 14},         # Right vertical - extends toward center  
                'center': {'x': 96, 'y': 32, 'width': 6, 'height': 16},       # Center (102,32) to (120,48) - extends toward x=60
                'left': {'x': 96, 'y': 18, 'width': 6, 'height': 14},          # Left vertical - extends toward center
                'left flank': {'x': 102, 'y': 12, 'width': 18, 'height': 6}     # Bottom horizontal - extends to y=0 (left sideline)
            }
        else:
            # OPP zones - inverted coordinates, rectangles extend outward from penalty area  
            zone_coords = {
                'right flank': {'x': 0, 'y': 12, 'width': 18, 'height': 6},     # Bottom horizontal - extends to y=0 (left sideline)
                'right': {'x': 18, 'y': 18, 'width': 6, 'height': 14},          # Right vertical - extends toward center
                'center': {'x': 18, 'y': 32, 'width': 6, 'height': 16},         # Center mirrored - extends toward x=60
                'left': {'x': 18, 'y': 48, 'width': 6, 'height': 14},           # Left vertical - extends toward center  
                'left flank': {'x': 0, 'y': 62, 'width': 18, 'height': 6}       # Top horizontal - extends to y=80 (right sideline)
            }
        
        # Zone order remains consistent
        zone_order = ['left flank', 'left', 'center', 'right', 'right flank']
        
        # Get max count for color scaling
        max_count = max(dribble_zones.values()) if any(dribble_zones.values()) else 1
        
        # Draw rectangles for each zone
        for zone in zone_order:
            count = dribble_zones[zone]
            if max_count > 0:
                intensity = count / max_count
            else:
                intensity = 0
            
            # Color based on team and intensity
            base_color = '#FFD100' if team == "VAS" else '#A6192E'
            alpha = 0.3 + (intensity * 0.7)  # Alpha between 0.3 and 1.0
            
            # Get zone coordinates - now all zones have proper width and height
            coords = zone_coords[zone]
            rect_x = coords['x']
            rect_y = coords['y']
            rect_width = coords['width']
            rect_height = coords['height']
            
            # Create rectangle
            rect = patches.Rectangle(
                (rect_x, rect_y), rect_width, rect_height,
                linewidth=2, edgecolor='white', facecolor=base_color, alpha=alpha
            )
            ax.add_patch(rect)
            
            # Add count text in center of rectangle
            text_x = rect_x + rect_width/2
            text_y = rect_y + rect_height/2
            ax.text(text_x, text_y, str(count), 
                   ha='center', va='center', fontsize=12, 
                   color='white', fontweight='bold')
        
        # Add legend/title explaining zones
        zone_display_names = {
            'left flank': 'Vasen laita',
            'left': 'Vasen', 
            'center': 'Keskusta',
            'right': 'Oikea',
            'right flank': 'Oikea laita'
        }
        zone_labels = ' | '.join([f"{zone_display_names[zone]}: {dribble_zones[zone]}" for zone in zone_order])
        ax.text(60, 75, f"Kuljetukset boksiin: {zone_labels}", 
               ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Create heatmap for non-dribble events
    elif all_x and all_y:
        pitch.kdeplot(
            x=all_x,
            y=all_y,
            ax=ax,
            cmap=cmr.amber if team == "VAS" else cmr.sunburst,
            fill=True,
            levels=125,
            shade_lowest=False,
            alpha=0.85,
            bw_adjust=0.3,
        )
    else:
        # Show "No Data" message for events with no coordinate data
        # This covers Behind Start/End with missing coordinates and other events with no data
        ax.text(60, 40, "Ei dataa", ha='center', va='center', fontsize=20, color='white', fontweight='bold')

    fig.patch.set_facecolor('#122c3d')
    return fig

# === NEW FUNCTIONS FOR KEHITYS TAB ===

def extract_aggregate_stats(uploaded_dfs, xgb_model, expected_cols, selected_half="Täysi ottelu"):
    """Calculate average stats across all uploaded matches"""
    all_jjk_stats = []
    all_opp_stats = []
    
    for df in uploaded_dfs:
        # Filter by half
        if selected_half == "1. puoliaika":
            df = df[df['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df = df[df['Half'] == '2. puoliaika']
        
        # Extract stats for this match
        row_labels, jjk_stats, opp_stats = extract_full_match_stats(df, xgb_model, expected_cols, "Average")
        
        # Convert string stats to numeric where possible for averaging
        numeric_jjk = []
        numeric_opp = []
        
        for jjk_val, opp_val in zip(jjk_stats, opp_stats):
            # Convert to numeric, handling special cases
            def convert_to_numeric(val):
                if isinstance(val, str):
                    if 'N/A' in val or val == 'N/A':
                        return np.nan
                    elif '%' in val:
                        return float(val.replace('%', ''))
                    elif ':' in val:  # Time format
                        parts = val.split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    else:
                        try:
                            return float(val)
                        except:
                            return np.nan
                return float(val)
            
            numeric_jjk.append(convert_to_numeric(jjk_val))
            numeric_opp.append(convert_to_numeric(opp_val))
        
        all_jjk_stats.append(numeric_jjk)
        all_opp_stats.append(numeric_opp)
    
    # Calculate averages
    avg_jjk = np.nanmean(all_jjk_stats, axis=0)
    avg_opp = np.nanmean(all_opp_stats, axis=0)
    
    # Format back to display format
    def format_avg_stat(val, idx):
        if np.isnan(val):
            return "N/A"
        elif idx in [4, 9, 13]:  # Percentage fields (Conversion %, AT %, Passes Behind Success %)
            return f"{val:.1f}%"
        elif idx == 8:  # Attacking time - convert back to mm:ss
            return format_seconds(val)
        elif idx == 10:  # AT/BE ratio
            return f"{val:.1f}"
        elif idx in [1, 3]:  # xG and xG/Shot - use 2 decimal places
            return f"{val:.2f}"
        else:
            return f"{int(val)}" if val == int(val) else f"{val:.1f}"
    
    formatted_jjk = [format_avg_stat(val, idx) for idx, val in enumerate(avg_jjk)]
    formatted_opp = [format_avg_stat(val, idx) for idx, val in enumerate(avg_opp)]
    
    return row_labels, formatted_jjk, formatted_opp

def generate_average_radar_chart(uploaded_dfs, xgb_model, expected_cols, selected_half="Täysi ottelu"):
    """Generate radar chart showing average performance across all matches"""
    all_jjk_kpis = []
    all_opp_kpis = []
    
    for df in uploaded_dfs:
        # Filter by half
        if selected_half == "1. puoliaika":
            df = df[df['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df = df[df['Half'] == '2. puoliaika']
        
        jjk_vals, opp_vals = extract_radar_kpis_smart(df, xgb_model, expected_cols)
        all_jjk_kpis.append(jjk_vals)
        all_opp_kpis.append(opp_vals)
    
    # Calculate averages
    avg_jjk = np.nanmean(all_jjk_kpis, axis=0)
    avg_opp = np.nanmean(all_opp_kpis, axis=0)
    
    # Use the existing radar chart function with average values
    params = [
        'Laukaukset', 'xG', 'xG/Laukaus', 'Box Entryt', 'HA%', 'HA/BE (s)',
        'Korkeat riistot', 'Onn. syötöt taakse', 'Onn. syötöt taakse %'
    ]
    
    # Set min/max for normalization
    low = [0, 0, 0.07, 0, 0, 20, 0, 0, 0]
    high = [
        max(avg_jjk[0], avg_opp[0]) + 5,
        max(avg_jjk[1], avg_opp[1]) + 0.5,
        max(avg_jjk[2], avg_opp[2]) + 0.03,
        max(avg_jjk[3], avg_opp[3]) + 5,
        max(avg_jjk[4], avg_opp[4]) + 5,
        max(avg_jjk[5], avg_opp[5]) + 5,
        max(avg_jjk[6], avg_opp[6]) + 5,
        max(avg_jjk[7], avg_opp[7]) + 5,
        max(avg_jjk[8], avg_opp[8]) + 5
    ]
    
    lower_is_better = ['HA/BE (s)']
    radar = Radar(params, low, high, lower_is_better=lower_is_better,
                  round_int = [True, False, False, True, False, False, True, True, False],
                  num_rings=5, ring_width=1,
                  center_circle_radius=1)
    
    fig, ax = radar.setup_axis(figsize=(9,9))
    radar.draw_circles(ax=ax, facecolor="#28252c", edgecolor="#39353f", lw=1.5)
    
    # Draw average performance for both teams
    radar.draw_radar_solid(avg_jjk, ax=ax, kwargs={'facecolor': 'none', 'alpha': 1.0, 'edgecolor': '#A6192E', 'lw': 3})
    radar.draw_radar_solid(avg_opp, ax=ax, kwargs={'facecolor': 'none', 'alpha': 1.0, 'edgecolor': '#FFD100', 'lw': 3})
    
    radar.draw_range_labels(ax=ax, fontsize=15, color='#fcfcfc')
    radar.draw_param_labels(ax=ax, fontsize=15, color='#fcfcfc')
    
    fig.patch.set_facecolor('#122c3d')
    ax.set_facecolor('#122c3d')
    plt.tight_layout()
    return fig

def parse_date_from_filename(filename):
    """Extract date from filename format 'team_name dd.mm.csv'"""
    import re
    from datetime import datetime
    
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Look for date pattern dd.mm at the end
    date_pattern = r'(\d{1,2})\.(\d{1,2})$'
    match = re.search(date_pattern, name)
    
    if match:
        day, month = match.groups()
        # Assume current year if not specified
        year = 2024  # You can adjust this or make it dynamic
        try:
            return datetime(year, int(month), int(day))
        except ValueError:
            return None
    return None

def plot_zscore_development(uploaded_files, xgb_model, expected_cols, selected_half="Täysi ottelu"):
    """Create line chart showing z-score development over time for JJK vs opponents across different KPIs"""
    # Parse dates and sort files chronologically
    file_data = []
    
    for uploaded_file in uploaded_files:
        date = parse_date_from_filename(uploaded_file.name)
        # Extract opponent name from filename (everything before the date)
        filename_base = uploaded_file.name.replace('.csv', '')
        # Remove date pattern to get opponent name
        import re
        date_pattern = r'\s+\d{1,2}\.\d{1,2}$'
        opponent_name = re.sub(date_pattern, '', filename_base).strip()
        
        if date:
            file_data.append((date, uploaded_file, opponent_name))
    
    # Sort by date
    file_data.sort(key=lambda x: x[0])
    
    if len(file_data) < 2:
        # Not enough data for development chart
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        fig.patch.set_facecolor('#122c3d')
        ax.set_facecolor('#122c3d')
        ax.text(0.5, 0.5, 'Vähintään 2 ottelua\nvaaditaan kehitysanalyysiin', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=14, color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig
    
    # Calculate KPIs for each match
    dates = []
    opponents = []
    jjk_kpis_over_time = []
    opp_kpis_over_time = []
    
    for date, uploaded_file, opponent_name in file_data:
        df, _ = load_data_from_upload_analysis(uploaded_file)
        
        # Filter by half
        if selected_half == "1. puoliaika":
            df = df[df['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df = df[df['Half'] == '2. puoliaika']
        
        # Count shots in the raw data for debugging
        shot_count = len(df[df['Name'] == 'Shot'])
        
        jjk_vals, opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
        dates.append(date)
        opponents.append(opponent_name)
        jjk_kpis_over_time.append(jjk_vals)
        opp_kpis_over_time.append(opp_vals)
    
    kpi_list_jjk = []
    kpi_list_opp = []
    # Use the same sorted file_data to maintain consistency
    for date, uploaded_file, opponent_name in file_data:
        df_temp, _ = load_data_from_upload_analysis(uploaded_file)
        # Filter by half (same logic as get_avg_kpis_from_uploads)
        if selected_half == "1. puoliaika":
            df_temp = df_temp[df_temp['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df_temp = df_temp[df_temp['Half'] == '2. puoliaika']
        jjk_vals_temp, opp_vals_temp = extract_radar_kpis(df_temp, xgb_model, expected_cols)
        kpi_list_jjk.append(jjk_vals_temp)
        kpi_list_opp.append(opp_vals_temp)
    
    # Step 2: Calculate baseline statistics (same as tab1)
    jjk_arr = np.array(kpi_list_jjk)
    opp_arr = np.array(kpi_list_opp)
    avg_jjk = np.nanmean(kpi_list_jjk, axis=0) if kpi_list_jjk else None
    avg_opp = np.nanmean(kpi_list_opp, axis=0) if kpi_list_opp else None
    std_jjk = np.nanstd(jjk_arr, axis=0) if len(jjk_arr) > 0 else np.ones(len(kpi_list_jjk[0]))
    std_opp = np.nanstd(opp_arr, axis=0) if len(opp_arr) > 0 else np.ones(len(kpi_list_opp[0]))
    
    # Fix for flatline issue: Use minimum threshold for standard deviation
    min_std_threshold = 0.1  # Minimum std dev to prevent flatlines
    std_jjk = np.maximum(std_jjk, min_std_threshold)
    std_opp = np.maximum(std_opp, min_std_threshold)
    
    # Step 3: Calculate z-scores for each match using the EXACT same logic as display_jjk_kpi_stats_vertical
    jjk_zscores_over_time = []
    opp_zscores_over_time = []
    
    # Process each match in the same order as file_data for consistency
    for i, ((date, uploaded_file, opponent_name), jjk_vals, opp_vals) in enumerate(zip(file_data, kpi_list_jjk, kpi_list_opp)):
        # Convert all inputs to float arrays (same as display_jjk_kpi_stats_vertical)
        jjk_vals = np.array(jjk_vals, dtype=float)
        opp_vals = np.array(opp_vals, dtype=float)
        avg_jjk_arr = np.array(avg_jjk, dtype=float)
        avg_opp_arr = np.array(avg_opp, dtype=float)
        std_jjk_arr = np.array(std_jjk, dtype=float)
        std_opp_arr = np.array(std_opp, dtype=float)
        
        # Avoid division by zero for std (same as display_jjk_kpi_stats_vertical)
        std_jjk_arr = np.where(std_jjk_arr == 0, 1e-8, std_jjk_arr)
        std_opp_arr = np.where(std_opp_arr == 0, 1e-8, std_opp_arr)
        
        # Calculate z-scores for both teams (identical formula to display_jjk_kpi_stats_vertical)
        z_jjk = (jjk_vals - avg_jjk_arr) / std_jjk_arr
        z_opp = (opp_vals - avg_opp_arr) / std_opp_arr
        
        # Load the correct file for debug analysis
        df_debug, _ = load_data_from_upload_analysis(uploaded_file)
        if selected_half == "1. puoliaika":
            df_debug = df_debug[df_debug['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df_debug = df_debug[df_debug['Half'] == '2. puoliaika']
        
        # Count shots in the correct current data for debugging
        current_shot_count = len(df_debug[df_debug['Name'] == 'Shot'])
        
        # DEBUG: Detailed shot breakdown by type
        all_shots = df_debug[df_debug['Name'] == 'Shot']
        regular_play_shots = len(all_shots[all_shots['Secondary'].str.contains('Regular Play', na=False)])
        penalty_shots = len(all_shots[all_shots['Secondary'].str.contains('Penalty', na=False)])
        freekick_shots = len(all_shots[all_shots['Secondary'].str.contains('From Free Kick', na=False)])
        other_shots = current_shot_count - regular_play_shots - penalty_shots - freekick_shots
        
        # Process shots through the same function to see what gets filtered
        processed_shots = process_shots(all_shots, expected_cols, xgb_model) if len(all_shots) > 0 else []
        processed_shot_count = len(processed_shots)
        
        # DEBUG: Check Box Entry events specifically
        box_entry_jjk = len(df_debug[df_debug['Name'] == 'Box Entry'])
        box_entry_opp = len(df_debug[df_debug['Name'] == 'Box Entry OPP'])
        attacking_time_jjk = df_debug[df_debug['Name'] == 'Attacking Time']['Duration'].sum()
        attacking_time_opp = df_debug[df_debug['Name'] == 'Attacking Time OPP']['Duration'].sum()
        
        # DEBUG: Collect detailed calculation info for shots (index 0) and Box Entry (index 3) for each match
        match_debug = f"""Match {i+1}: {date} vs {opponent_name}
File: {uploaded_file.name}
Total Shot Events: {current_shot_count}
- Regular Play: {regular_play_shots}
- Penalties: {penalty_shots} 
- Free Kicks: {freekick_shots}
- Other: {other_shots}
Processed Shots (after filtering): {processed_shot_count}
KPI Shots Value: {jjk_vals[0]:.3f}
Z-score: {z_jjk[0]:.3f} (Avg: {avg_jjk_arr[0]:.3f}, Std: {std_jjk_arr[0]:.3f})

Box Entry Debug:
- JJK Box Entries: {box_entry_jjk}, KPI Value: {jjk_vals[3]:.3f}
- OPP Box Entries: {box_entry_opp}, KPI Value: {opp_vals[3]:.3f}
- JJK Attacking Time: {attacking_time_jjk}s, OPP Attacking Time: {attacking_time_opp}s
- JJK HA/BE: {jjk_vals[5]:.3f}, OPP HA/BE: {opp_vals[5]:.3f}
- OPP Box Entry Z-score: {z_opp[3]:.3f} (Avg: {avg_opp_arr[3]:.3f}, Std: {std_opp_arr[3]:.3f})
- OPP HA/BE Z-score: {z_opp[5]:.3f} (Avg: {avg_opp_arr[5]:.3f}, Std: {std_opp_arr[5]:.3f})"""
        
        if i == 0:
            debug_text_dev = f"🔍 DEBUG: Development Chart Z-Score Calculations\n\n{match_debug}"
        else:
            debug_text_dev += f"\n\n{match_debug}"
        
        jjk_zscores_over_time.append(z_jjk)
        opp_zscores_over_time.append(z_opp)

    # KPI names that match the radar chart parameters
    kpi_names = [
        'Laukaukset', 'xG', 'xG/Laukaus', 'Box Entryt', 'HA%', 'HA/BE (s)',
        'Korkeat riistot', 'Onn. syötöt taakse', 'Onn. syötöt taakse %'
    ]
    
    # Create subplots - one for each KPI in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), dpi=100)
    fig.patch.set_facecolor('#122c3d')
    axes = axes.flatten()
    
    # Plot each KPI in its own subplot
    for kpi_idx in range(len(kpi_names)):
        ax = axes[kpi_idx]
        ax.set_facecolor('#122c3d')
        
        # Extract z-scores for this specific KPI across all matches
        jjk_kpi_zscores = [zscores[kpi_idx] for zscores in jjk_zscores_over_time]
        opp_kpi_zscores = [zscores[kpi_idx] for zscores in opp_zscores_over_time]
        
        # Use match indices for x-axis to avoid date overlap issues
        match_indices = range(len(dates))
        
        # Plot JJK and opponent z-scores for this KPI
        ax.plot(match_indices, jjk_kpi_zscores, 'o-', color='#A6192E', linewidth=2, 
                markersize=5, markeredgecolor='black', markeredgewidth=0.5)
        ax.plot(match_indices, opp_kpi_zscores, 'o-', color='#FFD100', linewidth=2, 
                markersize=5, markeredgecolor='black', markeredgewidth=0.5)
        
        # Add reference lines
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.8, linewidth=1.5)
        ax.axhline(y=1, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=-1, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=2, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=-2, color='white', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting for each subplot
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlim(-0.5, len(dates)-0.5)
        ax.tick_params(axis='both', colors='white', labelsize=8)
        ax.grid(True, axis='y', alpha=0.2, color='white')  # Only y-axis gridlines
        ax.set_title(kpi_names[kpi_idx], color='white', fontsize=10, fontweight='bold', pad=10)
        
        # Set x-axis ticks and labels
        ax.set_xticks(match_indices)
        x_labels = opponents  # Use only opponent names
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Suhteellinen kehitys (Z-arvot)', color='white', fontsize=16, fontweight='bold')
    
    # DEBUG: Add summary statistics to debug text
    debug_text_dev += f"\n\n📊 Summary:\nTotal matches processed: {len(jjk_kpis_over_time)}\nBaseline statistics calculated from {len(kpi_list_jjk)} matches"
    debug_text_dev += f"\nSelected half filter: {selected_half}"
    if len(jjk_kpis_over_time) > 0:
        shots_zscores = [zscores[0] for zscores in jjk_zscores_over_time]
        debug_text_dev += f"\nShots z-scores: {[round(z, 3) for z in shots_zscores]}\nMin: {min(shots_zscores):.3f}, Max: {max(shots_zscores):.3f}"
        
    # CRITICAL DEBUG: Check for potential data discrepancy
    debug_text_dev += f"\n\n⚠️  DATA DISCREPANCY CHECK:\nIf you see different shot counts between match report and development chart for the same match,\nthis indicates the functions are processing different data sources or applying different filters."
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    return fig, debug_text_dev

def plot_all_shots_pitch(uploaded_dfs, xgb_model, expected_cols, selected_half="Täysi ottelu"):
    """Create vertical pitch showing all shots from all matches"""
    pitch = VerticalPitch(pitch_color='#122c3d', line_color='white')
    fig, ax = pitch.draw(figsize=(4.8, 15))
    fig.patch.set_facecolor('#122c3d')

    all_jjk_shots = []
    all_opp_shots = []

    for df in uploaded_dfs:
        # Filter by half
        if selected_half == "1. puoliaika":
            df = df[df['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df = df[df['Half'] == '2. puoliaika']

        # JJK Shots
        shots = process_shots(df[df['Name'] == 'Shot'], expected_cols, xgb_model)
        if not shots.empty:
            shots['plot_x'] = shots['location_x']
            shots['plot_y'] = shots['location_y']
            shots_no_pen = shots[~shots['Secondary'].str.contains("Penalty", na=False)]
            all_jjk_shots.append(shots_no_pen)

        # Opponent Shots
        shots_opp = process_shots(df[df['Name'] == 'Shot OPP'], expected_cols, xgb_model)
        if not shots_opp.empty:
            shots_opp['plot_x'] = 120 - shots_opp['location_x']
            shots_opp['plot_y'] = 80 - shots_opp['location_y']
            shots_opp_no_pen = shots_opp[~shots_opp['Secondary'].str.contains("Penalty", na=False)]
            all_opp_shots.append(shots_opp_no_pen)

    # Combine all shots
    if all_jjk_shots:
        combined_jjk = pd.concat(all_jjk_shots, ignore_index=True)
        # Separate goals and non-goals for different colors
        goals_jjk = combined_jjk[combined_jjk['Primary'] == 'Goal']
        non_goals_jjk = combined_jjk[combined_jjk['Primary'] != 'Goal']
        
        # Plot non-goals in team color
        if not non_goals_jjk.empty:
            pitch.scatter(
                non_goals_jjk['plot_x'], non_goals_jjk['plot_y'],
                s=non_goals_jjk['xG'] * 300, color='#A6192E', edgecolors='black', alpha=0.7, ax=ax
            )
        
        # Plot goals in green
        if not goals_jjk.empty:
            pitch.scatter(
                goals_jjk['plot_x'], goals_jjk['plot_y'],
                s=goals_jjk['xG'] * 300, color='green', edgecolors='black', alpha=0.7, ax=ax
            )

    if all_opp_shots:
        combined_opp = pd.concat(all_opp_shots, ignore_index=True)
        # Separate goals and non-goals for different colors
        goals_opp = combined_opp[combined_opp['Primary'] == 'Goal']
        non_goals_opp = combined_opp[combined_opp['Primary'] != 'Goal']
        
        # Plot non-goals in team color
        if not non_goals_opp.empty:
            pitch.scatter(
                non_goals_opp['plot_x'], non_goals_opp['plot_y'],
                s=non_goals_opp['xG'] * 300, color='#FFD100', edgecolors='black', alpha=0.7, ax=ax
            )
        
        # Plot goals in green
        if not goals_opp.empty:
            pitch.scatter(
                goals_opp['plot_x'], goals_opp['plot_y'],
                s=goals_opp['xG'] * 300, color='green', edgecolors='black', alpha=0.7, ax=ax
            )

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02)
    
    return fig
