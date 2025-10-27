import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Radar
import joblib
import os
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
from matplotlib.backends.backend_pdf import PdfPages
import xml.etree.ElementTree as ET
import tempfile

# === CONFIG ===
st.set_page_config(page_title="JJK Match Dashboard", layout="wide", page_icon="⚽")

# === SIDEBAR ===
st.sidebar.title("Match Selection")

uploaded_files = st.sidebar.file_uploader(
    "Upload Match CSV Files", 
    type="csv", 
    accept_multiple_files=True
)

match_files = [f.name for f in uploaded_files] if uploaded_files else []
match_file_name = st.sidebar.selectbox("Select Match", match_files)

# Add half selection dropdown under match selection
half_options = ["Full Match", "1st Half", "2nd Half"]
selected_half = st.sidebar.selectbox("Select Half", half_options)

# === LOAD MODELS ===
xgb_model = joblib.load("xgboost_xg_model.pkl")
expected_cols = joblib.load("xgboost_expected_cols.pkl")

# === IMPORT ANALYSIS FUNCTIONS ===
from analysis_module import (
    process_shots,
    get_avg_kpis,
    format_seconds,
    parse_time,
    generate_kpi_table,
    generate_radar_chart,
    plot_cumulative_xg,
    plot_momentum_chart,
    plot_combined_pitch,
    extract_full_match_stats,
    extract_radar_kpis,
    plot_kpi_bars,
    plot_event_heatmap
)

# === LOAD MATCH DATA ===
def load_data(file_path):
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    team_name = os.path.splitext(filename)[0]
    team_name = team_name.split(" ")[0]
    return df, team_name

# --- Build radar chart values in the same way as kpit.py ---
def safe_div(a, b):
    try:
        return float(a) / float(b) if float(b) > 0 else 0
    except:
        return 0

# === LOAD MATCH DATA FROM UPLOAD ===
def load_data_from_upload(uploaded_file):
    uploaded_file.seek(0)  # Ensure pointer is at the start
    df = pd.read_csv(uploaded_file)
    filename = uploaded_file.name
    team_name = os.path.splitext(filename)[0]
    team_name = team_name.split(" ")[0]
    return df, team_name

def get_avg_kpis_from_uploads(uploaded_files, xgb_model, expected_cols, selected_half):
    kpi_list_jjk = []
    kpi_list_opp = []
    for f in uploaded_files:
        df, _ = load_data_from_upload(f)
        # Filter by half if needed
        if selected_half == "1st Half":
            df = df[df['Half'] == '1st Half']
        elif selected_half == "2nd Half":
            df = df[df['Half'] == '2nd Half']
        jjk_vals, opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
        kpi_list_jjk.append(jjk_vals)
        kpi_list_opp.append(opp_vals)
    avg_jjk = np.nanmean(kpi_list_jjk, axis=0) if kpi_list_jjk else None
    avg_opp = np.nanmean(kpi_list_opp, axis=0) if kpi_list_opp else None
    return avg_jjk, avg_opp

# === MAIN ===
if match_file_name and uploaded_files:
    # Find the selected file object
    selected_file = next((f for f in uploaded_files if f.name == match_file_name), None)
    if selected_file:
        df, team_name = load_data_from_upload(selected_file)

        # Filter df by half selection
        if selected_half == "1st Half":
            df = df[df['Half'] == '1st Half']
        elif selected_half == "2nd Half":
            df = df[df['Half'] == '2nd Half']

        st.title(f"Match Dashboard: JJK vs {team_name} ({selected_half})")

        # --- Inject CSS for larger tab font and padding ---
        st.markdown("""
            <style>
            /* Make tab labels bigger and bolder */
            .stTabs [data-baseweb="tab"] {
                font-size: 20.5rem !important;
                font-weight: 1000 !important;
                padding: 1.2rem 2.5rem !important;
            }
            </style>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab_heatmap, tab_dev, tab_export = st.tabs(["Summary", "Timeline", "Pitch Events", "Heatmap", "Development", "Export"])

        with tab1:
            st.subheader("Match Stats")
            col1, col2 = st.columns(2)
            with col1:
                row_labels, jjk_stats, opp_stats = extract_full_match_stats(df, xgb_model, expected_cols, team_name)
                st.pyplot(generate_kpi_table(row_labels, jjk_stats, opp_stats, team_name), use_container_width=True)
            with col2:
                jjk_vals, opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
                avg_jjk, avg_opp = get_avg_kpis_from_uploads(uploaded_files, xgb_model, expected_cols, selected_half)

                # Radar chart
                if (
                    avg_jjk is not None and avg_opp is not None and
                    not np.any(np.isnan(jjk_vals)) and
                    not np.any(np.isnan(opp_vals)) and
                    not np.any(np.isnan(avg_jjk)) and
                    not np.any(np.isnan(avg_opp)) and
                    np.all(np.array([
                        max(jjk_vals[i], opp_vals[i], avg_jjk[i], avg_opp[i]) >
                        min(jjk_vals[i], opp_vals[i], avg_jjk[i], avg_opp[i])
                        for i in range(len(jjk_vals))
                    ]))
                ):
                    st.pyplot(generate_radar_chart(
                        jjk_vals, opp_vals, avg_jjk, avg_opp, team_name
                    ))
                else:
                    st.info("Not enough valid data to display radar chart.")

            # Place the KPI bar chart below the columns, spanning full width
            if (
                avg_jjk is not None and avg_opp is not None and
                not np.any(np.isnan(jjk_vals)) and
                not np.any(np.isnan(opp_vals)) and
                not np.any(np.isnan(avg_jjk)) and
                not np.any(np.isnan(avg_opp))
            ):
                st.pyplot(plot_kpi_bars(jjk_vals, opp_vals, avg_jjk, avg_opp, team_name), use_container_width=True)
        with tab2:
            st.subheader("Cumulative xG")
            st.pyplot(plot_cumulative_xg(df, xgb_model, expected_cols, team_name, selected_half))

            st.subheader("Momentum Chart")
            st.pyplot(plot_momentum_chart(df, team_name, selected_half))  # <-- pass selected_half

        with tab3:
            st.subheader("Tactical Pitch View")
            event_types = ["Shots", "Fouls", "Interceptions", "Defensive Duels", "Box Entry Passes"]
            selected_events = st.multiselect("Select Events to Overlay", event_types, default=event_types)
            st.pyplot(plot_combined_pitch(df, xgb_model, expected_cols, selected_events, team_name), use_container_width=True)

        with tab_heatmap:
            st.subheader("Event Heatmap (All Matches)")
            st.write("Select an event to visualize its heatmap across all uploaded matches.")
            heatmap_options = [
                "Shots",
                "Box Entries (Pass)",
                "Defensive Duels",
                "Interceptions",
                "Fouls"
            ]
            selected_heatmap = st.selectbox("Select Event for Heatmap", heatmap_options)
            
            selected_team = st.radio("Select Team", ["JJK", "OPP"], horizontal=True, key="heatmap_team")
            
            if uploaded_files:
                dfs = [load_data_from_upload(f)[0] for f in uploaded_files]
                # Pass selected_team to the function!
                fig = plot_event_heatmap(dfs, selected_heatmap, selected_half, selected_team)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info(f"Heatmap for {selected_heatmap} will be shown here.")

        with tab_dev:
            st.subheader("KPI Development")

            kpi_options = [
                'Shots', 'xG', 'xG/Shot', 'AT%', 'AT/BE (s)',
                'High Recoveries', 'Succ. Passes Behind', 'Box Entries'
            ]
            selected_kpi = st.selectbox("Select KPI", kpi_options)

            selected_team_dev = st.radio("Select Team", ["JJK", "OPP"], horizontal=True, key="dev_team")

            if uploaded_files:
                from analysis_module import get_kpi_development
                labels, values, title = get_kpi_development(uploaded_files, xgb_model, expected_cols, selected_kpi, selected_team_dev, selected_half)

                if labels and values:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.set_facecolor('#122c3d')
                    fig.patch.set_facecolor('#122c3d')
                    ax.plot(labels, values, marker='o', color='#A6192E' if selected_team_dev == "JJK" else '#FFD100', linewidth=3)
                    ax.set_title(f'{selected_team_dev} - {title}', color='white')
                    ax.set_ylabel(title, color='white')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("No valid KPI data found in uploaded files.")
            else:
                st.info("Please upload match files to view KPI trends.")

        with tab_export:
            # --- CSV Export ---
            st.subheader("Export All Uploaded Matches' Stats as CSV")
            if uploaded_files:
                if st.button("Generate CSV"):
                    import csv
                    import io
                    import re

                    # Extract date at the end of the filename (formats: d.m, dd.mm, d.mm, dd.m)
                    def extract_date(filename):
                        # Find the last occurrence of a date pattern at the end of the filename (before extension)
                        match = re.search(r'(\d{1,2})\.(\d{1,2})(?=\D*$)', filename)
                        if match:
                            day, month = match.groups()
                            return int(month), int(day)
                        return (0, 0)

                    sorted_files = sorted(uploaded_files, key=lambda f: extract_date(f.name))

                    output = io.StringIO()
                    writer = csv.writer(output)
                    # Get KPI labels from the first file
                    first_df, first_team_name = load_data_from_upload(sorted_files[0])
                    row_labels, _, _ = extract_full_match_stats(first_df, xgb_model, expected_cols, first_team_name)
                    # Interleaved header: [kpi1 JJK, kpi1 OPP, ...]
                    kpi_header = []
                    for label in row_labels:
                        kpi_header.append(f"{label} JJK")
                        kpi_header.append(f"{label} OPP")
                    header = ["Match"] + kpi_header  # Removed "File"
                    writer.writerow(header)
                    for f in sorted_files:
                        df, team_name = load_data_from_upload(f)
                        row_labels, jjk_stats, opp_stats = extract_full_match_stats(df, xgb_model, expected_cols, team_name)
                        # Interleave jjk_stats and opp_stats
                        kpi_row = []
                        for jjk, opp in zip(jjk_stats, opp_stats):
                            kpi_row.append(jjk)
                            kpi_row.append(opp)
                        writer.writerow([team_name] + kpi_row)  # Removed f.name
                    csv_bytes = output.getvalue().encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_bytes,
                        file_name="all_matches_stats.csv",
                        mime="text/csv"
                    )
