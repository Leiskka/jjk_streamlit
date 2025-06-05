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
)

# === LOAD MATCH DATA ===
def load_data(file_path):
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    team_name = os.path.splitext(filename)[0]
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
    return df, team_name

def get_avg_kpis_from_uploads(uploaded_files, xgb_model, expected_cols):
    kpi_list_jjk = []
    kpi_list_opp = []
    for f in uploaded_files:
        df, _ = load_data_from_upload(f)
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

        st.title(f"Match Dashboard: JJK vs {team_name}")

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

        tab1, tab2, tab3 = st.tabs(["Summary", "Timeline", "Pitch Events"])

        with tab1:
            st.subheader("Match Stats & Radar Chart")
            col1, col2 = st.columns(2)
        with col1:
            row_labels, jjk_stats, opp_stats = extract_full_match_stats(df, xgb_model, expected_cols, team_name)
            st.pyplot(generate_kpi_table(row_labels, jjk_stats, opp_stats, team_name), use_container_width=True)
        with col2:
            jjk_vals, opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
            avg_jjk, avg_opp = get_avg_kpis_from_uploads(uploaded_files, xgb_model, expected_cols)

            # Check for valid radar chart data
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

        with tab2:
            st.subheader("Cumulative xG")
            st.pyplot(plot_cumulative_xg(df, xgb_model, expected_cols, team_name))

            st.subheader("Momentum Chart")
            st.pyplot(plot_momentum_chart(df, team_name))

        with tab3:
            st.subheader("Tactical Pitch View")
            event_types = ["Shots", "Fouls", "Interceptions", "Defensive Duels", "Box Entry Passes"]
            selected_events = st.multiselect("Select Events to Overlay", event_types, default=event_types)
            st.pyplot(plot_combined_pitch(df, xgb_model, expected_cols, selected_events, team_name), use_container_width=True)

