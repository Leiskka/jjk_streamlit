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
from PIL import Image
import re
from datetime import datetime

st.set_page_config(page_title="JJK Data 📊", layout="wide", initial_sidebar_state="expanded")

# File upload section
st.sidebar.markdown("### 📁 Lataa ottelutiedostot")
uploaded_files = st.sidebar.file_uploader(
    "Lataa CSV-tiedostot",
    type=['csv'],
    accept_multiple_files=True,
)

# === LOAD MODELS ===
try:
    xgb_model = joblib.load("xgboost_xg_model.pkl")
    expected_cols = joblib.load("xgboost_expected_cols.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.error("Please ensure 'xgboost_xg_model.pkl' and 'xgboost_expected_cols.pkl' are in the app directory")
    st.stop()

# === IMPORT ANALYSIS FUNCTIONS ===
from analysis_module import (
    generate_kpi_table,
    generate_radar_chart,
    plot_cumulative_xg,
    plot_combined_pitch_vertical,
    extract_full_match_stats,
    extract_radar_kpis,
    plot_event_heatmap,
    display_jjk_kpi_stats_vertical,
    extract_aggregate_stats,
    generate_average_radar_chart,
    plot_zscore_development,
    plot_all_shots_pitch
)

class MockFile(io.BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name

# === TRANSLATION FUNCTION ===
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
        "Box Entry VAS": "Box Entry OPP",
        "Murtautuminen boksiin VAS": "Box Entry OPP",
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
        "Keskeltä": "Center",
        # Additional coordinate and box entry related translations
        "Kuljettaen": "Dribbling",
        "Kuljettaen boksiin": "Dribble into Box",
        "Syöttö boksiin": "Pass into Box",
        "Syötön syötön lähtöpiste": "Pass Start Point",
        "Syötön syötön loppupiste": "Pass End Point",
        "syötön lähtöpiste": "Start Point",
        "syötön loppupiste": "End Point",
        "Koordinaatti": "Coordinate",
        "X-koordinaatti": "X Coordinate",
        "Y-koordinaatti": "Y Coordinate",
        "Alkupiste": "Start Point",
        "Päätepiste": "End Point",
        "X2": "X2",
        "Y2": "Y2",
        "EndX": "EndX",
        "EndY": "EndY",
        "StartX": "StartX",
        "StartY": "StartY",
        "Alku X": "Start X",
        "Alku Y": "Start Y",
        "Loppu X": "End X",
        "Loppu Y": "End Y"
    }
    
    # Create a copy to avoid modifying original
    df_translated = df.copy()
    
    # Translate column names if they're in Finnish
    column_name_translations = {
        "Nimi": "Name",
        "Primääri": "Primary", 
        "Sekundääri": "Secondary",
        "Kehonosa": "Body Part",
        "Puolisko": "Half",
        "Joukkue": "Team",
        "Aika": "Time",
        "Alkupiste X": "StartX",
        "Alkupiste Y": "StartY",
        "syötön loppupiste X": "End X",
        "syötön loppupiste Y": "End Y",
        "X2": "End X",
        "Y2": "End Y",
        "Alku X": "StartX",
        "Alku Y": "StartY",
        "Loppu X": "End X",
        "Loppu Y": "End Y"
        # Note: "End X" and "End Y" should stay as-is (with space) as the analysis module expects them
    }
    
    df_translated.columns = [column_name_translations.get(col, col) for col in df_translated.columns]
    
    # Translate values in specific columns
    for col in df_translated.columns:
        if col in ['Name', 'Primary', 'Secondary', 'Body Part', 'Half', 'Team']:
            df_translated[col] = df_translated[col].map(lambda x: finnish_to_english.get(str(x), x) if pd.notna(x) else x)
    
    return df_translated

# === HELPER FUNCTIONS ===
def load_data_from_upload(uploaded_file):
    """Load and translate data from uploaded file"""
    try:
        uploaded_file.seek(0)  # Ensure pointer is at the start
        df = pd.read_csv(uploaded_file)
        # Translate Finnish data to English for processing
        df = translate_finnish_to_english_data(df)
        filename = uploaded_file.name
        team_name = os.path.splitext(filename)[0]
        team_name = team_name.split(" ")[0]
        return df, team_name
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        return None, None

def parse_date_from_filename(filename):
    """Extract date from filename format 'team_name dd.mm.csv'"""
    # Look for date patterns like "27.7", "2.8", "14.9" at the end of filename
    match = re.search(r'(\d{1,2})\.(\d{1,2})(?:\.csv)?$', filename)
    if match:
        day, month = match.groups()
        # Assume current year for simplicity, or you can modify this logic
        year = 2024
        try:
            return datetime(year, int(month), int(day))
        except ValueError:
            return datetime.min
    return datetime.min

def get_file_display_name_with_date(filename):
    """Create a display name for file selection with date parsing"""
    date = parse_date_from_filename(filename)
    
    if date != datetime.min:
        date_str = date.strftime("%d.%m")
        team_name = filename.replace('.csv', '').split(' ')[0]
        return f"{team_name} ({date_str})"
    else:
        return filename.replace('.csv', '')

def get_avg_kpis_from_uploads(files_list, file_map, xgb_model, expected_cols, selected_half):
    """Calculate average KPIs from uploaded files"""
    kpi_list_jjk = []
    kpi_list_opp = []
    for display_name in files_list:
        uploaded_file = file_map[display_name]
        df, _ = load_data_from_upload(uploaded_file)
        if df is not None:
            # Filter by half if needed
            if selected_half == "1. puoliaika":
                df = df[df['Half'] == '1. puoliaika']
            elif selected_half == "2. puoliaika":
                df = df[df['Half'] == '2. puoliaika']
            jjk_vals, opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
            kpi_list_jjk.append(jjk_vals)
            kpi_list_opp.append(opp_vals)
    avg_jjk = np.nanmean(kpi_list_jjk, axis=0) if kpi_list_jjk else None
    avg_opp = np.nanmean(kpi_list_opp, axis=0) if kpi_list_opp else None
    return avg_jjk, avg_opp

# === PDF EXPORT FUNCTIONS ===
def create_match_report_pdf(df, xgb_model, expected_cols, team_name, selected_half, current_match_jjk_vals, current_match_opp_vals, avg_jjk, avg_opp, std_jjk, std_opp, match_date_str=""):
    """Create a 3-page PDF report with KPI Table & Shot Map, Radar Chart, and Cumulative xG & Z-Score Analysis"""
    
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        
        # Generate all figures first
        row_labels, jjk_stats, opp_stats = extract_full_match_stats(df, xgb_model, expected_cols, team_name)
        kpi_fig = generate_kpi_table(row_labels, jjk_stats, opp_stats, team_name)
        pitch_fig = plot_combined_pitch_vertical(df, xgb_model, expected_cols, team_name)
        xg_fig = plot_cumulative_xg(df, xgb_model, expected_cols, team_name, selected_half)
        
        # Generate radar chart
        radar_fig = None
        if (avg_jjk is not None and avg_opp is not None and
            not np.any(np.isnan(current_match_jjk_vals)) and
            not np.any(np.isnan(current_match_opp_vals)) and
            not np.any(np.isnan(avg_jjk)) and
            not np.any(np.isnan(avg_opp))):
            radar_fig = generate_radar_chart(current_match_jjk_vals, current_match_opp_vals)
        
        # Generate z-score analysis
        zscore_fig = None
        if (avg_jjk is not None and avg_opp is not None and
            not np.any(np.isnan(current_match_jjk_vals)) and
            not np.any(np.isnan(current_match_opp_vals)) and
            not np.any(np.isnan(avg_jjk)) and
            not np.any(np.isnan(avg_opp))):
            zscore_fig, debug_text_match = display_jjk_kpi_stats_vertical(current_match_jjk_vals, current_match_opp_vals, 
                                                       avg_jjk, avg_opp, std_jjk, std_opp, team_name)
        
        # === PAGE 1: KPI Table and Shot Map ===
        fig1 = plt.figure(figsize=(11.69, 8.27))
        fig1.patch.set_facecolor('#122c3d')
        
        # Add main title with date if available
        title_with_date = f"Otteluraportti: JJK vs {team_name}" + (f" {match_date_str}" if match_date_str else "")
        fig1.suptitle(title_with_date, 
                     fontsize=18, fontweight='bold', y=0.92, color='white')
        
        # Add JJK logo from GitHub repo (main branch, next to app_upload.py)
        try:
            logo_url = "https://raw.githubusercontent.com/Leiskka/jjk_streamlit/main/jjk-logo-VERKKO-1024x972.png"
            import requests
            from PIL import Image
            logo_response = requests.get(logo_url)
            logo_response.raise_for_status()
            logo_img = Image.open(io.BytesIO(logo_response.content))
            logo_ax = fig1.add_axes([0.83, 0.87, 0.12, 0.08], zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        except Exception:
            pass
        
        # Create 2-column layout for page 1 - similar to jaksoraportti
        gs1 = fig1.add_gridspec(1, 2, 
                               width_ratios=[1.2, 0.9],  # More space for KPI table, less for shot map
                               hspace=0.1, wspace=0.05,  # Small gap between columns
                               left=0.05, right=0.95, top=0.85, bottom=0.08)  # Use more page width
        
        # Page 1, Left: KPI Table - use full allocated space
        ax1_1 = fig1.add_subplot(gs1[0, 0])
        kpi_buf = io.BytesIO()
        kpi_fig.savefig(kpi_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        kpi_buf.seek(0)
        kpi_img = plt.imread(kpi_buf)
        ax1_1.imshow(kpi_img)
        ax1_1.axis('off')
        plt.close(kpi_fig)
        
        # Page 1, Right: Shot Map
        ax1_2 = fig1.add_subplot(gs1[0, 1])
        pitch_buf = io.BytesIO()
        pitch_fig.savefig(pitch_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        pitch_buf.seek(0)
        pitch_img = plt.imread(pitch_buf)
        ax1_2.imshow(pitch_img, aspect='equal', origin='upper')
        ax1_2.set_aspect('equal')
        ax1_2.set_anchor('C')
        ax1_2.axis('off')
        plt.close(pitch_fig)
        
        # Set backgrounds
        for ax in [ax1_1, ax1_2]:
            ax.set_facecolor('#122c3d')
        
        # Add endnote for page 1
        fig1.text(0.9, 0.02, "Grafiikka: Eino Neuvonen", ha='right', va='bottom', 
                 fontsize=9, color='white', fontstyle='italic')
        
        # Save page 1
        pdf.savefig(fig1, dpi=300, facecolor='#122c3d')
        plt.close(fig1)
        
        # === PAGE 2: Radar Chart ===
        fig2 = plt.figure(figsize=(11.69, 8.27))
        fig2.patch.set_facecolor('#122c3d')
        
        # Create single subplot for radar chart (full page)
        gs2 = fig2.add_gridspec(1, 1, 
                               left=0.08, right=0.92, top=0.85, bottom=0.08)  # Print-friendly margins
        
        # Page 2: Radar Chart (full page)
        ax2_1 = fig2.add_subplot(gs2[0, 0])
        if radar_fig is not None:
            radar_buf = io.BytesIO()
            radar_fig.savefig(radar_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
            radar_buf.seek(0)
            radar_img = plt.imread(radar_buf)
            ax2_1.imshow(radar_img)
            plt.close(radar_fig)
        else:
            ax2_1.text(0.5, 0.5, 'Radar Chart\nNot Available\n(Need multiple matches for baseline)', 
                    ha='center', va='center', transform=ax2_1.transAxes, 
                    fontsize=14, color='white', bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a3345"))
        ax2_1.axis('off')
        ax2_1.set_facecolor('#122c3d')
        
        # Add endnote for page 2
        fig2.text(0.9, 0.02, "Grafiikka: Eino Neuvonen", ha='right', va='bottom', 
                 fontsize=9, color='white', fontstyle='italic')
        
        # Save page 2
        pdf.savefig(fig2, dpi=300, facecolor='#122c3d')
        plt.close(fig2)
        
        # === PAGE 3: Cumulative xG and Z-Score Analysis ===
        fig3 = plt.figure(figsize=(11.69, 8.27))
        fig3.patch.set_facecolor('#122c3d')
        
        # Create 2-column layout for page 3
        gs3 = fig3.add_gridspec(1, 2, 
                               width_ratios=[1.0, 1.0],  # Equal width columns
                               hspace=0.1, wspace=0.05,
                               left=0.05, right=0.95, top=0.85, bottom=0.08)
        
        # Page 3, Left: Cumulative xG
        ax3_1 = fig3.add_subplot(gs3[0, 0])
        # Add title above cumulative xG chart
        ax3_1.text(0.5, 1.1, 'Kumulatiivinen maaliodottama', ha='center', va='bottom',
                   transform=ax3_1.transAxes, fontsize=12, fontweight='bold', color='white')
        
        xg_buf = io.BytesIO()
        xg_fig.savefig(xg_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        xg_buf.seek(0)
        xg_img = plt.imread(xg_buf)
        ax3_1.imshow(xg_img)
        ax3_1.axis('off')
        plt.close(xg_fig)
        
        # Page 3, Right: Z-Score Analysis
        ax3_2 = fig3.add_subplot(gs3[0, 1])
        # Add title above z-score analysis
        ax3_2.text(0.5, 1.1, 'Suhteellinen poikkeama keskiarvosta', ha='center', va='bottom',
                   transform=ax3_2.transAxes, fontsize=12, fontweight='bold', color='white')
        
        if zscore_fig is not None:
            zscore_buf = io.BytesIO()
            zscore_fig.savefig(zscore_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
            zscore_buf.seek(0)
            zscore_img = plt.imread(zscore_buf)
            ax3_2.imshow(zscore_img)
            plt.close(zscore_fig)
        else:
            ax3_2.text(0.5, 0.5, 'Z-Score Analysis\nNot Available\n(Need multiple matches for baseline)', 
                    ha='center', va='center', transform=ax3_2.transAxes, 
                    fontsize=12, color='white', bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a3345"))
        ax3_2.axis('off')
        
        # Set backgrounds
        for ax in [ax3_1, ax3_2]:
            ax.set_facecolor('#122c3d')
        
        # Add endnote for page 3
        fig3.text(0.9, 0.02, "Grafiikka: Eino Neuvonen", ha='right', va='bottom', 
                 fontsize=9, color='white', fontstyle='italic')
        
        # Save page 3
        pdf.savefig(fig3, dpi=300, facecolor='#122c3d')
        plt.close(fig3)
    
    buffer.seek(0)
    return buffer

def create_jaksoraportti_report_pdf(uploaded_files, uploaded_dfs, xgb_model, expected_cols, selected_half):
    
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    # Parse dates from filenames and create title with date range and match count
    def parse_date_from_filename_pdf(filename):
        import re
        from datetime import datetime
        # Look for date patterns like "27.7", "2.8", "14.9" at the end of filename
        match = re.search(r'(\d{1,2})\.(\d{1,2})(?:\.csv)?$', filename)
        if match:
            day, month = match.groups()
            # Assume current year for simplicity, or you can modify this logic
            year = 2024
            try:
                return datetime(year, int(month), int(day))
            except ValueError:
                return None
        return None
    
    # Extract dates and create title
    dates = []
    for uploaded_file in uploaded_files:
        date = parse_date_from_filename_pdf(uploaded_file.name)
        if date:
            dates.append(date)
    
    # Create title with date range and match count
    match_count = len(uploaded_files)
    if dates:
        dates.sort()
        start_date = dates[0].strftime("%d.%m")
        end_date = dates[-1].strftime("%d.%m") 
        title = f"Jaksoraportti {start_date} - {end_date} ({match_count} ottelua)"
    else:
        title = f"Jaksoraportti ({match_count} ottelua)"
    
    with PdfPages(buffer) as pdf:
        # === PAGE 1: KPI Table and Shot Map ===
        fig1 = plt.figure(figsize=(11.69, 8.27))
        fig1.patch.set_facecolor('#122c3d')
        
        # Add main title with date range and match count
        fig1.suptitle(title, 
                     fontsize=18, fontweight='bold', y=0.92, color='white')
        
        # Add JJK logo from GitHub repo (main branch, next to app_upload.py)
        try:
            logo_url = "https://raw.githubusercontent.com/Leiskka/jjk_streamlit/main/jjk-logo-VERKKO-1024x972.png"
            import requests
            from PIL import Image
            logo_response = requests.get(logo_url)
            logo_response.raise_for_status()
            logo_img = Image.open(io.BytesIO(logo_response.content))
            logo_ax = fig1.add_axes([0.83, 0.87, 0.12, 0.08], zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')
        except Exception:
            pass
        
        # Create 2-column layout for page 1 - give more space to KPI table
        gs1 = fig1.add_gridspec(1, 2, 
                               width_ratios=[1.2, 0.9],  # More space for KPI table, less for shot map
                               hspace=0.1, wspace=0.05,  # Small gap between columns
                               left=0.05, right=0.95, top=0.85, bottom=0.08)  # Use more page width
        
        # Generate figures
        row_labels, avg_jjk_stats, avg_opp_stats = extract_aggregate_stats(uploaded_dfs, xgb_model, expected_cols, selected_half)
        kpi_fig = generate_kpi_table(row_labels, avg_jjk_stats, avg_opp_stats, "Vastustaja")
        pitch_fig = plot_all_shots_pitch(uploaded_dfs, xgb_model, expected_cols, selected_half)
        
        # Page 1, Left: KPI Table - use full allocated space
        ax1_1 = fig1.add_subplot(gs1[0, 0])
        # Use the full space allocated to the KPI table for maximum size
        
        kpi_buf = io.BytesIO()
        kpi_fig.savefig(kpi_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        kpi_buf.seek(0)
        kpi_img = plt.imread(kpi_buf)
        ax1_1.imshow(kpi_img)
        ax1_1.axis('off')
        plt.close(kpi_fig)
        
        # Page 1, Right: Shot Map
        ax1_2 = fig1.add_subplot(gs1[0, 1])
        pitch_buf = io.BytesIO()
        pitch_fig.savefig(pitch_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        pitch_buf.seek(0)
        pitch_img = plt.imread(pitch_buf)
        ax1_2.imshow(pitch_img, aspect='equal', origin='upper')
        ax1_2.set_aspect('equal')
        ax1_2.set_anchor('C')
        ax1_2.axis('off')
        plt.close(pitch_fig)
        
        # Set backgrounds
        for ax in [ax1_1, ax1_2]:
            ax.set_facecolor('#122c3d')
        
        # Add endnote for page 1
        fig1.text(0.9, 0.02, "Grafiikka: Eino Neuvonen", ha='right', va='bottom', 
                 fontsize=9, color='white', fontstyle='italic')
        
        # Save page 1
        pdf.savefig(fig1, dpi=300, facecolor='#122c3d')
        plt.close(fig1)
        
        # === PAGE 2: Average Radar Chart ===
        fig2 = plt.figure(figsize=(11.69, 8.27))
        fig2.patch.set_facecolor('#122c3d')
        
        # Create single subplot for radar chart (full page)
        gs2 = fig2.add_gridspec(1, 1, 
                               left=0.08, right=0.92, top=0.85, bottom=0.08)  # Print-friendly margins
        
        # Generate radar chart
        radar_fig = generate_average_radar_chart(uploaded_dfs, xgb_model, expected_cols, selected_half)
        
        # Page 2: Radar Chart (full page)
        ax2_1 = fig2.add_subplot(gs2[0, 0])
        radar_buf = io.BytesIO()
        radar_fig.savefig(radar_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        radar_buf.seek(0)
        radar_img = plt.imread(radar_buf)
        ax2_1.imshow(radar_img)
        ax2_1.axis('off')
        ax2_1.set_facecolor('#122c3d')
        plt.close(radar_fig)
        
        # Add endnote for page 2
        fig2.text(0.9, 0.02, "Grafiikka: Eino Neuvonen", ha='right', va='bottom', 
                 fontsize=9, color='white', fontstyle='italic')
        
        # Save page 2
        pdf.savefig(fig2, dpi=300, facecolor='#122c3d')
        plt.close(fig2)
        
        # === PAGE 3: Z-Score Development ===
        fig3 = plt.figure(figsize=(11.69, 8.27))
        fig3.patch.set_facecolor('#122c3d')
        
        # Create single subplot for z-score development (full page)
        gs3 = fig3.add_gridspec(1, 1, 
                               left=0.08, right=0.92, top=0.85, bottom=0.08)  # Print-friendly margins
        
        # Generate z-score development figure
        development_fig, debug_text_dev = plot_zscore_development(uploaded_files, xgb_model, expected_cols, selected_half)
        
        # Page 3: Z-Score Development (full page)
        ax3_1 = fig3.add_subplot(gs3[0, 0])
        dev_buf = io.BytesIO()
        development_fig.savefig(dev_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#122c3d', dpi=200)
        dev_buf.seek(0)
        dev_img = plt.imread(dev_buf)
        ax3_1.imshow(dev_img)
        ax3_1.axis('off')
        ax3_1.set_facecolor('#122c3d')
        plt.close(development_fig)
        
        # Add endnote for page 3
        fig3.text(0.9, 0.02, "Grafiikka: Eino Neuvonen", ha='right', va='bottom', 
                 fontsize=9, color='white', fontstyle='italic')
        
        # Save page 3
        pdf.savefig(fig3, dpi=300, facecolor='#122c3d')
        plt.close(fig3)
    
    buffer.seek(0)
    return buffer

# Process uploaded files and create file selection
if uploaded_files:
    # Create file options with dates
    file_options = []
    file_map = {}
    
    for uploaded_file in uploaded_files:
        display_name = get_file_display_name_with_date(uploaded_file.name)
        file_options.append(display_name)
        file_map[display_name] = uploaded_file
    
    # Sort by date (most recent first)
    file_options.sort(key=lambda x: parse_date_from_filename(file_map[x].name), reverse=True)
    
    match_files = file_options
    default_index = 0
else:
    match_files = []
    default_index = 0

match_file_name = st.sidebar.selectbox("Valitse ottelu", match_files, index=default_index if match_files else 0)

# Add half selection dropdown under match selection
half_options = ["Täysi ottelu", "1. puoliaika", "2. puoliaika"]
selected_half = st.sidebar.selectbox("Valitse puoliaika", half_options)

# === MAIN ===
if match_file_name and match_files:
    # Find the selected file info
    file_info = file_map[match_file_name]
    df, team_name = load_data_from_upload(file_info)
    
    if df is not None:
        
        # Get match date
        match_date = parse_date_from_filename(file_info.name)
        match_date_str = match_date.strftime("%d.%m") if match_date != datetime.min else ""

        # Filter df by half selection
        if selected_half == "1. puoliaika":
            df = df[df['Half'] == '1. puoliaika']
        elif selected_half == "2. puoliaika":
            df = df[df['Half'] == '2. puoliaika']

        # Add both PDF export buttons next to title
        col_title, col_export1, col_export2 = st.columns([3, 1, 1])
        with col_title:
            pass  # Title already created above
        
        # Match Report PDF Export
        with col_export1:
            # Extract KPIs early for PDF generation
            current_match_jjk_vals, current_match_opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
            avg_jjk, avg_opp = get_avg_kpis_from_uploads(match_files, file_map, xgb_model, expected_cols, selected_half)
            
            # Calculate standard deviations
            kpi_list_jjk = []
            kpi_list_opp = []
            for display_name in match_files:
                uploaded_file_temp = file_map[display_name]
                df_temp, _ = load_data_from_upload(uploaded_file_temp)
                if df_temp is not None:
                    if selected_half == "1. puoliaika":
                        df_temp = df_temp[df_temp['Half'] == '1. puoliaika']
                    elif selected_half == "2. puoliaika":
                        df_temp = df_temp[df_temp['Half'] == '2. puoliaika']
                    jjk_vals_temp, opp_vals_temp = extract_radar_kpis(df_temp, xgb_model, expected_cols)
                    kpi_list_jjk.append(jjk_vals_temp)
                    kpi_list_opp.append(opp_vals_temp)
            
            jjk_arr = np.array(kpi_list_jjk)
            opp_arr = np.array(kpi_list_opp)
            std_jjk = np.nanstd(jjk_arr, axis=0) if len(jjk_arr) > 0 else np.ones(len(current_match_jjk_vals))
            std_opp = np.nanstd(opp_arr, axis=0) if len(opp_arr) > 0 else np.ones(len(current_match_opp_vals))
            
            if st.button("Otteluraportti PDF", key="pdf_export", help="Lataa otteluraportti PDF-muodossa"):
                try:
                    pdf_buffer = create_match_report_pdf(df, xgb_model, expected_cols, team_name, selected_half, 
                                                       current_match_jjk_vals, current_match_opp_vals, 
                                                       avg_jjk, avg_opp, std_jjk, std_opp, match_date_str)
                    st.download_button(
                        label="Lataa PDF",
                        data=pdf_buffer,
                        file_name=f"otteluraportti_{team_name}_{match_date_str}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Virhe PDF:n luomisessa: {str(e)}")

        # Jaksoraportti PDF Export (only show if multiple files uploaded)
        with col_export2:
            if len(uploaded_files) >= 2:
                if st.button("Jaksoraportti PDF", key="pdf_export_jaksoraportti", help="Lataa jaksoraportti PDF-muodossa"):
                    try:
                        # Convert uploaded files to dataframes for jaksoraportti
                        uploaded_dfs = []
                        for f in uploaded_files:
                            f.seek(0)
                            df_temp, _ = load_data_from_upload(f)
                            if selected_half == "1. puoliaika":
                                df_temp = df_temp[df_temp['Half'] == '1. puoliaika']
                            elif selected_half == "2. puoliaika":
                                df_temp = df_temp[df_temp['Half'] == '2. puoliaika']
                            uploaded_dfs.append(df_temp)
                        
                        pdf_buffer = create_jaksoraportti_report_pdf(
                            uploaded_files, uploaded_dfs, xgb_model, expected_cols, selected_half
                        )
                        st.download_button(
                            label="Lataa Jaksoraportti PDF", 
                            data=pdf_buffer,
                            file_name=f"jaksoraportti_{len(uploaded_files)}_ottelua.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Virhe PDF:n luomisessa: {str(e)}")
            else:
                st.info("Lataa vähintään 2 ottelua jaksoraporttia varten.")

        # Extract KPIs for the current selected match and calculate averages (if not already done for PDF)
        if 'current_match_jjk_vals' not in locals():
            current_match_jjk_vals, current_match_opp_vals = extract_radar_kpis(df, xgb_model, expected_cols)
            avg_jjk, avg_opp = get_avg_kpis_from_uploads(match_files, file_map, xgb_model, expected_cols, selected_half)
            
            # Calculate all match KPIs for statistics
            kpi_list_jjk = []
            kpi_list_opp = []
            for display_name in match_files:
                uploaded_file_loop = file_map[display_name]
                df_loop, _ = load_data_from_upload(uploaded_file_loop)
                if df_loop is not None:
                    if selected_half == "1. puoliaika":
                        df_loop = df_loop[df_loop['Half'] == '1. puoliaika']
                    elif selected_half == "2. puoliaika":
                        df_loop = df_loop[df_loop['Half'] == '2. puoliaika']
                    jjk_vals_loop, opp_vals_loop = extract_radar_kpis(df_loop, xgb_model, expected_cols)
                    kpi_list_jjk.append(jjk_vals_loop)
                    kpi_list_opp.append(opp_vals_loop)

            # Calculate standard deviations
            jjk_arr = np.array(kpi_list_jjk)
            opp_arr = np.array(kpi_list_opp)
            std_jjk = np.nanstd(jjk_arr, axis=0) if len(jjk_arr) > 0 else np.ones(len(current_match_jjk_vals))
            std_opp = np.nanstd(opp_arr, axis=0) if len(opp_arr) > 0 else np.ones(len(current_match_opp_vals))

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

        tab1, tab_jaksoraportti, tab_heatmap = st.tabs(["Otteluraportti", "Jaksoraportti", "Lämpökartta"])

        with tab1:
            # Add date to dashboard title if available
            title_with_date = f"Otteluraportti: JJK vs {team_name}" + (f" {match_date_str}" if match_date_str else "")
            st.title(title_with_date)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Tilastot")
                row_labels, jjk_stats, opp_stats = extract_full_match_stats(df, xgb_model, expected_cols, team_name)
                st.pyplot(generate_kpi_table(row_labels, jjk_stats, opp_stats, team_name), use_container_width=True)
                st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
                
            with col2:
                st.subheader("Kumulatiivinen maaliodottama")
                st.pyplot(plot_cumulative_xg(df, xgb_model, expected_cols, team_name, selected_half), use_container_width=True)
                st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

            # Second row: Z-Score Plot and Pitch Events
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Suhteellinen poikkeama keskiarvosta (Z-arvot)")
                # Use the pre-calculated values for vertical z-score display
                if (
                    avg_jjk is not None and avg_opp is not None and
                    not np.any(np.isnan(current_match_jjk_vals)) and
                    not np.any(np.isnan(current_match_opp_vals)) and
                    not np.any(np.isnan(avg_jjk)) and
                    not np.any(np.isnan(avg_opp))
                ):
                    kpi_fig, debug_text_match_tab = display_jjk_kpi_stats_vertical(current_match_jjk_vals, current_match_opp_vals, 
                                                   avg_jjk, avg_opp, std_jjk, std_opp, team_name)
                    st.pyplot(kpi_fig)
                    

                else:
                    st.info("Ei riittävästi dataa.")
                
            with col4:
                st.subheader("Laukaisukartta")
                # Add vertical pitch visualization
                pitch_fig = plot_combined_pitch_vertical(df, xgb_model, expected_cols, team_name)
                st.pyplot(pitch_fig)

            # Third row: Radar Chart (centered)
            col5, col6, col7 = st.columns([1, 2, 1])
            with col6:
                st.subheader("Tilastot sädekaavio")
                # Use the already extracted values for consistency
                if (
                    avg_jjk is not None and avg_opp is not None and
                    not np.any(np.isnan(current_match_jjk_vals)) and
                    not np.any(np.isnan(current_match_opp_vals)) and
                    not np.any(np.isnan(avg_jjk)) and
                    not np.any(np.isnan(avg_opp)) and
                    np.all(np.array([
                        max(current_match_jjk_vals[i], current_match_opp_vals[i], avg_jjk[i], avg_opp[i]) >
                        min(current_match_jjk_vals[i], current_match_opp_vals[i], avg_jjk[i], avg_opp[i])
                        for i in range(len(current_match_jjk_vals))
                    ]))
                ):
                    radar_fig = generate_radar_chart(
                        current_match_jjk_vals, current_match_opp_vals
                    )
                    st.pyplot(radar_fig, use_container_width=True)
                else:
                    st.info("Ei riittävästi dataa.")

        with tab_heatmap:
            st.write("Valitse analysoitava kenttätapahtuma, jonka lämpökartta näytetään ladatuista otteluista")
            heatmap_options = [
                "Laukaukset",
                "Box Entryt (syötön lähtöpiste)",
                "Box Entryt (syötön loppupiste)", 
                "Box Entryt (kuljettaen)",
                "Onnistuneet syötöt taakse (syötön lähtöpiste)",
                "Epäonnistuneet syötöt taakse (syötön lähtöpiste)", 
                "Onnistuneet syötöt taakse (syötön loppupiste)",
                "Epäonnistuneet syötöt taakse (syötön loppupiste)",
                "Pallonriistot",
                "Pallonriistot (Puolustuskaksinkamppailut)", 
                "Pallonriistot (Syötönkatkot)"
            ]
            selected_heatmap = st.selectbox("Valitse tapahtuma lämpökartalle", heatmap_options)
            
            selected_team = st.radio("Valitse joukkue", ["JJK", "VAS"], horizontal=True, key="heatmap_team")
            
            if uploaded_files:
                dfs = [load_data_from_upload(f)[0] for f in uploaded_files]
                st.pyplot(plot_event_heatmap(dfs, selected_heatmap, selected_half, selected_team), use_container_width=True)
            else:
                st.info(f"Lämpökartta {selected_heatmap} näytetään täällä.")

        with tab_jaksoraportti:
            if len(uploaded_files) >= 2:
                # Create title with same format as PDF (date range and match count)
                def parse_date_from_filename_tab(filename):
                    import re
                    from datetime import datetime
                    # Look for date patterns like "27.7", "2.8", "14.9" at the end of filename
                    match = re.search(r'(\d{1,2})\.(\d{1,2})(?:\.csv)?$', filename)
                    if match:
                        day, month = match.groups()
                        # Assume current year for simplicity, or you can modify this logic
                        year = 2024
                        try:
                            return datetime(year, int(month), int(day))
                        except ValueError:
                            return None
                    return None
                
                # Extract dates and create title
                dates = []
                for uploaded_file in uploaded_files:
                    date = parse_date_from_filename_tab(uploaded_file.name)
                    if date:
                        dates.append(date)
                
                # Create title with date range and match count
                match_count = len(uploaded_files)
                if dates:
                    dates.sort()
                    start_date = dates[0].strftime("%d.%m")
                    end_date = dates[-1].strftime("%d.%m") 
                    jaksoraportti_title = f"Jaksoraportti {start_date} - {end_date} ({match_count} ottelua)"
                else:
                    jaksoraportti_title = f"Jaksoraportti ({match_count} ottelua)"
                
                st.title(jaksoraportti_title)

                # First row: Aggregate KPI Table and Average Radar Chart
                col1_k, col2_k = st.columns(2)
                
                with col1_k:
                    st.subheader("Tilastokeskiarvot")
                    # Convert uploaded files to dataframes for aggregate stats
                    uploaded_dfs = []
                    for f in uploaded_files:
                        f.seek(0)
                        df_temp, _ = load_data_from_upload(f)
                        if selected_half == "1. puoliaika":
                            df_temp = df_temp[df_temp['Half'] == '1. puoliaika']
                        elif selected_half == "2. puoliaika":
                            df_temp = df_temp[df_temp['Half'] == '2. puoliaika']
                        uploaded_dfs.append(df_temp)
                    
                    row_labels, avg_jjk_stats, avg_opp_stats = extract_aggregate_stats(uploaded_dfs, xgb_model, expected_cols, selected_half)
                    st.pyplot(generate_kpi_table(row_labels, avg_jjk_stats,  avg_opp_stats, "Vastustaja"), use_container_width=True)
                    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
                    
                with col2_k:
                    st.subheader("Tilastokeskiarvot sädekaavio")
                    avg_radar_fig = generate_average_radar_chart(uploaded_dfs, xgb_model, expected_cols, selected_half)
                    st.pyplot(avg_radar_fig, use_container_width=True)
                    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

                # Second row: Z-Score Development (full width)
                st.subheader("Suhteellinen kehitys (Z-arvo-analyysi)")
                zscore_dev_fig, debug_text_dev_tab = plot_zscore_development(uploaded_files, xgb_model, expected_cols, selected_half)
                st.pyplot(zscore_dev_fig, use_container_width=True)
                

                st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
                
                # Third row: All Shots Pitch (full width)
                st.subheader("Laukaisukartta")
                all_shots_fig = plot_all_shots_pitch(uploaded_dfs, xgb_model, expected_cols, selected_half)
                st.pyplot(all_shots_fig, use_container_width=True)
                    
            else:
                st.info("Lataa vähintään 2 ottelua jaksoraporttia varten.")

else:
    st.info("Lataa CSV-tiedostot sivupalkista aloittaaksesi.")
