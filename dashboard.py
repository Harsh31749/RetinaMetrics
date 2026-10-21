import os
import numpy as np
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt
import streamlit as st
from retina_engine import segment_vessels, evaluate_scan, run_engine, parse_ehr_records

st.set_page_config(page_title="RetinaMetrics Dashboard", layout="wide")

if not os.path.exists("outputs/retinametrics_results.csv"):
    with st.spinner("Running RetinaMetrics Engine on STARE dataset..."):
        run_engine()

if not os.path.exists("outputs/stare_ehr_cohort.csv"):
    parse_ehr_records()

@st.cache_data
def load_data():
    results_df = pd.read_csv("outputs/retinametrics_results.csv")
    if os.path.exists("outputs/stare_ehr_cohort.csv"):
        ehr_df = pd.read_csv("outputs/stare_ehr_cohort.csv")
        if ehr_df.empty:
            ehr_df = results_df[["Patient_ID", "Primary_Diagnosis", "Disease_Category"]].copy()
    else:
        ehr_df = results_df[["Patient_ID", "Primary_Diagnosis", "Disease_Category"]].copy()
    return results_df, ehr_df

results_df, ehr_df = load_data()
live_results_df = results_df.copy() # Create a working copy for live updates

# --- HEADER & COHORT KPIs ---
st.title("RetinaMetrics: Retinal Vessel Extraction & Diagnostic Visualization")
st.markdown("**Dataset:** STARE (Structured Analysis of the Retina) | **Pipeline:** Green Channel → CLAHE → Multi-Scale Frangi → Otsu Binarization")

# --- RENDER TOP KPIs DYNAMICALLY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Benchmark Scans Evaluated", f"{len(live_results_df)} Scans")
col2.metric("Mean Pixel Accuracy", f"{live_results_df['Pixel_Accuracy_%'].mean():.2f}%")
col3.metric("Mean Dice Similarity", f"{live_results_df['Dice_Score'].mean():.4f}")
col4.metric("Mean Vascular Density", f"{live_results_df['Vascular_Density_%'].mean():.2f}%")
tab1, tab2 = st.tabs([
    "Tab 1: Patient Scan & Live Segmentation Inspector",
    "Tab 2: Clinical EHR & Cohort Visual Analytics"
])

# --- TAB 1: SINGLE PATIENT BIOMEDICAL IMAGE INSPECTOR ---
with tab1:
    st.sidebar.header("Patient Selection")
    patient_ids = results_df["Patient_ID"].tolist()
    selected_pid = st.sidebar.selectbox("Select STARE Patient ID:", patient_ids, index=0)

    st.sidebar.divider()
    st.sidebar.header("Live Pipeline Parameters")
    clahe_clip = st.sidebar.slider("CLAHE Contrast Limit", min_value=0.01, max_value=0.10, value=0.03, step=0.01)
    max_sigma = st.sidebar.slider("Max Frangi Vessel Scale (σ)", min_value=2, max_value=6, value=4, step=1)
    min_obj_size = st.sidebar.slider("Morphological Noise Filter (px)", min_value=10, max_value=150, value=80, step=10)

    patient_row = results_df[results_df["Patient_ID"] == selected_pid].iloc[0]

    # Load and dynamically segment with current slider values
    img_path = os.path.join("data/images", f"{selected_pid}.ppm")
    label_dir = "data/labels" if os.path.exists(f"data/labels/{selected_pid}.ah.ppm") else "data/labels_ah"
    gt_path = os.path.join(label_dir, f"{selected_pid}.ah.ppm")

    img = iio.imread(img_path)
    gt_mask = iio.imread(gt_path)
    green, clahe, vesselness, pred_mask, fov_mask = segment_vessels(
        img, clip_limit=clahe_clip, max_sigma=max_sigma, min_size=min_obj_size
    )
    live_metrics = evaluate_scan(pred_mask, gt_mask, fov_mask)

    # --- INJECT LIVE METRICS INTO COHORT DATAFRAME ---
    row_idx = live_results_df.index[live_results_df['Patient_ID'] == selected_pid].tolist()[0]
    live_results_df.at[row_idx, 'Vascular_Density_%'] = live_metrics['Vascular_Density_%']
    live_results_df.at[row_idx, 'Pixel_Accuracy_%'] = live_metrics['Pixel_Accuracy_%']
    live_results_df.at[row_idx, 'Dice_Score'] = live_metrics['Dice_Score']
    live_results_df.at[row_idx, 'Sensitivity_%'] = live_metrics['Sensitivity_%']

    st.subheader(f"Patient Record: `{selected_pid}`")
    info_col1, info_col2, info_col3, info_col4 = st.columns([2.2, 1, 1, 1])
    info_col1.info(f"**Clinical EHR Diagnosis:** {patient_row['Primary_Diagnosis']}  \n**Category:** {patient_row['Disease_Category']}")
    info_col2.metric("Live Vascular Density", f"{live_metrics['Vascular_Density_%']:.2f}%")
    info_col3.metric("Live Pixel Accuracy", f"{live_metrics['Pixel_Accuracy_%']:.2f}%")
    info_col4.metric("Live Dice / Sensitivity", f"{live_metrics['Dice_Score']:.3f} / {live_metrics['Sensitivity_%']:.2f}%")

    # Create validation overlay (Green = Match, Red = Missed Vessel, Blue = Over-segmented)
    overlay = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    gt_bin = (gt_mask > 0) & fov_mask
    overlay[np.logical_and(pred_mask, gt_bin)] = [0.0, 0.9, 0.2]       # Green: True Positive
    overlay[np.logical_and(~pred_mask, gt_bin)] = [1.0, 0.1, 0.1]      # Red: False Negative
    overlay[np.logical_and(pred_mask, ~gt_bin)] = [0.1, 0.5, 1.0]      # Blue: False Positive

    # Plot 6-panel visual analytics grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"[1] Raw RGB Fundus Scan ({selected_pid})", fontsize=11, fontweight="bold")
    axes[0, 1].imshow(clahe, cmap="gray")
    axes[0, 1].set_title(f"[2] Green + CLAHE (Clip={clahe_clip})", fontsize=11, fontweight="bold")
    axes[0, 2].imshow(vesselness, cmap="magma")
    axes[0, 2].set_title(f"[3] Frangi Vesselness (σ=1..{max_sigma})", fontsize=11, fontweight="bold")
    axes[1, 0].imshow(pred_mask, cmap="gray")
    axes[1, 0].set_title(f"[4] Segmented Mask (Density: {live_metrics['Vascular_Density_%']}%)", fontsize=11, fontweight="bold")
    axes[1, 1].imshow(gt_bin, cmap="gray")
    axes[1, 1].set_title("[5] Expert Ground Truth (Dr. Adam Hoover)", fontsize=11, fontweight="bold")
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("[6] Validation Overlay (Green=TP, Red=FN, Blue=FP)", fontsize=11, fontweight="bold")
    
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    fig.savefig(f"outputs/{selected_pid}_dashboard.png", dpi=300)
    st.caption(f"Saved high-resolution figure to `outputs/{selected_pid}_dashboard.png`.")


# --- TAB 2: COHORT EHR & BIOMARKER ANALYTICS ---
with tab2:
    st.subheader("Clinical EHR Cohort & Vascular Biomarker Analytics")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Chart A: STARE EHR Disease Distribution**")
        fig_ehr, ax_ehr = plt.subplots(figsize=(7, 4.5))
        cat_counts = ehr_df["Disease_Category"].value_counts().sort_values()
        cat_counts.plot(kind="barh", color="#2b5c8f", ax=ax_ehr)
        ax_ehr.set_xlabel("Number of Patient Records")
        ax_ehr.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_ehr)

    with chart_col2:
        st.markdown("**Chart B: Mean Vascular Density (%) by Disease Category**")
        fig_dens, ax_dens = plt.subplots(figsize=(7, 4.5))
        # Use live_results_df so the chart updates when sliders move!
        density_by_cat = live_results_df.groupby("Disease_Category")["Vascular_Density_%"].mean().sort_values()
        density_by_cat.plot(kind="barh", color="#d411cd", ax=ax_dens)
        ax_dens.set_xlabel("Mean Vascular Density (%)")
        ax_dens.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_dens)

    st.markdown("**Table C: Complete Patient Segmentation & EHR Evaluation Table**")
    
    # Style the table to highlight the selected row and keep all decimals identical
    def highlight_selected(row):
        return ['background-color: #2b5c8f' if row['Patient_ID'] == selected_pid else '' for _ in row]
    
    styled_df = live_results_df.style.apply(highlight_selected, axis=1).format({
            "Vascular_Density_%": "{:.2f}",
            "Pixel_Accuracy_%": "{:.2f}",
            "Precision_%": "{:.2f}",
            "Recall_%": "{:.2f}",
            "F1_Score": "{:.4f}",
            "IoU_Score": "{:.4f}"
    })
    
    st.dataframe(styled_df, use_container_width=True)
