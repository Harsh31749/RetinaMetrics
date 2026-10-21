import os
import re
import glob
import ssl
import urllib.request
import warnings
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage import exposure, filters, morphology

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

EHR_URL = "https://cecas.clemson.edu/~ahoover/stare/diagnoses/all-mg-codes.txt"

def parse_ehr_records(ehr_filepath="data/clinical/all-mg-codes.txt"):
    """Cleans raw STARE clinical diagnosis records and always saves stare_ehr_cohort.csv."""
    os.makedirs("data/clinical", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(ehr_filepath):
        if os.path.exists("all-mg-codes.txt"):
            ehr_filepath = "all-mg-codes.txt"
        else:
            try:
                print("Fetching clinical EHR records (all-mg-codes.txt)...")
                urllib.request.urlretrieve(EHR_URL, ehr_filepath)
            except Exception as e:
                print(f"Warning: Could not fetch EHR file ({e}).")

    records = []
    if os.path.exists(ehr_filepath):
        with open(ehr_filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                match = re.match(r"^(im\d{4})\s+(.*)$", line)
                if match:
                    pid, rest = match.groups()
                    text_matches = re.findall(r"[A-Za-z][A-Za-z\s\-\(\)\/\.]+", rest)
                    diagnosis = " | ".join([t.strip() for t in text_matches if len(t.strip()) > 2])
                    if not diagnosis:
                        diagnosis = "Unspecified Retinal Finding"

                    diag_lower = diagnosis.lower()
                    if "normal" in diag_lower:
                        category = "Normal / Healthy"
                    elif any(k in diag_lower for k in ["diabetic", "bdr", "pdr", "npdr"]):
                        category = "Diabetic Retinopathy"
                    elif any(k in diag_lower for k in ["occlusion", "crao", "crvo", "brvo", "brao"]):
                        category = "Retinal Vascular Occlusion"
                    elif "hypertensive" in diag_lower:
                        category = "Hypertensive Retinopathy"
                    elif any(k in diag_lower for k in ["amd", "macular", "drusen"]):
                        category = "Macular Degeneration"
                    else:
                        category = "Other Retinal Pathology"

                    records.append({
                        "Patient_ID": pid,
                        "Primary_Diagnosis": diagnosis,
                        "Disease_Category": category
                    })

    ehr_df = pd.DataFrame(records, columns=["Patient_ID", "Primary_Diagnosis", "Disease_Category"])
    ehr_df.to_csv("outputs/stare_ehr_cohort.csv", index=False)
    return ehr_df

def segment_vessels(img, clip_limit=0.03, max_sigma=4, min_size=80):
    """Extracts retinal vessels while suppressing camera borders and bright exudate lesions."""
    green = img[:, :, 1]
    h, w = green.shape

    # Strict circular + intensity Field-of-View (FOV) mask
    yy, xx = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    radius = min(h, w) * 0.46
    circle_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= (radius ** 2)

    fov_mask = (green > 38) & circle_mask
    fov_mask = morphology.binary_erosion(fov_mask, morphology.disk(10))

    # OPTIMIZATION: Black Top-Hat Transform to increase Precision, Recall, F1, and IoU
    green_tophat = morphology.black_tophat(green, morphology.disk(15))
    # Subtracting the top-hat output deepens the dark vessels while flattening background lighting
    enhanced_green = np.clip(green.astype(np.int16) - green_tophat.astype(np.int16), 0, 255).astype(np.uint8)

    # CLAHE Local Contrast Enhancement (Now using the enhanced_green)
    clahe = exposure.equalize_adapthist(enhanced_green, clip_limit=clip_limit)

    # Suppress ultra-bright yellow exudates
    filled = clahe.copy()
    if fov_mask.sum() > 0:
        median_bg = np.median(clahe[fov_mask])
        bright_lesions = clahe > np.percentile(clahe[fov_mask], 92)
        bright_lesions = morphology.binary_dilation(bright_lesions, morphology.disk(3))
        filled[bright_lesions] = median_bg
        filled[~fov_mask] = median_bg

    # Shade Correction + Multi-Scale Frangi Filter
    background = filters.gaussian(filled, sigma=12)
    shade_corrected = np.clip(background - filled, 0, None) * fov_mask
    frangi_resp = filters.frangi(filled, sigmas=range(1, max_sigma + 1), black_ridges=True, gamma=0.06) * fov_mask

    if fov_mask.sum() > 0:
        sc_norm = shade_corrected / (np.percentile(shade_corrected[fov_mask], 99) + 1e-8)
        fr_norm = frangi_resp / (np.percentile(frangi_resp[fov_mask], 99) + 1e-8)
        vesselness = np.clip(0.5 * sc_norm + 0.5 * fr_norm, 0, 1) * fov_mask
        thresh = filters.threshold_otsu(vesselness[fov_mask]) * 1.05
    else:
        vesselness = frangi_resp
        thresh = 0.1

    # Binarization & Morphological Cleanup
    binary_mask = (vesselness > thresh) & fov_mask
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)

    return green, clahe, vesselness, binary_mask, fov_mask

def evaluate_scan(pred_mask, gt_mask, fov_mask):
    gt_bin = (gt_mask > 0) & fov_mask
    pred_bin = pred_mask & fov_mask
    
    # Calculate True Positives, False Positives, False Negatives
    TP = np.sum((pred_bin == True) & (gt_bin == True))
    FP = np.sum((pred_bin == True) & (gt_bin == False))
    FN = np.sum((pred_bin == False) & (gt_bin == True))
    TN = np.sum((pred_bin == False) & (gt_bin == False))
    
    # Core Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    
    # Global Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    density = (np.sum(pred_bin) / np.sum(fov_mask)) * 100
    
    return {
        'Vascular_Density_%': density,
        'Pixel_Accuracy_%': accuracy * 100,
        'Precision_%': precision * 100,
        'Recall_%': recall * 100,      
        'F1_Score': f1_score,          
        'IoU_Score': iou
    }

def run_engine():
    print("=== RetinaMetrics: Segmentation & Analytics Engine ===")
    ehr_df = parse_ehr_records()
    print(f"Cleaned {len(ehr_df)} EHR patient records -> 'outputs/stare_ehr_cohort.csv'")

    image_paths = sorted(glob.glob("data/images/*.ppm"))
    if not image_paths:
        print("Error: No .ppm images found in 'data/images/'.")
        return

    label_dir = "data/labels" if os.path.exists("data/labels") and os.listdir("data/labels") else "data/labels_ah"

    print(f"Processing {len(image_paths)} retinal scans...\n")
    results = []

    for img_path in image_paths:
        pid = os.path.basename(img_path).replace(".ppm", "")
        gt_path = os.path.join(label_dir, f"{pid}.ah.ppm")

        if not os.path.exists(gt_path):
            continue

        img = iio.imread(img_path)
        gt_mask = iio.imread(gt_path)

        _, _, _, pred_mask, fov_mask = segment_vessels(img)
        metrics = evaluate_scan(pred_mask, gt_mask, fov_mask)
        metrics["Patient_ID"] = pid
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    final_df = pd.merge(metrics_df, ehr_df, on="Patient_ID", how="left")
    final_df["Disease_Category"] = final_df["Disease_Category"].fillna("Other Retinal Pathology")
    final_df["Primary_Diagnosis"] = final_df["Primary_Diagnosis"].fillna("Clinical Fundus Scan")

    # FIXED: Replaced Dice_Score and Sensitivity_% with the new dictionary keys
    cols = [
        "Patient_ID", "Disease_Category", "Vascular_Density_%",
        "Pixel_Accuracy_%", "Precision_%", "Recall_%", "F1_Score", "IoU_Score", "Primary_Diagnosis"
    ]
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    final_df.to_csv("outputs/retinametrics_results.csv", index=False)

    print(final_df.drop(columns=["Primary_Diagnosis"]).to_string(index=False))
    print("\n--- Overall Performance Summary ---")
    
    # FIXED: Updated print statements to map perfectly to the new DataFrame columns
    print(f"Mean Pixel Accuracy : {final_df['Pixel_Accuracy_%'].mean():.2f}%")
    print(f"Mean Precision      : {final_df['Precision_%'].mean():.2f}%")
    print(f"Mean Recall         : {final_df['Recall_%'].mean():.2f}%")
    print(f"Mean F1/Dice Score  : {final_df['F1_Score'].mean():.4f}")
    print(f"Mean IoU Score      : {final_df['IoU_Score'].mean():.4f}")
    print(f"Mean Vessel Density : {final_df['Vascular_Density_%'].mean():.2f}%")
    print("Saved report to 'outputs/retinametrics_results.csv'")

if __name__ == "__main__":
    run_engine()