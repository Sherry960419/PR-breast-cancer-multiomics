"""
Step 5: For each patient, classify each feature as over / under / neutral
relative to cohort means.

Inputs:
  - multiomic_merged_200.csv
  - feature_means_200.csv
Output:
  - over_under_summary_200.csv
"""

import pandas as pd
from config import OUT_DIR

def classify_diff(value, mean, threshold=0.1):
    """Classify over / under / neutral based on deviation from mean."""
    if pd.isna(value):
        return "NA"
    diff = value - mean
    if diff > threshold:
        return "over"
    elif diff < -threshold:
        return "under"
    else:
        return "neutral"

def run_step():
    print("[Step5] Loading data...")
    df = pd.read_csv(OUT_DIR / "multiomic_merged_200.csv")
    means_df = pd.read_csv(OUT_DIR / "feature_means_200.csv")

    mean_dict = dict(zip(means_df["feature"], means_df["mean_value"]))

    # IMPORTANT: use "Stage Code" here
    meta_cols = ["PATIENT_ID", "SHAPE", "Age", "Subtype", "Stage Code"]

    gene_cols = [
        c for c in df.columns
        if c not in meta_cols and not c.endswith("_cna")
    ]
    cna_cols = [c for c in df.columns if c.endswith("_cna")]
    features = gene_cols + cna_cols

    records = []
    for _, row in df.iterrows():
        rec = {
            "PATIENT_ID": row["PATIENT_ID"],
            "SHAPE": row["SHAPE"],
            "Age": row["Age"],
            "Subtype": row["Subtype"],
            "Stage Code": row["Stage Code"],
        }
        for f in features:
            rec[f] = classify_diff(row[f], mean_dict[f])
        records.append(rec)

    out_df = pd.DataFrame(records)
    out_path = OUT_DIR / "over_under_summary_200.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[Step5] Saved over/under table to: {out_path}")
    print("[Step5] Shape:", out_df.shape)
    return out_path

if __name__ == "__main__":
    run_step()
