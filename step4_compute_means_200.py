"""
Step 4: Compute cohort-level means for all 200 expression + 200 CNA features.

Input :
  - multiomic_merged_200.csv (expr_200 + cna_200 + clinical + SHAPE)
Output:
  - feature_means_200.csv  (columns: feature, mean_value)
"""

import pandas as pd
from config import OUT_DIR

def run_step():
    print("[Step4] Loading merged dataset...")
    df = pd.read_csv(OUT_DIR / "multiomic_merged_200.csv")
    print("[Step4] Data shape:", df.shape)
    print("[Step4] Columns (tail):", df.columns[-10:].tolist())

    # Meta columns: not used as numeric features
    # IMPORTANT: use "Stage Code" (with space), not "Stage_Code"
    meta_cols = ["PATIENT_ID", "SHAPE", "Age", "Subtype", "Stage Code"]

    # Expression genes: not meta, and not CNA
    gene_cols = [
        c for c in df.columns
        if c not in meta_cols and not c.endswith("_cna")
    ]
    # CNA genes
    cna_cols = [c for c in df.columns if c.endswith("_cna")]

    features = gene_cols + cna_cols

    print(f"[Step4] #Gene features: {len(gene_cols)}")
    print(f"[Step4] #CNA  features: {len(cna_cols)}")

    # Compute mean per feature over all patients (including Not Applicable shapes)
    means = df[features].astype(float).mean()

    out_df = means.reset_index()
    out_df.columns = ["feature", "mean_value"]

    out_path = OUT_DIR / "feature_means_200.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[Step4] Saved feature means to: {out_path}")
    return out_path

if __name__ == "__main__":
    run_step()
