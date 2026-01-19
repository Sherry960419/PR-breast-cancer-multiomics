"""
Preprocess Step 2:
Merge top-200 expression, top-200 CNA, SHAPE labels, and clinical metadata.

This script performs:
1. Load expr_top200.csv and cna_top200.csv
2. Load radiologist SHAPE labels
3. Load clinical metadata (age, subtype, AJCC stages)
4. Create a single combined 'Stage Code' column
5. Merge all tables into one large multi-omics + clinical dataset
6. Do NOT filter out 'Not Applicable' at this stage
7. Save multiomic_merged_200.csv
"""

import pandas as pd
from config import DATA_DIR, OUT_DIR

def run_step():
    print("[Preprocess Step2] Loading files...")

    # ================================
    # 1. Load datasets
    # ================================
    expr = pd.read_csv(OUT_DIR / "expr_top200.csv")
    cna  = pd.read_csv(OUT_DIR / "cna_top200.csv")
    shape_df = pd.read_csv(DATA_DIR / "tcga-breast-radiologist-reads_unique_shape.csv")

    clinical = pd.read_csv(
        DATA_DIR / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv",
        sep="\t"
    )

    print("[Preprocess Step2] Shapes:")
    print("expr    :", expr.shape)
    print("cna     :", cna.shape)
    print("shape   :", shape_df.shape)
    print("clinical:", clinical.shape)

    # ================================
    # 2. Keep only the needed clinical columns
    # ================================
    # These are the TRUE column names from the TSV
    col_T = "American Joint Committee on Cancer Tumor Stage Code"
    col_N = "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code"
    col_M = "American Joint Committee on Cancer Metastasis Stage Code"

    clinical_sub = clinical[
        [
            "Patient ID",      # Matches PATIENT_ID after ID fixing
            "Diagnosis Age",   # Will rename to Age
            "Subtype",
            col_T,
            col_N,
            col_M,
        ]
    ].copy()

    # ================================
    # 3. Rename columns for cleaner usage
    # ================================
    clinical_sub = clinical_sub.rename(
        columns={
            "Diagnosis Age": "Age",            # rename → Age
            col_T: "AJCC_T_Stage_Code",
            col_N: "AJCC_N_Stage_Code",
            col_M: "AJCC_M_Stage_Code",
        }
    )

    # ================================
    # 4. Create a single unified Stage Code column
    # ================================
    clinical_sub["Stage Code"] = (
        clinical_sub["AJCC_T_Stage_Code"].fillna("") + " " +
        clinical_sub["AJCC_N_Stage_Code"].fillna("") + " " +
        clinical_sub["AJCC_M_Stage_Code"].fillna("")
    ).str.strip()

    # Normalize multiple spaces to a single space
    clinical_sub["Stage Code"] = clinical_sub["Stage Code"].str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    # Only keep the final simplified columns
    clinical_sub = clinical_sub[["Patient ID", "Age", "Subtype", "Stage Code"]]

    print("[Preprocess Step2] Clinical subset shape:", clinical_sub.shape)
    print("[Preprocess Step2] Clinical columns:", clinical_sub.columns.tolist())

    # ================================
    # 5. Merge expression + CNA
    # ================================
    merged = expr.merge(cna, on="PATIENT_ID")
    print("[Preprocess Step2] After merging expr + cna:", merged.shape)

    # ================================
    # 6. Merge SHAPE labels
    # ================================
    merged2 = merged.merge(
        shape_df[["PATIENT_ID", "SHAPE"]],
        on="PATIENT_ID",
        how="inner",
    )
    print("[Preprocess Step2] After merging with SHAPE:", merged2.shape)

    # ================================
    # 7. Merge clinical metadata
    # ================================
    merged3 = merged2.merge(
        clinical_sub,
        left_on="PATIENT_ID",
        right_on="Patient ID",
        how="inner",
    )
    print("[Preprocess Step2] After merging with clinical:", merged3.shape)

    # Remove duplicate identifier column
    merged3 = merged3.drop(columns=["Patient ID"])

    # ================================
    # 8. Save final merged dataset
    # ================================
    out_path = OUT_DIR / "multiomic_merged_200.csv"
    merged3.to_csv(out_path, index=False)

    print(f"[Preprocess Step2] Saved merged dataset to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_step()
