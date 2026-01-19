"""
Preprocess Step 1
- Load raw expression (data22Select.csv) and CNA/CNV (data11Select.csv)
- Fix SAMPLE_ID → PATIENT_ID (TCGA.3C.AAAU -> TCGA-3C-AAAU)
- Find top-200 genes by variance in expression
- Save:
    - outputs/expr_top200.csv
    - outputs/cna_top200.csv
"""

import pandas as pd
import numpy as np
from config import DATA_DIR, OUT_DIR

def fix_id(x: str) -> str:
    """Convert TCGA.3C.AAAU -> TCGA-3C-AAAU."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace(".", "-")
    return x

def run_step():
    # ========== Step 0: Load raw expression & CNA tables ==========
    # NOTE: use index_col=0 because the first column is SAMPLE_ID (e.g. TCGA.3C.AAAU)
    expr_raw = pd.read_csv(DATA_DIR / "data22Select.csv", index_col=0)  # gene expression
    cna_raw  = pd.read_csv(DATA_DIR / "data11Select.csv", index_col=0)  # CNV/CNA

    print("expr_raw shape:", expr_raw.shape)
    print("cna_raw shape:",  cna_raw.shape)
    print("expr_raw index (first 5):", expr_raw.index[:5].tolist())
    print("expr_raw columns (first 10):", expr_raw.columns[:10].tolist())

    # ========== Step 1: Fix ID format & create PATIENT_ID ==========
    # Make a working copy
    expr = expr_raw.copy()
    cna  = cna_raw.copy()

    # Index is SAMPLE_ID like 'TCGA.3C.AAAU'
    expr.index = expr.index.map(fix_id)
    cna.index  = cna.index.map(fix_id)

    # Add PATIENT_ID column from index
    expr["PATIENT_ID"] = expr.index
    cna["PATIENT_ID"]  = cna.index

    print("\nExpr head after ID fix:")
    print(expr[["PATIENT_ID"]].head())
    print("\nCNA head after ID fix:")
    print(cna[["PATIENT_ID"]].head())

    # ========== Step 2: Select shared gene features & compute variance ==========
    # Drop PATIENT_ID for variance computation
    expr_features = expr.drop(columns=["PATIENT_ID"])
    cna_features  = cna.drop(columns=["PATIENT_ID"])

    # Keep only columns shared by both tables
    shared_genes = sorted(set(expr_features.columns) & set(cna_features.columns))
    print(f"\nNumber of shared gene features: {len(shared_genes)}")

    expr_shared = expr_features[shared_genes]
    cna_shared  = cna_features[shared_genes]

    # Variance of expression across patients (used for ranking)
    variances = expr_shared.var(axis=0)
    variances_sorted = variances.sort_values(ascending=False)

    TOP_K = 200
    top_genes = variances_sorted.head(TOP_K).index.tolist()
    print(f"Selected top {TOP_K} genes by variance.")

    # ========== Step 3: Build top-200 expression & CNA tables ==========
    expr_top = expr_shared[top_genes].copy()
    cna_top  = cna_shared[top_genes].copy()

    # Add PATIENT_ID as the first column
    expr_top.insert(0, "PATIENT_ID", expr["PATIENT_ID"])
    cna_top.insert(0, "PATIENT_ID",  cna["PATIENT_ID"])

    # For CNA, rename columns with _cna suffix (except PATIENT_ID)
    new_cna_cols = ["PATIENT_ID"] + [g + "_cna" for g in top_genes]
    cna_top.columns = new_cna_cols

    print("\nexpr_top shape:", expr_top.shape)
    print("cna_top shape:", cna_top.shape)
    print("expr_top columns (first 10):", expr_top.columns[:10].tolist())
    print("cna_top columns (first 10):",  cna_top.columns[:10].tolist())

    # ========== Step 4: Save to OUT_DIR ==========
    expr_out_path = OUT_DIR / "expr_top200.csv"
    cna_out_path  = OUT_DIR / "cna_top200.csv"

    expr_top.to_csv(expr_out_path, index=False)
    cna_top.to_csv(cna_out_path, index=False)

    print(f"\nSaved expr_top200 to: {expr_out_path}")
    print(f"Saved cna_top200  to: {cna_out_path}")

    return expr_out_path, cna_out_path


if __name__ == "__main__":
    run_step()
