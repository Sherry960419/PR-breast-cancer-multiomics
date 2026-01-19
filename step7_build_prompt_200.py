"""
Step 7: Build JSON inputs for LLM (Gemini).

Inputs:
  - predictions_200.csv    (CNA RF predictions + Age, Subtype, Stage Code)
  - over_under_summary_200.csv  (over/under status for genes & CNAs)

Output:
  - llm_inputs.json

Each JSON item (per patient) contains:
  - patient_id
  - age
  - subtype
  - stage_code
  - true_shape
  - predicted_shape
  - prob_irregular
  - correct
  - gene_overunder: { gene -> "over"/"under" }
  - cna_overunder:  { gene -> "over"/"under" }
"""

import json
import pandas as pd
from config import OUT_DIR

def run_step():
    print("[Step7] Loading predictions and over/under tables...")
    preds_path = OUT_DIR / "predictions_200.csv"
    over_path  = OUT_DIR / "over_under_summary_200.csv"

    preds = pd.read_csv(preds_path)
    over  = pd.read_csv(over_path)

    print("[Step7] preds columns (tail):", preds.columns[-10:].tolist())
    print("[Step7] over  columns (tail):",  over.columns[-10:].tolist())

    # Merge by PATIENT_ID
    merged = preds.merge(over, on="PATIENT_ID", suffixes=("", "_status"))
    print("[Step7] Merged shape:", merged.shape)

    # Identify gene/CNA status columns coming from over_under table
    # We assume over has: PATIENT_ID, SHAPE, Age, Subtype, Stage Code, plus 200 genes + 200 CNA
    meta_cols = {"PATIENT_ID", "SHAPE", "Age", "Subtype", "Stage Code"}

    gene_status_cols = [
        c for c in over.columns
        if (c not in meta_cols) and (not c.endswith("_cna"))
    ]
    cna_status_cols = [c for c in over.columns if c.endswith("_cna")]

    # Derive base gene names from CNA status columns (for consistent keys)
    base_genes = [c[:-4] for c in cna_status_cols]  # remove "_cna"

    items = []
    for _, row in merged.iterrows():
        gene_status = {}
        cna_status = {}

        # Loop over base gene names
        for g in base_genes:
            # gene expression over/under
            if g in merged.columns:
                status_g = row[g]
                if status_g in ("over", "under"):
                    gene_status[g] = status_g

            # CNA over/under
            cna_col = f"{g}_cna"
            if cna_col in merged.columns:
                status_c = row[cna_col]
                if status_c in ("over", "under"):
                    cna_status[g] = status_c

        item = {
            "patient_id": row["PATIENT_ID"],
            "age": row["Age"],
            "subtype": row["Subtype"],
            # IMPORTANT: use exact column name "Stage Code" from step2
            "stage_code": row["Stage Code"],
            "true_shape": row["true_SHAPE"],
            "predicted_shape": row["predicted_SHAPE"],
            "prob_irregular": float(row["prob_Irregular"]),
            "correct": int(row["correct"]),
            "gene_overunder": gene_status,
            "cna_overunder": cna_status,
        }
        items.append(item)

    out_path = OUT_DIR / "llm_inputs.json"  # keep same name for step8
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"[Step7] Saved LLM input JSON to: {out_path}")
    print(f"[Step7] Number of patients in JSON: {len(items)}")
    return out_path


if __name__ == "__main__":
    run_step()
