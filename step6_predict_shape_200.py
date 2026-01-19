"""
Step 6: Use the trained best_model (CNA RandomForest) to predict tumor shape
for all patients in multiomic_merged_200.csv.

Input:
  - multiomic_merged_200.csv
  - best_model.pkl   (trained on CNA-only features)

Output:
  - predictions_200.csv
      PATIENT_ID, Age, Subtype, Stage Code,
      true_SHAPE, predicted_SHAPE, prob_Irregular, correct
"""

import pandas as pd
import joblib
from config import OUT_DIR

def run_step():
    print("[Step6] Loading data and model...")
    df = pd.read_csv(OUT_DIR / "multiomic_merged_200.csv")
    model = joblib.load(OUT_DIR / "best_model.pkl")

    # Meta columns (not used as features)
    meta_cols = ["PATIENT_ID", "SHAPE", "Age", "Subtype", "Stage Code"]

    # Use CNA-only features for prediction (same as in training best model)
    cna_cols = [c for c in df.columns if c.endswith("_cna")]

    print(f"[Step6] #CNA features used for prediction: {len(cna_cols)}")

    # X as DataFrame so sklearn can align by column name
    X = df[cna_cols].astype(float)

    # Predict probabilities and class labels
    prob_Irregular = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    pred_shape = ["Irregular" if p == 1 else "Round-Oval" for p in y_pred]

    # Build output table
    out_df = pd.DataFrame({
        "PATIENT_ID": df["PATIENT_ID"],
        "Age": df["Age"],
        "Subtype": df["Subtype"],
        "Stage Code": df["Stage Code"],   # <- important: use exactly this name
        "true_SHAPE": df["SHAPE"],
        "predicted_SHAPE": pred_shape,
        "prob_Irregular": prob_Irregular,
    })
    out_df["correct"] = (out_df["true_SHAPE"] == out_df["predicted_SHAPE"]).astype(int)

    out_path = OUT_DIR / "predictions_200.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[Step6] Saved predictions to: {out_path}")
    print(out_df.head())
    return out_path


if __name__ == "__main__":
    run_step()
