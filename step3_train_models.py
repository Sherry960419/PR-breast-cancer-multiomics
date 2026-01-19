"""
Step 3: Train and compare models using:
- gene-only features (200)
- cna-only features (200)
- combined gene + cna features (400)

We will train:
- Logistic Regression (L2)
- Lasso Logistic Regression (L1)
- Random Forest

We will evaluate using Accuracy and F1.
We will save:
- model_performance.csv  (all model scores)
- best_model.pkl         (the final selected model)

This step reads:
    multiomic_merged_200.csv
Output:
    best_model.pkl
    model_performance.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import OUT_DIR

def run_step():

    print("[Step3] Loading merged dataset...")
    df = pd.read_csv(OUT_DIR / "multiomic_merged_200.csv")

    print("[Step3] Original data shape:", df.shape)

    # ============================================================
    # 1. Prepare labels
    # ============================================================
    # Convert shape to binary: Irregular=1, Round-Oval=0
    df = df[df["SHAPE"].isin(["Irregular", "Round-Oval"])].copy()
    df["y"] = (df["SHAPE"] == "Irregular").astype(int)

    # ============================================================
    # 2. Build feature sets
    # ============================================================
    gene_cols = [c for c in df.columns if c in df.columns and not c.endswith("_cna") 
                 and c not in ["PATIENT_ID", "SHAPE", "y", "Age", "Subtype", "Stage Code"]]

    cna_cols  = [c for c in df.columns if c.endswith("_cna")]

    combined_cols = gene_cols + cna_cols

    print(f"[Step3] #Gene features: {len(gene_cols)}")
    print(f"[Step3] #CNA features:  {len(cna_cols)}")
    print(f"[Step3] #Combined:       {len(combined_cols)}")

    # Store all model results
    results = []

    def train_and_eval(X, y, model_name_prefix=""):
        """Fit three models on a given feature set and record scores."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        models = {
            model_name_prefix + "_Logistic": LogisticRegression(max_iter=2000),
            model_name_prefix + "_Lasso":    LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000),
            model_name_prefix + "_RF":       RandomForestClassifier(n_estimators=200, random_state=42),
        }

        local_scores = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            local_scores.append({
                "Model": name,
                "Accuracy": acc,
                "F1": f1
            })

            # Return model + score for potential best selection
            yield name, model, acc, f1

        return local_scores

    # ============================================================
    # 3. Train models for each feature group
    # ============================================================

    y = df["y"]

    feature_sets = {
        "Gene": df[gene_cols],
        "CNA": df[cna_cols],
        "Combined": df[combined_cols]
    }

    # Track the best model across all experiments
    best_model = None
    best_score = -1
    best_name  = ""

    print("[Step3] Training models...")

    for group_name, X in feature_sets.items():
        print(f"\n=== Training on feature group: {group_name} ===")

        for name, model, acc, f1 in train_and_eval(X, y, model_name_prefix=group_name):
            print(f"{name}: Acc={acc:.3f}, F1={f1:.3f}")

            results.append({"Model": name, "Accuracy": acc, "F1": f1})

            # Track best model by F1
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name  = name

    # ============================================================
    # 4. Save model performance table
    # ============================================================
    perf_path = OUT_DIR / "model_performance.csv"
    pd.DataFrame(results).to_csv(perf_path, index=False)

    # ============================================================
    # 5. Save the best model overall
    # ============================================================
    model_path = OUT_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    print("\n============================")
    print("Best model:", best_name)
    print(f"Best F1 score = {best_score:.3f}")
    print("Saved to:", model_path)
    print("============================")

    return model_path


if __name__ == "__main__":
    run_step()
