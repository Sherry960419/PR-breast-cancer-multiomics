"""
Step 8 (200-gene version): Generate per-patient radiogenomic markdown reports.

Inputs:
  - llm_inputs.json           (from step7_build_prompt_200.py)
  - multiomic_merged_200.csv  (numeric gene/CNA values, including Age/Subtype/Stage Code)
  - feature_means_200.csv     (cohort means for each feature)

For each patient:
  - Select top over-expressed / under-expressed genes (by deviation from mean)
  - Select strongest CNA amplifications / losses
  - Build a structured prompt
  - Call Gemini to generate a markdown report
  - Save as OUT_DIR / "{PATIENT_ID}.md"
"""

import os
import json
from datetime import date

import pandas as pd
import google.generativeai as genai

from config import OUT_DIR

# ------------- Configuration -------------

# If True, only generate for the first N patients in llm_inputs.json
TEST_MODE = True
TEST_PATIENT_LIMIT = 1

# Gemini model name
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Number of items to show per category
TOP_N_GENES_OVER = 10
TOP_N_GENES_UNDER = 10
TOP_N_CNA_AMP = 10
TOP_N_CNA_LOSS = 10

# ------------- PROMPTS -------------

PROMPT_SYSTEM = """
You are a medical AI assistant generating a concise, academic-style
radiogenomic report for a breast cancer case.

You will be given:
- Basic clinical information (age, molecular subtype, stage code)
- Tumor shape information (radiologist label when available, otherwise model-derived)
- A small set of genes that are strongly over- or under-expressed
- A small set of genes with copy-number amplification or loss (CNA)

Your task is to integrate the clinical context with these molecular deviations
and write a structured narrative report in clear English.
"""

PROMPT_INSTRUCTIONS = """
Write the report with the following sections, using Markdown headings.
Do NOT add your own title; start directly from '## 1. Clinical Context'.

## 1. Clinical Context
- Summarize age, subtype, and stage code.
- Mention the tumor shape label (Irregular vs Round-Oval) and, if relevant,
  briefly comment on what this shape can mean conceptually in breast imaging.
- Comment briefly on typical prognosis or clinical implications for this subtype
  and stage (conceptually, not numerically).

## 2. Radiogenomic / Molecular Interpretation
- Summarize how the over-expressed genes and under-expressed genes might relate
  to tumor biology (e.g., proliferation, hormone signaling, immune response,
  invasion, EMT), when such relationships are reasonably known.
- Summarize how CNA amplifications and CNA losses might conceptually influence
  tumor behavior (e.g., oncogene activation, tumor suppressor loss).
- If you are not sure about a specific gene, acknowledge uncertainty and
  keep the discussion high-level instead of fabricating details.

## 3. Suggested Next Clinical Steps (Conceptual)
- Based on subtype, stage, age, tumor shape, and the pattern of molecular
  alterations, discuss the conceptual level of risk
  (e.g., relatively favorable vs potentially higher risk), without giving
  numerical risk estimates.
- Suggest reasonable, generic next steps in abstract terms
  (e.g., endocrine therapy, chemotherapy, imaging follow-up, multidisciplinary
  review), but do NOT give patient-specific medical advice, drug names,
  doses, or regimens.

## 4. Limitations
- Emphasize that this is an exploratory research report based on a limited panel
  of genes and a small dataset.
- State clearly that the report cannot replace clinical judgement, pathology,
  or guideline-based management.

Style requirements:
- Do NOT add a separate H1 title (like '# Radiogenomic Report').
- Do NOT simply copy the raw bullet lists of genes;
  instead, synthesize them into a narrative.
- Do NOT invent exact numerical risks, survival percentages, or regimen details.
- Keep the tone academic, neutral, and concise.
- Write in clear English prose.
- Output MUST be valid markdown, but do NOT use underline syntax
  (no '====' or '----' underline headers).
- Do NOT repeat the patient header (ID, Age, Subtype, Stage, Tumor Shape, Date)
  inside the sections; assume the reader has just seen it.
"""

# ------------- Helper Functions -------------


def load_llm_items():
    """Load the JSON list of patient items produced by step7."""
    json_path = OUT_DIR / "llm_inputs.json"
    print(f"[Step8] Reading LLM inputs from: {json_path}")
    with open(json_path, "r") as f:
        items = json.load(f)
    return items


def load_numeric_tables():
    """
    Load numeric multi-omic table and feature means.
    We use these to compute deviations (patient - cohort mean)
    and pick the strongest genes/CNAs.
    """
    merged_path = OUT_DIR / "multiomic_merged_200.csv"
    means_path = OUT_DIR / "feature_means_200.csv"

    df_num = pd.read_csv(merged_path)
    means_df = pd.read_csv(means_path)
    mean_dict = dict(zip(means_df["feature"], means_df["mean_value"]))

    # Index by PATIENT_ID for easier row lookup
    df_num = df_num.set_index("PATIENT_ID")

    return df_num, mean_dict


def select_top_by_deviation(
    row, mean_dict, feature_names, status_dict, desired_status, top_k
):
    """
    From the given feature list, select features that have the specified
    over/under status in status_dict, and rank them by |value - mean|.
    """
    cand = []
    for gene in feature_names:
        # For CNA features, status_dict keys are base gene names (without "_cna")
        if gene.endswith("_cna"):
            base = gene[:-4]
        else:
            base = gene

        if status_dict.get(base) != desired_status:
            continue

        # Safe numeric lookup
        if gene not in row.index:
            continue
        if gene not in mean_dict:
            continue

        diff = float(row[gene]) - float(mean_dict[gene])
        cand.append((gene, abs(diff)))

    cand.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in cand[:top_k]]
    return selected


def build_header_block(patient_id, age, subtype, stage_code, true_shape, predicted_shape):
    """
    Build the fixed text header for the report, including fallback logic for tumor shape.
    """
    # Decide which shape to display
    if true_shape and str(true_shape).lower() not in ["na", "not applicable", "nan"]:
        shape_text = str(true_shape)
    else:
        shape_text = f"{predicted_shape} (model-predicted; radiologist label unavailable)"

    today_str = date.today().isoformat()

    # markdown header
    header = (
        "# Radiogenomic Report\n\n"
        f"**Patient ID:** {patient_id}\n"
        f"**Age:** {age}\n"
        f"**Subtype:** {subtype}\n"
        f"**Stage Code:** {stage_code}\n"
        f"**Tumor Shape:** {shape_text}\n"
        f"**Date of Report:** {today_str}\n\n"
        "---\n\n"
    )
    return header


def build_patient_context_text(
    age,
    subtype,
    stage_code,
    true_shape,
    predicted_shape,
    prob_irregular,
    gene_over,
    gene_under,
    cna_amp,
    cna_loss,
):
    """
    Build the textual context block that will be sent to the LLM,
    containing all relevant patient-level information.
    """

    if true_shape and str(true_shape).lower() not in ["na", "not applicable", "nan"]:
        shape_line = f"Radiologist tumor shape label: {true_shape}."
    else:
        shape_line = (
            f"Radiologist tumor shape unavailable. "
            f"Model-derived tumor shape: {predicted_shape} "
            f"(probability of Irregular = {prob_irregular:.3f})."
        )

    def fmt_list(lst):
        return ", ".join(lst) if lst else "None"

    text = f"""
[Clinical Summary]
- Age: {age}
- Molecular subtype: {subtype}
- Stage Code: {stage_code}
- Tumor shape information: {shape_line}

[Molecular Highlights]
- Over-expressed genes (top-ranked): {fmt_list(gene_over)}
- Under-expressed genes (top-ranked): {fmt_list(gene_under)}
- Amplified CNAs (top-ranked): {fmt_list(cna_amp)}
- CNA losses (top-ranked): {fmt_list(cna_loss)}

Use this information to ground your radiogenomic interpretation.
"""
    return text


# ------------- Main Step Function -------------


def run_step():
    # 1) Load LLM items (from step7)
    items = load_llm_items()

    if TEST_MODE:
        print(f"[Step8] TEST_MODE → using first {TEST_PATIENT_LIMIT} patient(s).")
        items = items[:TEST_PATIENT_LIMIT]

    # 2) Load numeric tables (for deviations)
    df_num, mean_dict = load_numeric_tables()

    # 3) Configure Gemini
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # 4) Loop over patients and generate one .md file per patient
    for item in items:
        patient_id = item["patient_id"]
        age = item["age"]
        subtype = item["subtype"]
        stage_code = item["stage_code"]
        true_shape = item["true_shape"]
        predicted_shape = item["predicted_shape"]
        prob_irregular = item["prob_irregular"]

        gene_status = item["gene_overunder"]  # dict: gene -> "over"/"under"
        cna_status = item["cna_overunder"]    # dict: gene -> "over"/"under"

        print(f"[Step8] Generating report for patient: {patient_id}")

        if patient_id not in df_num.index:
            print(f"[Step8] WARNING: {patient_id} not found in numeric table. Skipping.")
            continue
        row = df_num.loc[patient_id]

        # Feature lists from numeric table
        gene_cols = [
            c for c in df_num.columns
            if (not c.endswith("_cna"))
            and c not in ["PATIENT_ID", "SHAPE", "Age", "Subtype", "Stage Code"]
        ]
        cna_cols = [c for c in df_num.columns if c.endswith("_cna")]

        # Select top deviations
        gene_over = select_top_by_deviation(
            row, mean_dict, gene_cols, gene_status, desired_status="over", top_k=TOP_N_GENES_OVER
        )
        gene_under = select_top_by_deviation(
            row, mean_dict, gene_cols, gene_status, desired_status="under", top_k=TOP_N_GENES_UNDER
        )
        cna_amp = select_top_by_deviation(
            row, mean_dict, cna_cols, cna_status, desired_status="over", top_k=TOP_N_CNA_AMP
        )
        cna_loss = select_top_by_deviation(
            row, mean_dict, cna_cols, cna_status, desired_status="under", top_k=TOP_N_CNA_LOSS
        )

        print(f"[Step8] #Gene numeric features: {len(gene_cols)}")
        print(f"[Step8] #CNA  numeric features: {len(cna_cols)}")
        print(f"[Step8] Selected {len(gene_over)} over-expressed genes, {len(gene_under)} under-expressed genes.")
        print(f"[Step8] Selected {len(cna_amp)} CNA amplifications, {len(cna_loss)} CNA losses.")

        # Header
        header_text = build_header_block(
            patient_id=patient_id,
            age=age,
            subtype=subtype,
            stage_code=stage_code,
            true_shape=true_shape,
            predicted_shape=predicted_shape,
        )

        # Patient context for LLM
        context_text = build_patient_context_text(
            age=age,
            subtype=subtype,
            stage_code=stage_code,
            true_shape=true_shape,
            predicted_shape=predicted_shape,
            prob_irregular=prob_irregular,
            gene_over=gene_over,
            gene_under=gene_under,
            cna_amp=cna_amp,
            cna_loss=cna_loss,
        )

        # Final prompt
        full_prompt = (
            PROMPT_SYSTEM.strip()
            + "\n\n"
            + PROMPT_INSTRUCTIONS.strip()
            + "\n\n===== PATIENT-SPECIFIC DATA =====\n"
            + context_text
        )

        print("[Step8] Sending request to Gemini...")
        response = model.generate_content(full_prompt)
        body_markdown = response.text

        # Combine header + body
        file_text = header_text + body_markdown

        out_path = OUT_DIR / f"{patient_id}.md"
        with open(out_path, "w") as f:
            f.write(file_text)

        print(f"[Step8] Saved report for {patient_id} to: {out_path}")

    print("[Step8] Done.")
    return True


if __name__ == "__main__":
    run_step()
