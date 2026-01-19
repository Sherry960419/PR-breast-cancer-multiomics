"""
Step 9 (200-feature version): Orchestrate the full pipeline with LangChain.

Pipeline:

  [Preprocess]
    1) preprocess_step1_select_200_features.run_step()
       - From raw data11Select.csv / data22Select.csv
       - Select top-200 variable genes and build expr_top200.csv & cna_top200.csv

    2) preprocess_step2_merge_multiomic_clinical.run_step()
       - Merge expr_top200 + cna_top200 + SHAPE + clinical (Age, Subtype, Stage Code)
       - Save multiomic_merged_200.csv

  [Model & Radiogenomics]
    3) step3_train_models_200.run_step()
       - Compare Gene / CNA / Combined models
       - Save best model (CNA_RF) to best_model_200.pkl

    4) step4_compute_means_200.run_step()
       - Compute cohort mean of 200 gene + 200 CNA
       - Save feature_means_200.csv

    5) step5_compute_overunder_200.run_step()
       - For each patient, classify each feature as over / under / neutral
       - Save over_under_summary_200.csv

    6) step6_predict_shape_200.run_step()
       - Use best CNA RF model to predict tumor shape
       - Save predictions_200.csv

    7) step7_build_prompt_200.run_step()
       - Merge predictions + over/under + clinical
       - Build llm_inputs.json (one JSON object per patient)

    8) step8_generate_report_200.run_step()
       - For selected patients (TEST_MODE controls how many)
       - Build prompt, call Gemini for each patient
       - Save one Markdown report per patient: reports/{PATIENT_ID}.md
"""

from langchain_core.runnables import RunnableLambda, RunnableSequence

import preprocess_step1_select_200_features as p1
import preprocess_step2_merge_multiomic_clinical as p2
import step3_train_models as s3
import step4_compute_means_200 as s4
import step5_compute_overunder_200 as s5
import step6_predict_shape_200 as s6
import step7_build_prompt_200 as s7
import step8_generate_report_200 as s8


def as_runnable(func, name: str) -> RunnableLambda:
    """Wrap a no-arg function into a LangChain Runnable with logging."""
    def inner(_input=None):
        print(f"\n===== Running {name} =====")
        result = func()
        return result
    return RunnableLambda(inner)


# Build the chain (purely sequential)
chain = RunnableSequence(
    as_runnable(p1.run_step, "Preprocess Step1: select 200 features"),
    as_runnable(p2.run_step, "Preprocess Step2: merge multiomic + clinical"),
    as_runnable(s3.run_step, "Step3: train models (200 features)"),
    as_runnable(s4.run_step, "Step4: compute feature means (200)"),
    as_runnable(s5.run_step, "Step5: compute over/under (200)"),
    as_runnable(s6.run_step, "Step6: predict shape (CNA RF)"),
    as_runnable(s7.run_step, "Step7: build LLM inputs JSON"),
    as_runnable(s8.run_step, "Step8: generate per-patient reports"),
)

def run_pipeline():
    """Entry point to run the whole pipeline via LangChain."""
    final_output = chain.invoke(None)
    print("\n===== Workflow finished (200-feature version) =====")
    print("Last step output:", final_output)
    return final_output


if __name__ == "__main__":
    run_pipeline()
