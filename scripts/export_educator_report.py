"""Export a flat, educator-readable table combining ground-truth marking,
model predictions, and XAI explanations for one CCU.

Produces one row per response with plain-English columns — no JSON nesting,
no code required to read it — suitable for opening directly in Excel/Sheets,
handing to an educator, or feeding into a future dashboard/UI as a data
source.

Combines three sources:
  1. The marked dataset (data/raw/Monica_Data_CCUS_converted.csv) — ground
     truth validity/confidence from manual marking, plus the MCQ answer.
  2. Known correct MCQ answers per CCU — to compute whether the student's
     MCQ selection was itself correct.
  3. The XAI explanation outputs from scripts/explain.py (LIME, SHAP,
     attention, evaluation) — the model's prediction and its explanation
     for that prediction.

Flags two independent kinds of disagreement, which mean different things
educationally:
  - mcq_vs_reasoning_mismatch: the STUDENT's own MCQ answer disagrees with
    the STUDENT's own marked reasoning validity (e.g. picked the right
    option but wrote invalid reasoning — likely a guess; or picked the
    wrong option but wrote valid reasoning — likely a slip, not a
    misconception).
  - model_vs_groundtruth_mismatch: the MODEL's prediction disagrees with
    the human-marked ground truth — a model error worth auditing.

Usage
-----
    python scripts/export_educator_report.py \\
        --csv data/raw/Monica_Data_CCUS_converted.csv \\
        --explanations-dir outputs/explanations/Monica_Data_CCUS_converted_ccu1_ensemble \\
        --ccu ccu1 \\
        --output outputs/educator_reports/ccu1_report.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Correct MCQ answer per CCU, established during dataset analysis.
CORRECT_MCQ = {
    "ccu1": "a", "ccu2": "c", "ccu3": "b",
    "ccu4": "b", "ccu5": "d", "ccu6": "b",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export an educator-readable XAI report.")
    p.add_argument("--csv", required=True, help="The converted marked CSV")
    p.add_argument("--explanations-dir", required=True,
                    help="Directory containing lime_explanations.json etc. from explain.py")
    p.add_argument("--ccu", required=True, help="Which CCU, e.g. ccu1")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--top-k", type=int, default=3,
                    help="How many top supporting/opposing words to include per response")
    return p.parse_args(argv)


def top_words(feature_weights: list, k: int, positive: bool) -> str:
    """Extract the top-k words in one direction from a LIME feature_weights list."""
    pairs = [(w, s) for w, s in feature_weights]
    pairs = [p for p in pairs if (p[1] >= 0) == positive]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return ", ".join(w for w, _ in pairs[:k]) if pairs else ""


def main(argv=None):
    args = parse_args(argv)
    exp_dir = Path(args.explanations_dir)

    # ---- Ground truth + MCQ correctness -----------------------------
    df = pd.read_csv(args.csv)
    df = df[df["ccuname"] == args.ccu].copy()
    correct_answer = CORRECT_MCQ.get(args.ccu)
    df["mcq_correct"] = df["q1mcr"].astype(str).str.strip().str.lower() == correct_answer
    df["mcq_vs_reasoning_mismatch"] = (
        (df["mcq_correct"]) != (df["validity"] == "correct")
    )

    # ---- Load explanation outputs ------------------------------------
    lime_path = exp_dir / "lime_explanations.json"
    if not lime_path.exists():
        raise FileNotFoundError(
            f"No lime_explanations.json found in {exp_dir}. "
            f"Run scripts/explain.py first."
        )
    lime_data = json.loads(lime_path.read_text())

    shap_path = exp_dir / "shap_explanations.json"
    shap_data = json.loads(shap_path.read_text()) if shap_path.exists() else []

    # explain.py samples the same texts in the same order for every
    # technique in one run, so LIME/SHAP entries line up by index.
    shap_by_text = {e["text"]: e for e in shap_data}

    # ---- Build the flattened rows -------------------------------------
    rows = []
    unmatched = 0
    for entry in lime_data:
        text = entry["text"]
        gt_row = df[df["q1txr"].astype(str).str.strip() == text.strip()]
        if gt_row.empty:
            unmatched += 1
            gt = {"q1mcr": None, "mcq_correct": None, "validity": None,
                  "confidence": None, "mcq_vs_reasoning_mismatch": None}
        else:
            gt = gt_row.iloc[0].to_dict()

        model_pred = entry["predicted_class_name"]
        ground_truth_validity = gt.get("validity")
        model_vs_groundtruth_mismatch = (
            model_pred != ground_truth_validity
            if ground_truth_validity is not None else None
        )

        row = {
            "ccu": args.ccu,
            "student_response": text,
            "mcq_answer": gt.get("q1mcr"),
            "mcq_correct": gt.get("mcq_correct"),
            "marked_validity": ground_truth_validity,
            "marked_confidence": gt.get("confidence"),
            "mcq_vs_reasoning_mismatch": gt.get("mcq_vs_reasoning_mismatch"),
            "model_predicted_validity": model_pred,
            "model_confidence_score": round(max(entry["predicted_proba"]), 3),
            "model_vs_groundtruth_mismatch": model_vs_groundtruth_mismatch,
            "lime_top_supporting_words": top_words(
                entry["feature_weights"], args.top_k, positive=True),
            "lime_top_opposing_words": top_words(
                entry["feature_weights"], args.top_k, positive=False),
        }

        shap_entry = shap_by_text.get(text)
        if shap_entry:
            shap_pairs = list(zip(shap_entry["tokens"], shap_entry["shap_values"]))
            pos = sorted([p for p in shap_pairs if p[1] >= 0],
                        key=lambda p: -abs(p[1]))[: args.top_k]
            neg = sorted([p for p in shap_pairs if p[1] < 0],
                        key=lambda p: -abs(p[1]))[: args.top_k]
            row["shap_top_supporting_words"] = ", ".join(w.strip() for w, _ in pos)
            row["shap_top_opposing_words"] = ", ".join(w.strip() for w, _ in neg)
        else:
            row["shap_top_supporting_words"] = ""
            row["shap_top_opposing_words"] = ""

        rows.append(row)

    out_df = pd.DataFrame(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows -> {out_path}")
    if unmatched:
        print(f"Note: {unmatched} explained responses had no matching row in the "
              f"marked CSV (text didn't match exactly) — ground-truth columns are blank for these.")

    if "mcq_vs_reasoning_mismatch" in out_df.columns:
        n_mcq_mismatch = out_df["mcq_vs_reasoning_mismatch"].sum()
        print(f"Student MCQ-vs-reasoning mismatches: {n_mcq_mismatch} / {len(out_df)}")
    if "model_vs_groundtruth_mismatch" in out_df.columns:
        n_model_mismatch = out_df["model_vs_groundtruth_mismatch"].sum()
        print(f"Model-vs-ground-truth mismatches: {n_model_mismatch} / {len(out_df)}")


if __name__ == "__main__":
    main()
