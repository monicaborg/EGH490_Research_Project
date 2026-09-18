"""Generate Appendix A14: MCQ-correctness x reasoning-validity cross-tabulation,
and the classifier's error distribution across those four categories.

Two outputs per CCU:

  Table A14.1 (ground truth only, no model needed):
    Valid+Correct | Valid+Incorrect (slip) | Invalid+Correct (guess) | Invalid+Incorrect

  Table A14.2 (classifier errors on held-out folds):
    for each of the four ground-truth categories, how many responses in that
    category did the ensemble misclassify (predicted validity != annotated
    validity)?

Table A14.2 requires re-predicting on the held-out test split of each fold,
using the same soft-voting ensemble and the same fold reconstruction as
scripts/evaluate_ensemble.py, so that "held-out" is genuine (a response is
only ever scored by a fold that did not train on it).

Usage
-----
    for ccu in ccu1 ccu2 ccu3 ccu4 ccu5 ccu6; do
      python scripts/mcq_mismatch_analysis.py \\
        --csv data/raw/signals_systems_validity_corpus.csv --ccu $ccu \\
        --checkpoint-dir outputs/checkpoints \\
        --dataset-tag signals_systems_validity_corpus_$ccu
    done

Writes outputs/agreement/mcq_mismatch_<ccu>.json and prints both tables.
Also writes a combined outputs/agreement/mcq_mismatch_all.json with every
CCU's numbers, for pasting straight into Appendix A14.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORRECT_MCQ = {
    "ccu1": "a", "ccu2": "c", "ccu3": "b",
    "ccu4": "b", "ccu5": "d", "ccu6": "b",
}

CATEGORIES = ["valid_correct", "valid_incorrect_slip",
              "invalid_correct_guess", "invalid_incorrect"]
CATEGORY_LABELS = {
    "valid_correct": "Valid + Correct",
    "valid_incorrect_slip": "Valid + Incorrect (slip)",
    "invalid_correct_guess": "Invalid + Correct (guess)",
    "invalid_incorrect": "Invalid + Incorrect",
}


def categorise(mcq_correct: bool, reasoning_valid: bool) -> str:
    if reasoning_valid and mcq_correct:
        return "valid_correct"
    if reasoning_valid and not mcq_correct:
        return "valid_incorrect_slip"
    if not reasoning_valid and mcq_correct:
        return "invalid_correct_guess"
    return "invalid_incorrect"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--ccu", required=True)
    p.add_argument("--task", default="validity")
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    p.add_argument("--dataset-tag", required=True)
    p.add_argument("--models", nargs="*", default=["roberta", "albert", "xlnet"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--output-dir", default="outputs/agreement")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from egh490.data import DataModule
    from egh490.data.schema import COL_CCU
    from egh490.models import Ensemble, TransformerClassifier
    from egh490.utils import get_logger, set_global_seed

    set_global_seed(args.seed)
    logger = get_logger("mcq_mismatch_analysis")

    correct_answer = CORRECT_MCQ.get(args.ccu)
    if correct_answer is None:
        raise ValueError(f"No known correct MCQ answer for {args.ccu!r}")

    # ── Table A14.1: ground-truth cross-tab (all responses, no model) ──
    df = pd.read_csv(args.csv)
    ccu_df = df[df[COL_CCU] == args.ccu].copy()
    ccu_df["mcq_correct"] = (
        ccu_df["q1mcr"].astype(str).str.strip().str.lower() == correct_answer
    )
    ccu_df["reasoning_valid"] = ccu_df["validity"] == "correct"
    ccu_df["category"] = [
        categorise(mc, rv) for mc, rv in zip(ccu_df["mcq_correct"], ccu_df["reasoning_valid"])
    ]
    gt_counts = {c: int((ccu_df["category"] == c).sum()) for c in CATEGORIES}
    gt_counts["n"] = int(len(ccu_df))

    logger.info("Table A14.1 (%s): %s", args.ccu,
               {CATEGORY_LABELS[c]: gt_counts[c] for c in CATEGORIES})

    # ── Table A14.2: classifier errors on held-out folds ────────────────
    dm = DataModule(csv_path=args.csv, task=args.task, seed=args.seed,
                    n_folds=args.n_folds)
    df_full = dm.get_dataframe()
    ccu_mask = df_full[COL_CCU] == args.ccu
    dm._df = df_full[ccu_mask].reset_index(drop=True)
    dm.texts = dm._df[dm.text_col].tolist()
    dm.labels = dm._df["_label"].tolist()
    logger.info("CCU filter: %s — %d responses", args.ccu, len(dm.texts))

    folds = list(dm.kfold_splits())

    # uid -> (predicted_label, true_label) collected across folds (each
    # response is predicted exactly once, by the fold that held it out).
    predictions: dict = {}
    uid_col = "uid" if "uid" in dm._df.columns else None

    for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
        texts, labels = dm.get_texts_and_labels(list(test_idx))
        labels = np.asarray(labels)

        classifiers = []
        for key in args.models:
            path = Path(args.checkpoint_dir) / \
                f"{args.task}_{key}_fold{fold_num}_{args.dataset_tag}" / "final"
            if not path.exists():
                logger.warning("Missing %s fold%d — skipping member", key, fold_num)
                continue
            classifiers.append(TransformerClassifier.load(str(path), num_labels=2))

        if len(classifiers) < 2:
            logger.error("Fold %d: fewer than 2 members — skipping fold", fold_num)
            continue

        ens = Ensemble(classifiers, strategy="soft")
        proba = ens.predict_proba(texts)
        preds = proba.argmax(axis=1)

        for local_i, global_i in enumerate(test_idx):
            uid = dm._df.iloc[global_i][uid_col] if uid_col else global_i
            predictions[uid] = (int(preds[local_i]), int(labels[local_i]))

        logger.info("Fold %d: predicted %d held-out responses", fold_num, len(test_idx))

    # Join predictions back to the category table.
    if uid_col:
        ccu_df = ccu_df.set_index(uid_col, drop=False)

    error_counts = {c: 0 for c in CATEGORIES}
    n_predicted = 0
    for uid, (pred, true) in predictions.items():
        if uid not in ccu_df.index:
            continue
        cat = ccu_df.loc[uid, "category"]
        if isinstance(cat, pd.Series):  # duplicate uid guard
            cat = cat.iloc[0]
        n_predicted += 1
        if pred != true:
            error_counts[cat] += 1

    error_counts["total_errors"] = sum(error_counts[c] for c in CATEGORIES)
    error_counts["n_predicted"] = n_predicted

    logger.info("Table A14.2 (%s): %s", args.ccu,
               {CATEGORY_LABELS[c]: error_counts[c] for c in CATEGORIES})

    # ── Save ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ccu": args.ccu,
        "correct_mcq_answer": correct_answer,
        "ground_truth_counts": gt_counts,
        "error_counts": error_counts,
    }
    out_path = out_dir / f"mcq_mismatch_{args.ccu}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved -> %s", out_path)

    print("\n" + "=" * 70)
    print(f"CCU: {args.ccu}  (correct MCQ answer: {correct_answer})")
    print("=" * 70)
    print(f"{'Category':<28}{'Ground truth n':>16}{'Classifier errors':>20}")
    for c in CATEGORIES:
        print(f"{CATEGORY_LABELS[c]:<28}{gt_counts[c]:>16}{error_counts[c]:>20}")
    print(f"{'TOTAL':<28}{gt_counts['n']:>16}{error_counts['total_errors']:>20}")
    print("=" * 70)


if __name__ == "__main__":
    main()
