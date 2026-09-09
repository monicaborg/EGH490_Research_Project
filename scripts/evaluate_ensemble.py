"""Evaluate the soft-voting ensemble on the same folds used for training.

train_all_models.py reports per-model cross-validation accuracy, but the
research question concerns the ensemble. This script reconstructs the
identical stratified folds (same seed, same CSV, same CCU filter), loads the
per-fold checkpoints for each ensemble member, soft-votes their probabilities
on that fold's held-out test set, and reports ensemble metrics in the same
format as the per-model results.

No retraining occurs — this is inference only over checkpoints already saved
by train_all_models.py --save-models.

Usage
-----
    python scripts/evaluate_ensemble.py \\
        --csv data/raw/signals_systems_validity_corpus.csv \\
        --ccu ccu1 \\
        --checkpoint-dir outputs/checkpoints \\
        --dataset-tag signals_systems_validity_corpus_ccu1

    # all six CCUs
    for ccu in ccu1 ccu2 ccu3 ccu4 ccu5 ccu6; do
      python scripts/evaluate_ensemble.py \\
        --csv data/raw/signals_systems_validity_corpus.csv --ccu $ccu \\
        --checkpoint-dir outputs/checkpoints \\
        --dataset-tag signals_systems_validity_corpus_$ccu
    done
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the soft-voting ensemble.")
    p.add_argument("--csv", required=True)
    p.add_argument("--ccu", default=None, help="Restrict to one CCU (ccu1-ccu6)")
    p.add_argument("--task", default="validity", choices=["validity", "confidence"])
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    p.add_argument("--dataset-tag", required=True,
                   help="Tag used when the checkpoints were saved")
    p.add_argument("--models", nargs="*", default=["roberta", "albert", "xlnet"],
                   help="Ensemble members (default excludes electra)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--strategy", default="soft", choices=["soft", "hard"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from egh490.data import DataModule
    from egh490.data.schema import get_task_config
    from egh490.models import Ensemble, TransformerClassifier
    from egh490.utils import get_logger, set_global_seed
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    set_global_seed(args.seed)
    logger = get_logger("evaluate_ensemble")

    task_cfg = get_task_config(args.task)
    num_labels = task_cfg["num_labels"]
    label_names = [task_cfg["label_names"][i] for i in range(num_labels)]

    # Rebuild the exact same DataModule / folds used during training.
    # train_all_models.py filters by mutating the DataModule in place *before*
    # calling kfold_splits(), so fold indices are relative to the CCU subset.
    # This must be replicated exactly or the folds will not match.
    dm = DataModule(csv_path=args.csv, task=args.task, seed=args.seed,
                    n_folds=args.n_folds)

    if args.ccu:
        from egh490.data.schema import COL_CCU
        df_full = dm.get_dataframe()
        if COL_CCU not in df_full.columns:
            raise ValueError(f"--ccu given but no {COL_CCU!r} column in the CSV")
        available = sorted(df_full[COL_CCU].unique().tolist())
        if args.ccu not in available:
            raise ValueError(f"--ccu {args.ccu!r} not found. Available: {available}")
        ccu_df = df_full[df_full[COL_CCU] == args.ccu].reset_index(drop=True)
        dm._df = ccu_df
        dm.texts = ccu_df[dm.text_col].tolist()
        dm.labels = ccu_df["_label"].tolist()
        logger.info("CCU filter: %s — %d responses", args.ccu, len(dm.texts))

    folds = list(dm.kfold_splits())

    fold_results = {}
    for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
        texts, labels = dm.get_texts_and_labels(list(test_idx))
        labels = np.asarray(labels)

        classifiers = []
        for key in args.models:
            path = Path(args.checkpoint_dir) / \
                f"{args.task}_{key}_fold{fold_num}_{args.dataset_tag}" / "final"
            if not path.exists():
                logger.warning("Missing %s fold%d at %s — skipping member",
                               key, fold_num, path)
                continue
            classifiers.append(TransformerClassifier.load(str(path),
                                                          num_labels=num_labels))

        if len(classifiers) < 2:
            logger.error("Fold %d: fewer than 2 members available — skipping", fold_num)
            continue

        ens = Ensemble(classifiers, strategy=args.strategy)
        proba = ens.predict_proba(texts)
        preds = proba.argmax(axis=1)

        res = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "precision": float(precision_score(labels, preds, average="macro",
                                                zero_division=0)),
            "recall": float(recall_score(labels, preds, average="macro",
                                          zero_division=0)),
            "n_test": int(len(labels)),
            "n_members": len(classifiers),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }
        if num_labels == 2 and len(set(labels.tolist())) == 2:
            res["auc"] = float(roc_auc_score(labels, proba[:, 1]))
        fold_results[str(fold_num)] = res
        logger.info("Fold %d: acc=%.4f f1=%.4f auc=%s (n=%d, %d members)",
                    fold_num, res["accuracy"], res["f1_macro"],
                    f"{res.get('auc', float('nan')):.4f}", res["n_test"],
                    res["n_members"])

    if not fold_results:
        raise RuntimeError("No folds evaluated — check checkpoint paths.")

    keys = ["accuracy", "f1_macro", "precision", "recall"] + \
           (["auc"] if "auc" in next(iter(fold_results.values())) else [])
    mean = {k: float(np.mean([f[k] for f in fold_results.values() if k in f]))
            for k in keys}
    std = {k: float(np.std([f[k] for f in fold_results.values() if k in f]))
           for k in keys}

    out = {
        "model_key": f"ensemble_{args.strategy}",
        "members": args.models,
        "task": args.task,
        "ccu": args.ccu,
        "dataset_tag": args.dataset_tag,
        "n_folds": len(fold_results),
        "label_names": label_names,
        "folds": fold_results,
        "mean": mean,
        "std": std,
    }

    out_dir = Path(args.metrics_dir) / args.dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ensemble_{args.task}_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    logger.info("=" * 60)
    logger.info("ENSEMBLE (%s vote, members=%s) — %s",
                args.strategy, ", ".join(args.models), args.dataset_tag)
    logger.info("  Accuracy: %.4f (sd %.4f)", mean["accuracy"], std["accuracy"])
    logger.info("  F1 macro: %.4f", mean["f1_macro"])
    if "auc" in mean:
        logger.info("  AUC:      %.4f", mean["auc"])
    logger.info("  Saved -> %s", out_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
