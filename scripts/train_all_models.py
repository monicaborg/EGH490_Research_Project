"""Train all four transformer models across all 5 folds and save metrics.

Runs sequentially: ELECTRA → RoBERTa → XLNet → ALBERT, 5 folds each.
After each model completes, saves a JSON file to outputs/metrics/ so that
plot_results.py can read the results without any manual data entry.

Usage
-----
Default — all four models on synthetic data (100 rows)::

    python scripts/train_all_models.py

Use 2000-row synthetic dataset::

    python scripts/train_all_models.py --csv data/synthetic/synthetic_responses_2000.csv

Real data (when available)::

    python scripts/train_all_models.py --csv data/raw/labelled_responses.csv

Skip models already trained::

    python scripts/train_all_models.py --skip-models electra roberta

Single model only::

    python scripts/train_all_models.py --models albert

Each model's results are saved to:
    outputs/metrics/<dataset_tag>/<model_name>_validity_results.json

The dataset_tag is derived from the CSV filename so results from different
datasets don't overwrite each other.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


MODEL_CONFIGS = [
    {
        "key": "electra",
        "checkpoint": "google/electra-small-discriminator",
        "batch_size": 16,
        "max_length": 256,
    },
    {
        "key": "roberta",
        "checkpoint": "roberta-base",
        "batch_size": 8,
        "max_length": 256,
    },
    {
        "key": "xlnet",
        "checkpoint": "xlnet-base-cased",
        "batch_size": 4,
        "max_length": 128,
    },
    {
        "key": "albert",
        "checkpoint": "albert-base-v2",
        "batch_size": 16,
        "max_length": 256,
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train all four transformer models across all 5 folds."
    )
    p.add_argument("--csv", default="data/synthetic/synthetic_responses.csv")
    p.add_argument("--task", default="validity", choices=["validity", "confidence"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--output-dir", default="outputs/checkpoints")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--models",
        nargs="*",
        choices=["electra", "roberta", "xlnet", "albert"],
        default=None,
        help="Only train these models (default: all four)",
    )
    p.add_argument(
        "--skip-models",
        nargs="*",
        choices=["electra", "roberta", "xlnet", "albert"],
        default=[],
        help="Skip these models",
    )
    return p.parse_args(argv)


def save_metrics(
    metrics_dir: Path,
    dataset_tag: str,
    model_key: str,
    task: str,
    checkpoint: str,
    batch_size: int,
    fold_metrics: dict,
    fold_times: dict,
    total_time: float,
) -> Path:
    """Save fold metrics to a JSON file."""
    tag_dir = metrics_dir / dataset_tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        k for k in sorted(next(iter(fold_metrics.values())).keys())
        if isinstance(next(iter(fold_metrics.values()))[k], float)
        and k not in ("runtime", "samples_per_second", "steps_per_second", "epoch")
    ]

    folds_data = {}
    for fold_num, m in fold_metrics.items():
        folds_data[str(fold_num)] = {k: m[k] for k in metric_keys if k in m}
        folds_data[str(fold_num)]["time_seconds"] = fold_times[fold_num]

    means = {k: float(np.mean([fold_metrics[f][k] for f in fold_metrics if k in fold_metrics[f]])) for k in metric_keys}
    stds = {k: float(np.std([fold_metrics[f][k] for f in fold_metrics if k in fold_metrics[f]])) for k in metric_keys}

    result = {
        "model_key": model_key,
        "checkpoint": checkpoint,
        "task": task,
        "batch_size": batch_size,
        "n_folds": len(fold_metrics),
        "total_time_seconds": total_time,
        "folds": folds_data,
        "mean": means,
        "std": stds,
    }

    path = tag_dir / f"{model_key}_{task}_results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from egh490.utils import set_global_seed, get_logger
    from egh490.data import DataModule

    set_global_seed(args.seed)
    logger = get_logger("train_all_models")

    # Dataset tag from CSV filename (e.g. "synthetic_responses" or "synthetic_responses_2000")
    dataset_tag = Path(args.csv).stem

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Filter models
    configs = [c for c in MODEL_CONFIGS if args.models is None or c["key"] in args.models]
    configs = [c for c in configs if c["key"] not in args.skip_models]

    logger.info("=" * 70)
    logger.info("EGH490 — Train all models")
    logger.info("=" * 70)
    logger.info("CSV:        %s  (tag: %s)", args.csv, dataset_tag)
    logger.info("Task:       %s", args.task)
    logger.info("Folds:      %d", args.n_folds)
    logger.info("Epochs:     %d", args.epochs)
    logger.info("Models:     %s", [c["key"] for c in configs])
    logger.info("=" * 70)

    # Load data once
    dm = DataModule(csv_path=args.csv, task=args.task, n_folds=args.n_folds, seed=args.seed)
    folds = list(dm.kfold_splits())

    overall_start = time.time()
    model_summaries = {}

    for cfg in configs:
        from egh490.models import TransformerClassifier, Trainer, TrainingConfig

        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL: %s  (batch=%d, max_length=%d)",
                    cfg["checkpoint"], cfg["batch_size"], cfg["max_length"])
        logger.info("=" * 70)

        fold_metrics: dict[int, dict] = {}
        fold_times: dict[int, float] = {}
        failed_folds: list[int] = []

        for fold_num in range(1, args.n_folds + 1):
            logger.info("  Fold %d / %d", fold_num, args.n_folds)
            train_idx, test_idx = folds[fold_num - 1]
            train_texts, train_labels = dm.get_texts_and_labels(train_idx)
            test_texts, test_labels = dm.get_texts_and_labels(test_idx)

            try:
                clf = TransformerClassifier(
                    cfg["checkpoint"],
                    num_labels=dm.num_labels,
                    max_length=cfg["max_length"],
                    device=args.device,
                )
                run_name = f"{args.task}_{cfg['key']}_fold{fold_num}_{dataset_tag}"
                training_cfg = TrainingConfig(
                    epochs=args.epochs,
                    batch_size=cfg["batch_size"],
                    eval_batch_size=cfg["batch_size"] * 2,
                    learning_rate=args.lr,
                    warmup_ratio=args.warmup_ratio,
                    early_stopping_patience=args.patience,
                    seed=args.seed,
                    output_dir=str(Path(args.output_dir) / run_name),
                    fp16=False,
                )
                trainer = Trainer(clf, training_cfg)
                t0 = time.time()
                metrics = trainer.fit(train_texts, train_labels, test_texts, test_labels)
                elapsed = time.time() - t0

                fold_metrics[fold_num] = metrics
                fold_times[fold_num] = elapsed

                logger.info("    acc=%.3f  f1=%.3f  auc=%.3f  time=%.0fs",
                            metrics.get("accuracy", 0),
                            metrics.get("f1_macro", 0),
                            metrics.get("auc", 0),
                            elapsed)

                if args.save_models:
                    clf.save(Path(args.output_dir) / run_name / "final")

            except Exception as e:
                logger.error("    Fold %d FAILED: %s", fold_num, str(e))
                logger.error(traceback.format_exc())
                failed_folds.append(fold_num)

            finally:
                try:
                    del clf, trainer
                except NameError:
                    pass
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except ImportError:
                    pass

        if not fold_metrics:
            logger.error("No folds completed for %s — skipping JSON save.", cfg["key"])
            continue

        total_time = sum(fold_times.values())

        # Save JSON
        json_path = save_metrics(
            metrics_dir=metrics_dir,
            dataset_tag=dataset_tag,
            model_key=cfg["key"],
            task=args.task,
            checkpoint=cfg["checkpoint"],
            batch_size=cfg["batch_size"],
            fold_metrics=fold_metrics,
            fold_times=fold_times,
            total_time=total_time,
        )
        logger.info("  Saved metrics → %s", json_path)

        model_summaries[cfg["key"]] = {
            "mean_accuracy": float(np.mean([fold_metrics[f]["accuracy"] for f in fold_metrics])),
            "mean_f1": float(np.mean([fold_metrics[f]["f1_macro"] for f in fold_metrics])),
            "mean_auc": float(np.mean([fold_metrics[f]["auc"] for f in fold_metrics])),
            "total_time_min": total_time / 60,
            "failed_folds": failed_folds,
        }

    # Final summary
    overall_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL SUMMARY — %s task, dataset: %s", args.task, dataset_tag)
    logger.info("=" * 70)
    logger.info("%-12s %-12s %-12s %-12s %-12s", "Model", "Acc (mean)", "F1 (mean)", "AUC (mean)", "Time (min)")
    logger.info("-" * 60)
    for key, s in model_summaries.items():
        logger.info("%-12s %-12.4f %-12.4f %-12.4f %-12.1f",
                    key, s["mean_accuracy"], s["mean_f1"], s["mean_auc"], s["total_time_min"])
    logger.info("-" * 60)
    logger.info("Total wall time: %.1f minutes", overall_time / 60)
    logger.info("")
    logger.info("Results saved to: outputs/metrics/%s/", dataset_tag)
    logger.info("Run `python scripts/plot_results.py --dataset %s` to generate charts.", dataset_tag)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()