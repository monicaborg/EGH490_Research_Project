"""Train a single model across all k folds and report averaged metrics.

Runs each fold sequentially (not concurrently) to stay within MacBook
RAM limits. If a fold fails (e.g. XLNet OOM), it logs the error and
continues to the next fold so you still get partial results.

Usage
-----
Default — ELECTRA on synthetic data, all 5 folds::

    python scripts/train_all_folds.py

Different model::

    python scripts/train_all_folds.py --checkpoint roberta-base

XLNet with reduced memory::

    python scripts/train_all_folds.py --checkpoint xlnet-base-cased --batch-size 4 --max-length 128

Real data::

    python scripts/train_all_folds.py --csv data/raw/labelled_responses.csv

Save all trained models for later ensemble/XAI use::

    python scripts/train_all_folds.py --save-models

Skip a fold that previously failed::

    python scripts/train_all_folds.py --skip-folds 3 5

The final output is a summary table suitable for pasting into your report.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one model across all CV folds, report averaged metrics."
    )
    # Data
    p.add_argument("--csv", default="data/synthetic/synthetic_responses.csv")
    p.add_argument("--task", default="validity", choices=["validity", "confidence"])
    p.add_argument("--n-folds", type=int, default=5)
    # Model
    p.add_argument("--checkpoint", default="google/electra-small-discriminator")
    p.add_argument("--max-length", type=int, default=256)
    # Training
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260413)
    # Output
    p.add_argument("--output-dir", default="outputs/checkpoints")
    p.add_argument("--save-models", action="store_true", help="Save each fold's model")
    p.add_argument("--device", default=None)
    # Recovery
    p.add_argument(
        "--skip-folds",
        type=int,
        nargs="*",
        default=[],
        help="Fold numbers to skip (1-indexed), e.g. --skip-folds 3 5",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from egh490.utils import set_global_seed, get_logger

    set_global_seed(args.seed)
    logger = get_logger("train_all_folds")

    # ---- Header -------------------------------------------------------- #
    model_name = Path(args.checkpoint).name
    logger.info("=" * 70)
    logger.info("EGH490 — Train all folds")
    logger.info("=" * 70)
    logger.info("CSV:        %s", args.csv)
    logger.info("Task:       %s", args.task)
    logger.info("Folds:      %d", args.n_folds)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Epochs:     %d", args.epochs)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("LR:         %s", args.lr)
    logger.info("Max length: %d", args.max_length)
    logger.info("Seed:       %d", args.seed)
    logger.info("Device:     %s", args.device or "auto")
    if args.skip_folds:
        logger.info("Skipping:   folds %s", args.skip_folds)
    logger.info("=" * 70)

    # ---- Load data once ------------------------------------------------ #
    from egh490.data import DataModule

    dm = DataModule(
        csv_path=args.csv,
        task=args.task,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    folds = list(dm.kfold_splits())

    # ---- Train each fold sequentially ---------------------------------- #
    fold_metrics: dict[int, dict[str, float]] = {}
    fold_times: dict[int, float] = {}
    failed_folds: list[int] = []

    for fold_num in range(1, args.n_folds + 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info("FOLD %d / %d", fold_num, args.n_folds)
        logger.info("=" * 70)

        if fold_num in args.skip_folds:
            logger.info("Skipping fold %d (--skip-folds)", fold_num)
            continue

        train_idx, test_idx = folds[fold_num - 1]
        train_texts, train_labels = dm.get_texts_and_labels(train_idx)
        test_texts, test_labels = dm.get_texts_and_labels(test_idx)

        logger.info(
            "Train: %d examples (%s)",
            len(train_texts),
            {dm.label_names[l]: train_labels.count(l) for l in sorted(set(train_labels))},
        )
        logger.info(
            "Test:  %d examples (%s)",
            len(test_texts),
            {dm.label_names[l]: test_labels.count(l) for l in sorted(set(test_labels))},
        )

        try:
            # Build a fresh model for each fold to avoid weight leakage
            from egh490.models import TransformerClassifier, Trainer, TrainingConfig

            clf = TransformerClassifier(
                args.checkpoint,
                num_labels=dm.num_labels,
                max_length=args.max_length,
                device=args.device,
            )

            run_name = f"{args.task}_{model_name}_fold{fold_num}"
            checkpoint_dir = str(Path(args.output_dir) / run_name)

            cfg = TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.lr,
                warmup_ratio=args.warmup_ratio,
                early_stopping_patience=args.patience,
                seed=args.seed,
                output_dir=checkpoint_dir,
                fp16=False,
            )

            trainer = Trainer(clf, cfg)
            start_time = time.time()
            metrics = trainer.fit(train_texts, train_labels, test_texts, test_labels)
            elapsed = time.time() - start_time

            fold_metrics[fold_num] = metrics
            fold_times[fold_num] = elapsed

            logger.info("Fold %d complete in %.1fs", fold_num, elapsed)
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    logger.info("  %-20s %.4f", key, value)

            # Save if requested
            if args.save_models:
                save_path = Path(checkpoint_dir) / "final"
                clf.save(save_path)
                logger.info("Model saved to %s", save_path)

        except Exception as e:
            logger.error("=" * 70)
            logger.error("FOLD %d FAILED: %s", fold_num, str(e))
            logger.error("=" * 70)
            logger.error(traceback.format_exc())
            failed_folds.append(fold_num)

        finally:
            # Free memory between folds — critical for XLNet/RoBERTa on 16GB
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

    # ---- Summary table ------------------------------------------------- #
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY — %s on %s task", model_name, args.task)
    logger.info("=" * 70)

    if not fold_metrics:
        logger.error("No folds completed successfully. Cannot report metrics.")
        sys.exit(1)

    # Collect metric keys from the first successful fold
    metric_keys = [
        k for k in sorted(next(iter(fold_metrics.values())).keys())
        if isinstance(next(iter(fold_metrics.values()))[k], float)
        and k not in ("runtime", "samples_per_second", "steps_per_second", "epoch")
    ]

    # Per-fold table
    header = f"{'Fold':<8}" + "".join(f"{k:<14}" for k in metric_keys)
    logger.info(header)
    logger.info("-" * len(header))

    for fold_num in sorted(fold_metrics.keys()):
        row = f"{fold_num:<8}"
        for k in metric_keys:
            val = fold_metrics[fold_num].get(k, float("nan"))
            row += f"{val:<14.4f}"
        row += f"  ({fold_times[fold_num]:.0f}s)"
        logger.info(row)

    # Averages
    logger.info("-" * len(header))
    avg_row = f"{'Mean':<8}"
    std_row = f"{'Std':<8}"
    for k in metric_keys:
        values = [fold_metrics[f][k] for f in fold_metrics if k in fold_metrics[f]]
        avg_row += f"{np.mean(values):<14.4f}"
        std_row += f"{np.std(values):<14.4f}"
    logger.info(avg_row)
    logger.info(std_row)

    # Total time
    total_time = sum(fold_times.values())
    logger.info("")
    logger.info("Total training time: %.1f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("Successful folds: %d / %d", len(fold_metrics), args.n_folds)

    if failed_folds:
        logger.warning(
            "Failed folds: %s — try re-running with --batch-size %d or --skip-folds %s",
            failed_folds,
            max(1, args.batch_size // 2),
            " ".join(str(f) for f in failed_folds),
        )

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()