"""Train a single transformer model on one fold of the CCU data.

This is the first script that produces visible output — training logs,
per-epoch metrics, and final evaluation results printed to the terminal.

Usage
-----
Train ELECTRA on the validity task using synthetic data (default fold 1)::

    python scripts/train.py

Train on a specific fold::

    python scripts/train.py --fold 3

Train on the confidence task::

    python scripts/train.py --task confidence

Train with a different model::

    python scripts/train.py --checkpoint roberta-base

Use the real data (after ethics approval)::

    python scripts/train.py --csv data/raw/labelled_responses.csv

Override training parameters::

    python scripts/train.py --epochs 10 --lr 3e-5 --batch-size 8

All arguments have sensible defaults matching configs/base.yaml so you
can just run ``python scripts/train.py`` with no arguments to see it work.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a transformer on CCU validity/confidence data."
    )
    # Data
    p.add_argument(
        "--csv",
        default="data/synthetic/synthetic_responses.csv",
        help="Path to the labelled CSV (default: synthetic data)",
    )
    p.add_argument(
        "--task",
        default="validity",
        choices=["validity", "confidence"],
        help="Which classification task (default: validity)",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Which CV fold to train on, 1-indexed (default: 1)",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    # Model
    p.add_argument(
        "--checkpoint",
        default="google/electra-small-discriminator",
        help="HuggingFace model checkpoint (default: ELECTRA-small)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max tokenised sequence length (default: 256)",
    )
    # Training
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260413)
    # Output
    p.add_argument(
        "--output-dir",
        default="outputs/checkpoints",
        help="Directory for saved checkpoints",
    )
    p.add_argument(
        "--save-model",
        action="store_true",
        help="Save the fine-tuned model after training",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device: cpu, cuda, mps (default: auto-detect)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ---- Setup --------------------------------------------------------- #
    from egh490.utils import set_global_seed, get_logger

    set_global_seed(args.seed)
    logger = get_logger("train")

    logger.info("=" * 60)
    logger.info("EGH490 — Training script")
    logger.info("=" * 60)
    logger.info("CSV:        %s", args.csv)
    logger.info("Task:       %s", args.task)
    logger.info("Fold:       %d / %d", args.fold, args.n_folds)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Epochs:     %d", args.epochs)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("LR:         %s", args.lr)
    logger.info("Seed:       %d", args.seed)
    logger.info("Device:     %s", args.device or "auto")

    # ---- Load data ----------------------------------------------------- #
    from egh490.data import DataModule

    dm = DataModule(
        csv_path=args.csv,
        task=args.task,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    logger.info("Loaded %d responses (%d after preprocessing)", len(dm), len(dm))

    # Select the requested fold
    folds = list(dm.kfold_splits())
    if args.fold < 1 or args.fold > len(folds):
        logger.error("Fold %d is out of range (1-%d)", args.fold, len(folds))
        sys.exit(1)

    train_idx, test_idx = folds[args.fold - 1]
    train_texts, train_labels = dm.get_texts_and_labels(train_idx)
    test_texts, test_labels = dm.get_texts_and_labels(test_idx)

    label_names = dm.label_names
    logger.info(
        "Fold %d: %d train, %d test",
        args.fold,
        len(train_texts),
        len(test_texts),
    )
    logger.info(
        "Train label distribution: %s",
        {label_names[l]: train_labels.count(l) for l in sorted(set(train_labels))},
    )
    logger.info(
        "Test label distribution:  %s",
        {label_names[l]: test_labels.count(l) for l in sorted(set(test_labels))},
    )

    # ---- Build model --------------------------------------------------- #
    from egh490.models import TransformerClassifier, Trainer, TrainingConfig

    clf = TransformerClassifier(
        args.checkpoint,
        num_labels=dm.num_labels,
        max_length=args.max_length,
        device=args.device,
    )

    # ---- Configure training -------------------------------------------- #
    run_name = (
        f"{args.task}_{Path(args.checkpoint).name}_fold{args.fold}"
    )
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

    # ---- Train --------------------------------------------------------- #
    logger.info("-" * 60)
    logger.info("Starting training: %s", run_name)
    logger.info("-" * 60)

    trainer = Trainer(clf, cfg)
    start_time = time.time()
    metrics = trainer.fit(train_texts, train_labels, test_texts, test_labels)
    elapsed = time.time() - start_time

    # ---- Report results ------------------------------------------------ #
    logger.info("-" * 60)
    logger.info("Training complete in %.1f seconds", elapsed)
    logger.info("-" * 60)

    if metrics:
        logger.info("Evaluation metrics on test fold:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info("  %-20s %.4f", key, value)
            else:
                logger.info("  %-20s %s", key, value)

    # ---- Post-training inference check --------------------------------- #
    logger.info("-" * 60)
    logger.info("Post-training inference check on test fold:")

    probs = clf.predict_proba(test_texts)
    preds = probs.argmax(axis=1)
    test_labels_arr = np.array(test_labels)
    correct = (preds == test_labels_arr).sum()
    total = len(test_labels_arr)

    logger.info("  %d / %d correct (%.1f%%)", correct, total, 100 * correct / total)

    # Show a few example predictions
    logger.info("")
    logger.info("Sample predictions (first 5 test responses):")
    for i in range(min(5, len(test_texts))):
        pred_label = label_names[preds[i]]
        true_label = label_names[test_labels[i]]
        prob_str = f"[{probs[i][0]:.3f}, {probs[i][1]:.3f}]"
        match = "✓" if preds[i] == test_labels[i] else "✗"
        text_preview = test_texts[i][:60] + ("..." if len(test_texts[i]) > 60 else "")
        logger.info(
            "  %s  pred=%-10s  true=%-10s  probs=%s  %s",
            match,
            pred_label,
            true_label,
            prob_str,
            text_preview,
        )

    # ---- Save model ---------------------------------------------------- #
    if args.save_model:
        save_path = Path(checkpoint_dir) / "final"
        clf.save(save_path)
        logger.info("Model saved to %s", save_path)

    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 