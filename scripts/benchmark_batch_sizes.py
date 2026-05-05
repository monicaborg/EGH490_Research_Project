"""Benchmark batch size impact on model accuracy.

Trains the same model on the same fold with different batch sizes and
compares the final metrics. This produces a table proving (or disproving)
that reducing batch size for memory reasons doesn't hurt accuracy.

The key insight: with the same data, same seed, same learning rate, and
same number of epochs, different batch sizes should produce similar final
accuracy. Small differences are expected due to gradient noise; large
differences would indicate a problem.

Usage
-----
Default — test ELECTRA at batch sizes 16, 8, 4::

    python scripts/benchmark_batch_sizes.py

Different model::

    python scripts/benchmark_batch_sizes.py --checkpoint roberta-base

Custom batch sizes::

    python scripts/benchmark_batch_sizes.py --batch-sizes 16 8 4 2

Fewer epochs for a quick check::

    python scripts/benchmark_batch_sizes.py --epochs 3

The output is a comparison table suitable for your progress report.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare training outcomes across different batch sizes."
    )
    p.add_argument("--csv", default="data/synthetic/synthetic_responses.csv")
    p.add_argument("--task", default="validity", choices=["validity", "confidence"])
    p.add_argument("--fold", type=int, default=1, help="Which fold to use (1-indexed)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--checkpoint", default="google/electra-small-discriminator")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 8, 4],
        help="Batch sizes to compare (default: 16 8 4)",
    )
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from egh490.utils import set_global_seed, get_logger

    logger = get_logger("batch_benchmark")
    model_name = Path(args.checkpoint).name

    logger.info("=" * 70)
    logger.info("EGH490 — Batch size benchmark")
    logger.info("=" * 70)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Task:       %s", args.task)
    logger.info("Fold:       %d", args.fold)
    logger.info("Epochs:     %d", args.epochs)
    logger.info("LR:         %s", args.lr)
    logger.info("Seed:       %d", args.seed)
    logger.info("Batch sizes: %s", args.batch_sizes)
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
    train_idx, test_idx = folds[args.fold - 1]
    train_texts, train_labels = dm.get_texts_and_labels(train_idx)
    test_texts, test_labels = dm.get_texts_and_labels(test_idx)

    logger.info("Train: %d examples, Test: %d examples", len(train_texts), len(test_texts))

    # ---- Run each batch size ------------------------------------------- #
    results: list[dict] = []

    for batch_size in args.batch_sizes:
        logger.info("")
        logger.info("=" * 70)
        logger.info("BATCH SIZE: %d", batch_size)
        logger.info("=" * 70)

        # Same seed every time so weight initialisation is identical
        set_global_seed(args.seed)

        try:
            from egh490.models import TransformerClassifier, Trainer, TrainingConfig

            clf = TransformerClassifier(
                args.checkpoint,
                num_labels=dm.num_labels,
                max_length=args.max_length,
                device=args.device,
            )

            cfg = TrainingConfig(
                epochs=args.epochs,
                batch_size=batch_size,
                eval_batch_size=batch_size * 2,
                learning_rate=args.lr,
                warmup_ratio=args.warmup_ratio,
                early_stopping_patience=args.patience,
                seed=args.seed,
                output_dir=f"outputs/checkpoints/_batch_bench_{batch_size}",
                fp16=False,
            )

            steps_per_epoch = len(train_texts) // batch_size + (
                1 if len(train_texts) % batch_size else 0
            )

            trainer = Trainer(clf, cfg)
            start_time = time.time()
            metrics = trainer.fit(train_texts, train_labels, test_texts, test_labels)
            elapsed = time.time() - start_time

            # Also run direct inference as a sanity check
            probs = clf.predict_proba(test_texts)
            preds = probs.argmax(axis=1)
            direct_accuracy = (preds == np.array(test_labels)).mean()

            result = {
                "batch_size": batch_size,
                "steps_per_epoch": steps_per_epoch,
                "accuracy": metrics.get("accuracy", float("nan")),
                "f1_macro": metrics.get("f1_macro", float("nan")),
                "auc": metrics.get("auc", float("nan")),
                "loss": metrics.get("loss", float("nan")),
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "direct_accuracy": direct_accuracy,
                "time_seconds": elapsed,
                "status": "OK",
            }
            results.append(result)

            logger.info("Batch %d complete in %.1fs", batch_size, elapsed)
            logger.info("  accuracy=%.4f  f1=%.4f  auc=%.4f  loss=%.4f",
                        result["accuracy"], result["f1_macro"],
                        result["auc"], result["loss"])

        except Exception as e:
            logger.error("Batch size %d FAILED: %s", batch_size, str(e))
            results.append({
                "batch_size": batch_size,
                "steps_per_epoch": 0,
                "accuracy": float("nan"),
                "f1_macro": float("nan"),
                "auc": float("nan"),
                "loss": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "direct_accuracy": float("nan"),
                "time_seconds": 0,
                "status": f"FAILED: {e}",
            })

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

    # ---- Comparison table ---------------------------------------------- #
    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH SIZE COMPARISON — %s on %s (fold %d)", model_name, args.task, args.fold)
    logger.info("=" * 70)
    logger.info("")

    header = (
        f"{'Batch':<8}{'Steps/ep':<10}{'Accuracy':<12}{'F1 macro':<12}"
        f"{'AUC':<12}{'Loss':<12}{'Time':<10}{'Status':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        row = (
            f"{r['batch_size']:<8}"
            f"{r['steps_per_epoch']:<10}"
            f"{r['accuracy']:<12.4f}"
            f"{r['f1_macro']:<12.4f}"
            f"{r['auc']:<12.4f}"
            f"{r['loss']:<12.4f}"
            f"{r['time_seconds']:<10.0f}s"
            f"{r['status']:<10}"
        )
        logger.info(row)

    # ---- Analysis ------------------------------------------------------ #
    successful = [r for r in results if r["status"] == "OK"]

    if len(successful) >= 2:
        accuracies = [r["accuracy"] for r in successful]
        f1s = [r["f1_macro"] for r in successful]

        acc_range = max(accuracies) - min(accuracies)
        f1_range = max(f1s) - min(f1s)

        logger.info("")
        logger.info("Analysis:")
        logger.info("  Accuracy range across batch sizes: %.4f (%.1f%%)", acc_range, acc_range * 100)
        logger.info("  F1 macro range across batch sizes: %.4f (%.1f%%)", f1_range, f1_range * 100)

        if acc_range < 0.05:
            logger.info("  → Accuracy varies by less than 5%% — batch size has minimal impact.")
        elif acc_range < 0.10:
            logger.info("  → Accuracy varies by 5-10%% — some impact, consider gradient accumulation.")
        else:
            logger.info("  → Accuracy varies by more than 10%% — batch size significantly affects results.")
            logger.info("    Consider using gradient accumulation to maintain effective batch size.")

        # Time scaling
        times = [r["time_seconds"] for r in successful]
        batches = [r["batch_size"] for r in successful]
        if len(times) >= 2:
            fastest = min(times)
            slowest = max(times)
            logger.info(
                "  Training time: %.0fs (batch %d) to %.0fs (batch %d) — %.1fx slowdown",
                fastest,
                batches[times.index(fastest)],
                slowest,
                batches[times.index(slowest)],
                slowest / fastest if fastest > 0 else 0,
            )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Done. Use this table to justify batch size choices in your report.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()