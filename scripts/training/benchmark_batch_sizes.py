"""Benchmark batch size impact on model performance and save results to JSON.

Trains the same model on the same fold with different batch sizes,
compares metrics, and saves a JSON file that plot_results.py can read.

Usage
-----
Default — ELECTRA at batch sizes 16, 8, 4::

    python scripts/benchmark_batch_sizes.py

RoBERTa::

    python scripts/benchmark_batch_sizes.py --checkpoint roberta-base --batch-sizes 16 8 4

XLNet (full range 16 down to 2 as per hardware testing)::

    python scripts/benchmark_batch_sizes.py --checkpoint xlnet-base-cased --batch-sizes 16 8 4 2 --max-length 128

Run all four models in sequence (saves JSON for each)::

    python scripts/benchmark_batch_sizes.py --all-models

JSON saved to outputs/metrics/<dataset_tag>/<model_key>_<task>_batch_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


ALL_MODEL_CONFIGS = [
    {
        "key": "electra",
        "checkpoint": "google/electra-small-discriminator",
        "batch_sizes": [16, 8, 4],
        "max_length": 256,
    },
    {
        "key": "roberta",
        "checkpoint": "roberta-base",
        "batch_sizes": [16, 8, 4],
        "max_length": 256,
    },
    {
        "key": "xlnet",
        "checkpoint": "xlnet-base-cased",
        "batch_sizes": [16, 8, 4, 2],
        "max_length": 128,
    },
    {
        "key": "albert",
        "checkpoint": "albert-base-v2",
        "batch_sizes": [16, 8, 4],
        "max_length": 256,
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark batch sizes for one or all models.")
    p.add_argument("--csv", default="data/synthetic/synthetic_responses.csv")
    p.add_argument("--task", default="validity", choices=["validity", "confidence"])
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--checkpoint", default="google/electra-small-discriminator")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 8, 4])
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--device", default=None)
    p.add_argument("--all-models", action="store_true",
                   help="Run benchmarks for all four models sequentially")
    return p.parse_args(argv)


def run_benchmark_for_model(
    checkpoint: str,
    model_key: str,
    batch_sizes: list[int],
    max_length: int,
    csv_path: str,
    task: str,
    fold: int,
    n_folds: int,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    patience: int,
    seed: int,
    device,
    metrics_dir: Path,
    logger,
) -> dict:
    from egh490.utils import set_global_seed
    from egh490.data import DataModule
    from egh490.models import TransformerClassifier, Trainer, TrainingConfig

    dataset_tag = Path(csv_path).stem

    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK: %s  batch_sizes=%s", checkpoint, batch_sizes)
    logger.info("=" * 70)

    dm = DataModule(csv_path=csv_path, task=task, n_folds=n_folds, seed=seed)
    folds = list(dm.kfold_splits())
    train_idx, test_idx = folds[fold - 1]
    train_texts, train_labels = dm.get_texts_and_labels(train_idx)
    test_texts, test_labels = dm.get_texts_and_labels(test_idx)
    logger.info("Train: %d  Test: %d", len(train_texts), len(test_texts))

    results = []

    for batch_size in batch_sizes:
        logger.info("")
        logger.info("  Batch size: %d", batch_size)
        set_global_seed(seed)

        try:
            clf = TransformerClassifier(
                checkpoint, num_labels=dm.num_labels,
                max_length=max_length, device=device,
            )
            cfg = TrainingConfig(
                epochs=epochs, batch_size=batch_size,
                eval_batch_size=batch_size * 2,
                learning_rate=lr, warmup_ratio=warmup_ratio,
                early_stopping_patience=patience, seed=seed,
                output_dir=f"outputs/checkpoints/_bench_{model_key}_b{batch_size}",
                fp16=False,
            )
            steps_per_epoch = len(train_texts) // batch_size + (1 if len(train_texts) % batch_size else 0)

            trainer = Trainer(clf, cfg)
            t0 = time.time()
            metrics = trainer.fit(train_texts, train_labels, test_texts, test_labels)
            elapsed = time.time() - t0

            preds = clf.predict_proba(test_texts).argmax(axis=1)
            direct_acc = float((preds == np.array(test_labels)).mean())

            result = {
                "batch_size": batch_size,
                "steps_per_epoch": steps_per_epoch,
                "accuracy": metrics.get("accuracy", float("nan")),
                "f1_macro": metrics.get("f1_macro", float("nan")),
                "auc": metrics.get("auc", float("nan")),
                "loss": metrics.get("loss", float("nan")),
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "direct_accuracy": direct_acc,
                "time_seconds": elapsed,
                "status": "OK",
            }
            logger.info("    acc=%.3f  f1=%.3f  auc=%.3f  time=%.0fs",
                        result["accuracy"], result["f1_macro"], result["auc"], elapsed)

        except Exception as e:
            logger.error("    FAILED: %s", str(e))
            result = {
                "batch_size": batch_size, "steps_per_epoch": 0,
                "accuracy": float("nan"), "f1_macro": float("nan"),
                "auc": float("nan"), "loss": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "direct_accuracy": float("nan"), "time_seconds": 0,
                "status": f"FAILED: {e}",
            }

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

        results.append(result)

    # Summary table
    logger.info("")
    logger.info("  SUMMARY — %s", model_key)
    logger.info("  %-8s %-10s %-12s %-12s %-12s %-10s %-10s",
                "Batch", "Steps/ep", "Accuracy", "F1", "AUC", "Time", "Status")
    logger.info("  " + "-" * 74)
    for r in results:
        logger.info("  %-8d %-10d %-12.4f %-12.4f %-12.4f %-10.0fs %-10s",
                    r["batch_size"], r["steps_per_epoch"],
                    r["accuracy"], r["f1_macro"], r["auc"],
                    r["time_seconds"], r["status"])

    # Analysis
    successful = [r for r in results if r["status"] == "OK"]
    if len(successful) >= 2:
        acc_range = max(r["accuracy"] for r in successful) - min(r["accuracy"] for r in successful)
        auc_range = max(r["auc"] for r in successful) - min(r["auc"] for r in successful)
        logger.info("")
        logger.info("  Accuracy range: %.4f (%.1f%%)", acc_range, acc_range * 100)
        logger.info("  AUC range:      %.4f (%.1f%%)", auc_range, auc_range * 100)
        if acc_range < 0.05:
            logger.info("  → Accuracy stable across batch sizes (<5%% range)")
        elif acc_range < 0.10:
            logger.info("  → Moderate accuracy variance (5-10%%) — likely due to dataset size")
        else:
            logger.info("  → High accuracy variance (>10%%) — expected on small dataset, verify on real data")
        if auc_range < 0.05:
            logger.info("  → AUC stable — model learns similar representations regardless of batch size")

    # Save JSON
    tag_dir = metrics_dir / dataset_tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    json_path = tag_dir / f"{model_key}_{task}_batch_benchmark.json"

    output = {
        "model_key": model_key,
        "checkpoint": checkpoint,
        "task": task,
        "fold": fold,
        "max_length": max_length,
        "batch_sizes": [r["batch_size"] for r in results],
        "accuracy": [r["accuracy"] for r in results],
        "f1_macro": [r["f1_macro"] for r in results],
        "auc": [r["auc"] for r in results],
        "loss": [r["loss"] for r in results],
        "time_seconds": [r["time_seconds"] for r in results],
        "status": [r["status"] for r in results],
        "steps_per_epoch": [r["steps_per_epoch"] for r in results],
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("  Saved JSON → %s", json_path)

    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from egh490.utils import set_global_seed, get_logger
    logger = get_logger("batch_benchmark")
    set_global_seed(args.seed)

    metrics_dir = Path(args.metrics_dir)

    if args.all_models:
        logger.info("Running batch benchmarks for all four models...")
        for cfg in ALL_MODEL_CONFIGS:
            run_benchmark_for_model(
                checkpoint=cfg["checkpoint"],
                model_key=cfg["key"],
                batch_sizes=cfg["batch_sizes"],
                max_length=cfg["max_length"],
                csv_path=args.csv,
                task=args.task,
                fold=args.fold,
                n_folds=args.n_folds,
                epochs=args.epochs,
                lr=args.lr,
                warmup_ratio=args.warmup_ratio,
                patience=args.patience,
                seed=args.seed,
                device=args.device,
                metrics_dir=metrics_dir,
                logger=logger,
            )
    else:
        # Determine model_key from checkpoint name
        key_map = {
            "electra": "electra",
            "roberta": "roberta",
            "xlnet": "xlnet",
            "albert": "albert",
        }
        model_key = next(
            (v for k, v in key_map.items() if k in args.checkpoint.lower()), "unknown"
        )
        run_benchmark_for_model(
            checkpoint=args.checkpoint,
            model_key=model_key,
            batch_sizes=args.batch_sizes,
            max_length=args.max_length,
            csv_path=args.csv,
            task=args.task,
            fold=args.fold,
            n_folds=args.n_folds,
            epochs=args.epochs,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            metrics_dir=metrics_dir,
            logger=logger,
        )

    logger.info("")
    logger.info("Benchmark complete. Run plot_results.py --benchmarks to generate charts.")


if __name__ == "__main__":
    main()