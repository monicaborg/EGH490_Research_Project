"""Train all four transformer models across all 5 folds and save rich metrics.

Saves per-model JSON files containing everything plot_results.py needs:
  - Final accuracy, F1, AUC, precision, recall (mean ± std across folds)
  - Per-epoch training and validation loss (loss curves)
  - Confusion matrix data (aggregatable across folds)
  - Class distribution of the dataset (for the distribution chart)
  - Per-fold timing

Usage
-----
All four models on 2000-row synthetic data::

    python scripts/train_all_models.py --csv data/synthetic/synthetic_responses_2000.csv

All four models on real data::

    python scripts/train_all_models.py --csv data/raw/labelled_responses.csv

Single model only::

    python scripts/train_all_models.py --models albert

Skip models already run::

    python scripts/train_all_models.py --skip-models electra roberta

Results saved to:
    outputs/metrics/<dataset_tag>/<model_key>_<task>_results.json

Then generate all charts with:
    python scripts/plot_results.py --dataset <dataset_tag> --csv <csv_path>
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
# Model configs — batch sizes reflect MacBook memory constraints
# ------------------------------------------------------------------ #

MODEL_CONFIGS = [
    {
        "key":        "electra",
        "checkpoint": "google/electra-small-discriminator",
        "batch_size": 16,
        "max_length": 256,
    },
    {
        "key":        "roberta",
        "checkpoint": "roberta-base",
        "batch_size": 8,
        "max_length": 256,
    },
    {
        "key":        "xlnet",
        "checkpoint": "xlnet-base-cased",
        "batch_size": 4,
        "max_length": 128,
    },
    {
        "key":        "albert",
        "checkpoint": "albert-base-v2",
        "batch_size": 16,
        "max_length": 256,
    },
]


# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train all four transformer models and save rich metrics to JSON."
    )
    # Data
    p.add_argument("--csv",          default="data/synthetic/synthetic_responses.csv",
                   help="Path to labelled CSV (default: 100-row synthetic)")
    p.add_argument("--task",         default="validity", choices=["validity", "confidence"])
    p.add_argument("--n-folds",      type=int, default=5)
    # Training
    p.add_argument("--epochs",       type=int,   default=6)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=2)
    p.add_argument("--class-weighted", action="store_true",
                    help="Use inverse-frequency class weights in the loss "
                         "(helps prevent majority-class collapse on imbalanced CCUs)")
    p.add_argument("--seed",         type=int,   default=20260413)
    # Output
    p.add_argument("--output-dir",   default="outputs/checkpoints")
    p.add_argument("--metrics-dir",  default="outputs/metrics")
    p.add_argument("--save-models",  action="store_true",
                   help="Save fine-tuned model weights after each fold")
    p.add_argument("--device",       default=None,
                   help="Force device: cpu | cuda | mps (default: auto)")
    # Model selection
    p.add_argument("--models",       nargs="*",
                   choices=["electra", "roberta", "xlnet", "albert"], default=None,
                   help="Only train these models (default: all four)")
    p.add_argument("--skip-models",  nargs="*",
                   choices=["electra", "roberta", "xlnet", "albert"], default=[],
                   help="Skip these models")
    # CCU filter — train on one question only (Somers methodology)
    p.add_argument("--ccu",          default=None,
                   help="Filter to a single CCU question, e.g. --ccu ccu3. "
                        "Use this when training on a marked single-CCU CSV. "
                        "If the CSV already contains only one CCU, this flag "
                        "is optional but adds a validation log message.")
    # Column name overrides (in case marked CSVs use different headers)
    p.add_argument("--text-column",  default=None,
                   help="Override the free-text column name (default: q1txr)")
    p.add_argument("--label-column", default=None,
                   help="Override the label column name (default: validity or confidence)")
    return p.parse_args(argv)


# ------------------------------------------------------------------ #
# Loss-curve callback
# ------------------------------------------------------------------ #

class LossCurveCallback:
    """Records train loss and eval loss+accuracy at the end of each epoch.

    Registered as a HuggingFace TrainerCallback via the _Wrapper class
    defined inside fit_with_loss_curve.
    """

    def __init__(self):
        self.train_losses:    list[float] = []
        self.eval_losses:     list[float] = []
        self.eval_accuracies: list[float] = []
        self._pending_train:  float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Training loss is logged first, then eval loss in the same epoch
        if "loss" in logs:
            self._pending_train = logs["loss"]
        if "eval_loss" in logs:
            # Pair train and eval losses for the same epoch
            if self._pending_train is not None:
                self.train_losses.append(self._pending_train)
                self._pending_train = None
            self.eval_losses.append(logs["eval_loss"])
            if "eval_accuracy" in logs:
                self.eval_accuracies.append(logs["eval_accuracy"])

    # Required no-op stubs for the HuggingFace callback interface
    def on_epoch_begin(self,      args, state, control, **kw): pass
    def on_epoch_end(self,        args, state, control, **kw): pass
    def on_train_begin(self,      args, state, control, **kw): pass
    def on_train_end(self,        args, state, control, **kw): pass
    def on_evaluate(self,         args, state, control, **kw): pass
    def on_save(self,             args, state, control, **kw): pass
    def on_step_begin(self,       args, state, control, **kw): pass
    def on_step_end(self,         args, state, control, **kw): pass
    def on_substep_end(self,      args, state, control, **kw): pass
    def on_prediction_step(self,  args, state, control, **kw): pass
    def on_init_end(self,         args, state, control, **kw): pass
    def on_optimizer_step(self,   args, state, control, **kw): pass


# ------------------------------------------------------------------ #
# Core training function (returns metrics + loss callback)
# ------------------------------------------------------------------ #

def fit_with_loss_curve(clf, training_cfg, train_texts, train_labels, eval_texts, eval_labels):
    """Fine-tune clf and return (final_metrics_dict, LossCurveCallback).

    Wraps the HuggingFace Trainer directly so we can inject a custom
    callback for loss-curve recording without modifying the Trainer class.
    """
    from transformers import (
        EarlyStoppingCallback,
        Trainer as HFTrainer,
        TrainerCallback,
        TrainingArguments,
    )
    from egh490.models.trainer import (
        _wrap_save_pretrained,
        _TokenisedDataset,
        compute_metrics,
    )

    # Apply the ELECTRA contiguity fix
    _wrap_save_pretrained(clf.model)

    train_ds = _TokenisedDataset(train_texts, train_labels, clf.tokenizer, clf.max_length)
    eval_ds  = _TokenisedDataset(eval_texts,  eval_labels,  clf.tokenizer, clf.max_length)

    hf_args = TrainingArguments(
        output_dir=training_cfg.output_dir,
        num_train_epochs=training_cfg.epochs,
        per_device_train_batch_size=training_cfg.batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        warmup_ratio=training_cfg.warmup_ratio,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        fp16=training_cfg.fp16,
        seed=training_cfg.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=training_cfg.early_stopping_metric,
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="epoch",
        report_to=[],
        disable_tqdm=False,
    )

    loss_cb = LossCurveCallback()

    class _Wrapper(TrainerCallback):
        """Thin wrapper so LossCurveCallback can receive HF Trainer events."""
        def on_log(self, args, state, control, logs=None, **kw):
            loss_cb.on_log(args, state, control, logs=logs, **kw)

    trainer_cls = HFTrainer
    if getattr(training_cfg, "class_weights", None) is not None:
        import torch

        weight_tensor = torch.tensor(training_cfg.class_weights, dtype=torch.float32)

        class WeightedLossTrainer(HFTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(logits.device))
                loss = loss_fct(logits.view(-1, clf.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedLossTrainer
        from egh490.utils import get_logger
        get_logger("train_all_models").info(
            "fit_with_loss_curve: using class-weighted loss: weights=%s",
            training_cfg.class_weights,
        )

    hf_trainer = trainer_cls(
        model=clf.model,
        args=hf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=lambda p: compute_metrics(p, include_auc=True),
        callbacks=[
            _Wrapper(),
            EarlyStoppingCallback(early_stopping_patience=training_cfg.early_stopping_patience),
        ],
    )

    hf_trainer.train()
    clf.model.eval()
    final = hf_trainer.evaluate()
    # Strip the "eval_" prefix that HuggingFace adds
    final = {k.removeprefix("eval_"): v for k, v in final.items()}
    return final, loss_cb


# ------------------------------------------------------------------ #
# Confusion matrix
# ------------------------------------------------------------------ #

def compute_confusion_matrix(clf, texts, labels, label_names: dict) -> dict:
    """Run inference and return a serialisable confusion matrix dict."""
    from sklearn.metrics import confusion_matrix as sk_cm

    probs      = clf.predict_proba(texts)
    preds      = probs.argmax(axis=1)
    labels_arr = np.array(labels)
    classes    = sorted(label_names.keys())
    cm         = sk_cm(labels_arr, preds, labels=classes)

    return {
        "matrix":      cm.tolist(),
        "classes":     [label_names[c] for c in classes],
        "true_labels": labels_arr.tolist(),
        "pred_labels": preds.tolist(),
    }


# ------------------------------------------------------------------ #
# Save JSON
# ------------------------------------------------------------------ #

def save_metrics(
    *,
    metrics_dir:            Path,
    dataset_tag:            str,
    model_key:              str,
    task:                   str,
    checkpoint:             str,
    batch_size:             int,
    fold_metrics:           dict[int, dict],
    fold_times:             dict[int, float],
    fold_loss_curves:       dict[int, dict],
    fold_confusion_matrices: dict[int, dict],
    class_distribution:     dict,
    total_time:             float,
    label_names:            dict,
) -> Path:
    """Serialise all fold results to a single JSON file."""

    metric_keys = [
        k for k in sorted(next(iter(fold_metrics.values())).keys())
        if isinstance(next(iter(fold_metrics.values()))[k], float)
        and k not in ("runtime", "samples_per_second", "steps_per_second", "epoch")
    ]

    folds_data = {}
    for fn, m in fold_metrics.items():
        folds_data[str(fn)] = {k: m[k] for k in metric_keys if k in m}
        folds_data[str(fn)]["time_seconds"]    = fold_times[fn]
        folds_data[str(fn)]["loss_curve"]      = fold_loss_curves.get(fn, {})
        folds_data[str(fn)]["confusion_matrix"] = fold_confusion_matrices.get(fn, {})

    means = {
        k: float(np.mean([fold_metrics[f][k] for f in fold_metrics if k in fold_metrics[f]]))
        for k in metric_keys
    }
    stds = {
        k: float(np.std([fold_metrics[f][k] for f in fold_metrics if k in fold_metrics[f]]))
        for k in metric_keys
    }

    result = {
        "model_key":          model_key,
        "checkpoint":         checkpoint,
        "task":               task,
        "batch_size":         batch_size,
        "n_folds":            len(fold_metrics),
        "total_time_seconds": total_time,
        "label_names":        {str(k): v for k, v in label_names.items()},
        "class_distribution": class_distribution,
        "folds":              folds_data,
        "mean":               means,
        "std":                stds,
    }

    tag_dir = metrics_dir / dataset_tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    path = tag_dir / f"{model_key}_{task}_results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


# ------------------------------------------------------------------ #
# Memory cleanup helper
# ------------------------------------------------------------------ #

def free_memory(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
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


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv=None):
    args = parse_args(argv)

    from egh490.utils import set_global_seed, get_logger
    from egh490.data import DataModule
    from egh490.models import TransformerClassifier, TrainingConfig

    set_global_seed(args.seed)
    logger = get_logger("train_all_models")

    dataset_tag = Path(args.csv).stem
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Filter model list
    configs = [c for c in MODEL_CONFIGS
               if (args.models is None or c["key"] in args.models)
               and c["key"] not in args.skip_models]

    logger.info("=" * 70)
    logger.info("EGH490 — Train all models")
    logger.info("=" * 70)
    logger.info("CSV:        %s  (tag: %s)", args.csv, dataset_tag)
    logger.info("Task:       %s", args.task)
    logger.info("Folds:      %d", args.n_folds)
    logger.info("Epochs:     %d  (max, early stopping patience=%d)", args.epochs, args.patience)
    logger.info("LR:         %s", args.lr)
    logger.info("Models:     %s", [c["key"] for c in configs])
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    import pandas as pd

    dm_kwargs: dict = dict(
        csv_path=args.csv,
        task=args.task,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    if getattr(args, "text_column", None):
        dm_kwargs["text_column"] = args.text_column

    dm = DataModule(**dm_kwargs)

    # CCU filter — restrict to a single question (mirrors Somers per-question training)
    if getattr(args, "ccu", None):
        from egh490.data.schema import COL_CCU
        df_full = dm.get_dataframe()
        if COL_CCU not in df_full.columns:
            logger.warning(
                "--ccu %s specified but no 'ccuname' column found — ignoring filter",
                args.ccu,
            )
        else:
            available = sorted(df_full[COL_CCU].unique().tolist())
            if args.ccu not in available:
                raise ValueError(
                    f"--ccu {args.ccu!r} not found. Available CCUs: {available}"
                )
            mask      = df_full[COL_CCU] == args.ccu
            ccu_df    = df_full[mask].reset_index(drop=True)
            dm._df    = ccu_df
            dm.texts  = ccu_df[dm.text_col].tolist()
            dm.labels = ccu_df["_label"].tolist()
            logger.info(
                "CCU filter: training on %s only — %d responses",
                args.ccu, len(dm.texts),
            )

        # Update dataset_tag to include CCU name for clean metric file naming
        dataset_tag = f"{dataset_tag}_{args.ccu}"
        logger.info("Dataset tag updated to: %s", dataset_tag)

    folds = list(dm.kfold_splits())

    # Class distribution for the chart
    label_counts       = Counter(dm.labels)
    class_distribution = {dm.label_names[k]: v for k, v in sorted(label_counts.items())}
    logger.info("Class distribution (whole dataset): %s", class_distribution)

    class_weights = None
    if args.class_weighted:
        n_samples = sum(label_counts.values())
        n_classes = len(label_counts)
        class_weights = tuple(
            n_samples / (n_classes * label_counts[k])
            for k in sorted(label_counts)
        )
        logger.info("Class weights (balanced): %s", class_weights)

    # Per-CCU breakdown if column exists
    try:
        df = dm.get_dataframe()
        if "ccuname" in df.columns and "validity" in df.columns:
            ccu_dist = (
                df.groupby(["ccuname", "validity"]).size()
                  .unstack(fill_value=0)
                  .to_dict(orient="index")
            )
            class_distribution["per_ccu"] = {
                ccu: {str(k): int(v) for k, v in counts.items()}
                for ccu, counts in ccu_dist.items()
            }
    except Exception:
        pass

    # ── Train each model ─────────────────────────────────────────────
    overall_start    = time.time()
    overall_summaries = {}

    for cfg in configs:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL: %s   batch=%d   max_length=%d",
                    cfg["checkpoint"], cfg["batch_size"], cfg["max_length"])
        logger.info("=" * 70)

        fold_metrics:             dict[int, dict]  = {}
        fold_times:               dict[int, float] = {}
        fold_loss_curves:         dict[int, dict]  = {}
        fold_confusion_matrices:  dict[int, dict]  = {}
        failed_folds:             list[int]         = []

        for fold_num in range(1, args.n_folds + 1):
            logger.info("  Fold %d / %d", fold_num, args.n_folds)
            train_idx, test_idx    = folds[fold_num - 1]
            train_texts, train_lbs = dm.get_texts_and_labels(train_idx)
            test_texts,  test_lbs  = dm.get_texts_and_labels(test_idx)

            logger.info("    train=%d  test=%d", len(train_texts), len(test_texts))

            clf = None
            try:
                clf = TransformerClassifier(
                    cfg["checkpoint"],
                    num_labels=dm.num_labels,
                    max_length=cfg["max_length"],
                    device=args.device,
                )

                run_name    = f"{args.task}_{cfg['key']}_fold{fold_num}_{dataset_tag}"
                training_cfg = TrainingConfig(
                    epochs=args.epochs,
                    batch_size=cfg["batch_size"],
                    eval_batch_size=cfg["batch_size"] * 2,
                    learning_rate=args.lr,
                    warmup_ratio=args.warmup_ratio,
                    early_stopping_patience=args.patience,
                    seed=args.seed,
                    class_weights=class_weights,
                    output_dir=str(Path(args.output_dir) / run_name),
                    fp16=False,
                )

                t0                  = time.time()
                metrics, loss_cb    = fit_with_loss_curve(
                    clf, training_cfg, train_texts, train_lbs, test_texts, test_lbs
                )
                elapsed             = time.time() - t0

                # ── Store loss curve ──────────────────────────────
                fold_loss_curves[fold_num] = {
                    "train_loss":    loss_cb.train_losses,
                    "eval_loss":     loss_cb.eval_losses,
                    "eval_accuracy": loss_cb.eval_accuracies,
                    "epochs":        list(range(1, len(loss_cb.eval_losses) + 1)),
                }

                # ── Store confusion matrix ────────────────────────
                fold_confusion_matrices[fold_num] = compute_confusion_matrix(
                    clf, test_texts, test_lbs, dm.label_names
                )

                fold_metrics[fold_num] = metrics
                fold_times[fold_num]   = elapsed

                logger.info(
                    "    acc=%.4f  f1=%.4f  auc=%.4f  "
                    "time=%.0fs  epochs_run=%d  "
                    "train_losses=%s",
                    metrics.get("accuracy", 0),
                    metrics.get("f1_macro", 0),
                    metrics.get("auc", 0),
                    elapsed,
                    len(loss_cb.eval_losses),
                    [f"{l:.4f}" for l in loss_cb.train_losses],
                )

                if args.save_models:
                    save_path = Path(args.output_dir) / run_name / "final"
                    clf.save(save_path)
                    logger.info("    Saved model → %s", save_path)

            except Exception as e:
                logger.error("    Fold %d FAILED: %s", fold_num, str(e))
                logger.error(traceback.format_exc())
                failed_folds.append(fold_num)

            finally:
                free_memory(clf)

        if not fold_metrics:
            logger.error("  No folds completed for %s — skipping JSON save.", cfg["key"])
            continue

        total_time = sum(fold_times.values())
        json_path  = save_metrics(
            metrics_dir=metrics_dir,
            dataset_tag=dataset_tag,
            model_key=cfg["key"],
            task=args.task,
            checkpoint=cfg["checkpoint"],
            batch_size=cfg["batch_size"],
            fold_metrics=fold_metrics,
            fold_times=fold_times,
            fold_loss_curves=fold_loss_curves,
            fold_confusion_matrices=fold_confusion_matrices,
            class_distribution=class_distribution,
            total_time=total_time,
            label_names=dm.label_names,
        )
        logger.info("  ✓ Saved → %s", json_path)

        overall_summaries[cfg["key"]] = {
            "mean_accuracy":  float(np.mean([fold_metrics[f]["accuracy"] for f in fold_metrics])),
            "mean_f1":        float(np.mean([fold_metrics[f]["f1_macro"] for f in fold_metrics])),
            "mean_auc":       float(np.mean([fold_metrics[f]["auc"]      for f in fold_metrics])),
            "total_time_min": total_time / 60,
            "failed_folds":   failed_folds,
        }

    # ── Overall summary ───────────────────────────────────────────────
    overall_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL SUMMARY — task=%s  dataset=%s", args.task, dataset_tag)
    logger.info("=" * 70)
    logger.info("%-12s %-12s %-12s %-12s %-12s %-10s",
                "Model", "Accuracy", "F1", "AUC", "Time (min)", "Failed")
    logger.info("-" * 70)
    for key, s in overall_summaries.items():
        logger.info("%-12s %-12.4f %-12.4f %-12.4f %-12.1f %-10s",
                    key, s["mean_accuracy"], s["mean_f1"], s["mean_auc"],
                    s["total_time_min"], str(s["failed_folds"]) if s["failed_folds"] else "none")
    logger.info("-" * 70)
    logger.info("Total wall time: %.1f minutes", overall_time / 60)
    logger.info("")
    logger.info("Generate charts:")
    logger.info("  python scripts/plot_results.py --dataset %s --csv %s",
                dataset_tag, args.csv)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()