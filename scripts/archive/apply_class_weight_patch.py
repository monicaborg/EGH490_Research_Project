"""One-off patch: adds optional class-weighted loss to the trainer."""
from pathlib import Path

def patch(path, replacements):
    p = Path(path)
    text = p.read_text()
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"FAILED — pattern not found in {path}:\n{old[:80]}...")
        text = text.replace(old, new, 1)
    p.write_text(text)
    print(f"Patched {path}")

# --- egh490/models/trainer.py -------------------------------------------
patch("egh490/models/trainer.py", [
    (
'''    fp16: bool = False
    seed: int = 20260413
    output_dir: str = "outputs/checkpoints/_tmp"
    # Metrics to log. "auc" is dropped automatically for >2-class tasks.''',
'''    fp16: bool = False
    seed: int = 20260413
    output_dir: str = "outputs/checkpoints/_tmp"
    # Per-class loss weights, e.g. [1.0, 6.8] to upweight a minority class.
    # None disables weighting (standard unweighted cross-entropy).
    class_weights: tuple[float, ...] | None = None
    # Metrics to log. "auc" is dropped automatically for >2-class tasks.'''
    ),
    (
'''        callbacks: list[Any] = []
        if eval_ds is not None and cfg.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
            )

        hf_trainer = HFTrainer(
            model=classifier.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=(
                lambda p: compute_metrics(p, include_auc="auc" in cfg.metrics)
            )
            if eval_ds is not None
            else None,
            callbacks=callbacks,
        )''',
'''        callbacks: list[Any] = []
        if eval_ds is not None and cfg.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
            )

        trainer_cls = HFTrainer
        extra_kwargs: dict[str, Any] = {}
        if cfg.class_weights is not None:
            import torch

            weight_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32)

            class WeightedLossTrainer(HFTrainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=weight_tensor.to(logits.device)
                    )
                    loss = loss_fct(logits.view(-1, classifier.num_labels), labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

            trainer_cls = WeightedLossTrainer
            logger.info("Using class-weighted loss: weights=%s", cfg.class_weights)

        hf_trainer = trainer_cls(
            model=classifier.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=(
                lambda p: compute_metrics(p, include_auc="auc" in cfg.metrics)
            )
            if eval_ds is not None
            else None,
            callbacks=callbacks,
            **extra_kwargs,
        )'''
    ),
])

# --- scripts/train_all_models.py ----------------------------------------
patch("scripts/train_all_models.py", [
    (
'''    p.add_argument("--patience",     type=int,   default=2)''',
'''    p.add_argument("--patience",     type=int,   default=2)
    p.add_argument("--class-weighted", action="store_true",
                    help="Use inverse-frequency class weights in the loss "
                         "(helps prevent majority-class collapse on imbalanced CCUs)")'''
    ),
    (
'''    # Class distribution for the chart
    label_counts       = Counter(dm.labels)
    class_distribution = {dm.label_names[k]: v for k, v in sorted(label_counts.items())}
    logger.info("Class distribution (whole dataset): %s", class_distribution)''',
'''    # Class distribution for the chart
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
        logger.info("Class weights (balanced): %s", class_weights)'''
    ),
    (
'''                training_cfg = TrainingConfig(
                    epochs=args.epochs,
                    batch_size=cfg["batch_size"],
                    eval_batch_size=cfg["batch_size"] * 2,
                    learning_rate=args.lr,
                    warmup_ratio=args.warmup_ratio,
                    early_stopping_patience=args.patience,
                    seed=args.seed,''',
'''                training_cfg = TrainingConfig(
                    epochs=args.epochs,
                    batch_size=cfg["batch_size"],
                    eval_batch_size=cfg["batch_size"] * 2,
                    learning_rate=args.lr,
                    warmup_ratio=args.warmup_ratio,
                    early_stopping_patience=args.patience,
                    seed=args.seed,
                    class_weights=class_weights,'''
    ),
])

print("\nAll patches applied successfully.")