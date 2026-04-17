"""Fine-tuning for ``TransformerClassifier``.

Thin wrapper around HuggingFace's ``Trainer`` API. The wrapper exists for
three reasons:

1. Turn an already-constructed ``TransformerClassifier`` into a trained one,
   without callers needing to know about ``TrainingArguments``, data
   collators, or the raw HuggingFace dataset format.
2. Centralise the small handful of training hyperparameters that come from
   ``configs/base.yaml`` (epochs, batch size, learning rate, early-stopping
   patience) so every base model is trained with identical settings by
   default — a requirement for a fair Somers et al. (2021) replication.
3. Compute and log the evaluation metrics the proposal commits to: accuracy,
   precision, recall, F1 (macro), and AUC.

What this module does *not* do
------------------------------
- It does not load data. Callers pass texts and labels directly; the data
  layer (``egh490/data/``) is responsible for turning a corpus into those.
- It does not split data. Cross-validation and train/val/test splits live
  in ``egh490/evaluation/cv.py``, also to be built.
- It does not ensemble. That is ``egh490/models/ensemble.py``.

Example
-------
>>> from egh490.models import TransformerClassifier, Trainer, TrainingConfig
>>> clf = TransformerClassifier("google/electra-small-discriminator")
>>> trainer = Trainer(clf, TrainingConfig(epochs=3))
>>> trainer.fit(
...     train_texts=["valid reasoning", "invalid reasoning"],
...     train_labels=[1, 0],
...     eval_texts=["another valid one"],
...     eval_labels=[1],
... )
>>> clf.predict(["new unseen text"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from egh490.models.base import TransformerClassifier
from egh490.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning.

    Defaults mirror ``configs/base.yaml`` so the dataclass can be constructed
    from a config dict via ``TrainingConfig(**config["training"])``.
    """

    epochs: int = 6
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2
    early_stopping_metric: str = "eval_f1_macro"
    fp16: bool = False
    seed: int = 20260413
    output_dir: str = "outputs/checkpoints/_tmp"
    # Metrics to log. "auc" is dropped automatically for >2-class tasks.
    metrics: tuple[str, ...] = field(
        default_factory=lambda: ("accuracy", "precision", "recall", "f1_macro", "auc")
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: Any, *, include_auc: bool = True) -> dict[str, float]:
    """Compute classification metrics for HuggingFace's ``Trainer``.

    HuggingFace calls this with an ``EvalPrediction`` whose ``predictions``
    are logits (not probabilities) and ``label_ids`` are integers.

    We report accuracy, macro-precision/recall/F1 (robust to class imbalance
    in the minority-class direction, per Somers et al. 2021), and — for
    two-class tasks — AUC over softmax probabilities of the positive class.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

    if include_auc and logits.shape[-1] == 2 and len(np.unique(labels)) == 2:
        # Softmax over the two logits -> probability of the positive class.
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))

    return metrics


# ---------------------------------------------------------------------------
# Contiguity patch — works around a safetensors/ELECTRA/MPS interaction
# ---------------------------------------------------------------------------


def _make_model_contiguous(model: Any) -> None:
    """Force every parameter to own its memory.

    Some architectures (notably ELECTRA) use non-contiguous tensor views as
    a memory-sharing optimisation. The ``safetensors`` format used by
    HuggingFace's checkpoint saver refuses to serialise these. The issue
    is especially visible on Apple Silicon (MPS), where gradient updates
    can leave additional tensors non-contiguous mid-training.

    Calling ``.contiguous()`` is a no-op on already-contiguous tensors, so
    this is safe to call on every architecture and doesn't affect training.
    """
    for parameter in model.parameters():
        if not parameter.is_contiguous():
            parameter.data = parameter.data.contiguous()
    # Buffers can be non-contiguous too (registered state that isn't a
    # trainable parameter — position embeddings, running stats, etc.).
    for buffer in model.buffers():
        if not buffer.is_contiguous():
            buffer.data = buffer.data.contiguous()


def _wrap_save_pretrained(model: Any) -> None:
    """Monkey-patch ``model.save_pretrained`` to run contiguity first.

    The ``TrainerCallback.on_save`` hook fires before checkpoint write on
    some code paths but not all (best-model saves, final saves, and MPS-
    scheduled kernel reshapes can bypass it). Wrapping the model's own
    ``save_pretrained`` method is the narrowest intervention that catches
    every save path, because HuggingFace's ``Trainer`` always ends up
    calling it.

    Safe to call multiple times; the wrapper records itself with a flag
    so subsequent calls are no-ops.
    """
    if getattr(model, "_ears_contiguous_wrapped", False):
        return

    original_save_pretrained = model.save_pretrained

    def save_pretrained(*args: Any, **kwargs: Any) -> Any:
        _make_model_contiguous(model)
        return original_save_pretrained(*args, **kwargs)

    # Preserve the docstring/signature as best we can for debuggability.
    save_pretrained.__doc__ = original_save_pretrained.__doc__

    model.save_pretrained = save_pretrained  # type: ignore[method-assign]
    model._ears_contiguous_wrapped = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Training dataset shim
# ---------------------------------------------------------------------------


class _TokenisedDataset:
    """Minimal PyTorch-style dataset of tokenised texts and integer labels.

    HuggingFace's ``Trainer`` accepts anything that implements ``__len__``
    and ``__getitem__`` returning a dict. Rolling our own 20-line version
    avoids a dependency on the ``datasets`` library for this simple case
    and makes the tokenisation step explicit in the code path.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must be same length"
            )
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Fine-tune a ``TransformerClassifier`` on labelled text.

    The wrapped classifier is mutated in place — after ``fit`` returns, the
    same ``TransformerClassifier`` instance holds the fine-tuned weights.
    """

    def __init__(
        self,
        classifier: TransformerClassifier,
        config: TrainingConfig | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or TrainingConfig()

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        eval_texts: Sequence[str] | None = None,
        eval_labels: Sequence[int] | None = None,
    ) -> dict[str, float]:
        """Fine-tune on training data, optionally with an eval set.

        Parameters
        ----------
        train_texts, train_labels
            Training examples. Labels are integers in ``[0, num_labels)``.
        eval_texts, eval_labels
            Validation set used for early stopping and final metrics. If
            omitted, training runs for the full number of epochs with no
            eval, and an empty metrics dict is returned.

        Returns
        -------
        dict
            Final evaluation metrics on the eval set, or ``{}`` if no eval
            set was provided.
        """
        # Deferred imports — keep this module importable without torch.
        from transformers import (
            EarlyStoppingCallback,
            Trainer as HFTrainer,
            TrainingArguments,
        )

        classifier = self.classifier
        cfg = self.config

        # Ensure every save during training goes through the contiguity fix.
        # Works for all save paths: periodic checkpoints, best-model saves,
        # and final saves.
        _wrap_save_pretrained(classifier.model)

        logger.info(
            "Starting fine-tuning: %d train examples, %s eval examples, %d epochs",
            len(train_texts),
            len(eval_texts) if eval_texts is not None else 0,
            cfg.epochs,
        )

        train_ds = _TokenisedDataset(
            train_texts, train_labels, classifier.tokenizer, classifier.max_length
        )
        eval_ds = None
        if eval_texts is not None and eval_labels is not None:
            eval_ds = _TokenisedDataset(
                eval_texts, eval_labels, classifier.tokenizer, classifier.max_length
            )

        args = TrainingArguments(
            output_dir=str(Path(cfg.output_dir)),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            fp16=cfg.fp16,
            seed=cfg.seed,
            # Evaluation + checkpointing only when we have an eval set.
            eval_strategy="epoch" if eval_ds is not None else "no",
            save_strategy="epoch" if eval_ds is not None else "no",
            load_best_model_at_end=eval_ds is not None,
            metric_for_best_model=cfg.early_stopping_metric if eval_ds is not None else None,
            greater_is_better=True,
            # Keep disk usage bounded during long runs.
            save_total_limit=2,
            # Less noisy logs; full details still captured by our own logger.
            logging_strategy="epoch",
            report_to=[],
            disable_tqdm=False,
        )

        callbacks: list[Any] = []
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
        )

        hf_trainer.train()

        # After training, the in-place model still sits in train mode. Put it
        # back into eval so predict_proba calls immediately afterwards behave.
        classifier.model.eval()

        if eval_ds is not None:
            final_metrics = hf_trainer.evaluate()
            # HuggingFace prefixes metric names with "eval_"; strip for clarity.
            return {k.removeprefix("eval_"): v for k, v in final_metrics.items()}
        return {}