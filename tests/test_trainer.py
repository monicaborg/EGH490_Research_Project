"""Tests for ``egh490.models.trainer``.

Two groups:

1. **Unit tests** for ``compute_metrics`` and ``TrainingConfig`` — no torch
   or transformers dependency, run in milliseconds.
2. **Integration tests** marked ``@slow`` that actually fine-tune ELECTRA-small
   and verify the training loop completes, metrics are returned, and
   post-training inference produces valid outputs.

Run the fast subset::

    pytest tests/test_trainer.py -v -m "not slow"

Run everything::

    pytest tests/test_trainer.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from egh490.models.trainer import TrainingConfig, compute_metrics


# ---------------------------------------------------------------------------
# Unit: TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_has_proposal_defaults() -> None:
    """Defaults must match ``configs/base.yaml`` values."""
    cfg = TrainingConfig()
    assert cfg.epochs == 6
    assert cfg.batch_size == 16
    assert cfg.learning_rate == pytest.approx(2.0e-5)
    assert cfg.early_stopping_patience == 2
    assert cfg.seed == 20260413


def test_training_config_accepts_dict_unpack() -> None:
    """Must be constructible from ``configs/base.yaml``'s ``training`` block."""
    config_dict = {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
    }
    cfg = TrainingConfig(**config_dict)
    assert cfg.epochs == 1
    assert cfg.batch_size == 4


# ---------------------------------------------------------------------------
# Unit: compute_metrics
# ---------------------------------------------------------------------------


def _fake_eval_pred(logits: np.ndarray, labels: np.ndarray) -> SimpleNamespace:
    """Mimic HuggingFace's EvalPrediction object."""
    return SimpleNamespace(predictions=logits, label_ids=labels)


def test_compute_metrics_perfect_predictions() -> None:
    logits = np.array([[-2, 2], [2, -2], [-2, 2], [2, -2]], dtype=np.float32)
    labels = np.array([1, 0, 1, 0])

    metrics = compute_metrics(_fake_eval_pred(logits, labels))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong_predictions() -> None:
    logits = np.array([[2, -2], [-2, 2], [2, -2], [-2, 2]], dtype=np.float32)
    labels = np.array([1, 0, 1, 0])

    metrics = compute_metrics(_fake_eval_pred(logits, labels))

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["auc"] == pytest.approx(0.0)


def test_compute_metrics_drops_auc_for_single_class() -> None:
    """AUC is undefined when only one class is present in labels."""
    logits = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=np.float32)
    labels = np.array([1, 1, 1])

    metrics = compute_metrics(_fake_eval_pred(logits, labels))

    assert "auc" not in metrics
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_without_auc_flag() -> None:
    logits = np.array([[-2, 2], [2, -2]], dtype=np.float32)
    labels = np.array([1, 0])

    metrics = compute_metrics(_fake_eval_pred(logits, labels), include_auc=False)

    assert "auc" not in metrics
    assert metrics["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: fine-tune ELECTRA-small
# ---------------------------------------------------------------------------


SMOKE_CHECKPOINT = "google/electra-small-discriminator"


@pytest.mark.slow
def test_fit_completes_and_returns_metrics(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The full training loop must complete and return valid metrics.

    This test proves:
    - Tokenisation works for ELECTRA's tokeniser
    - The HuggingFace Trainer runs gradient updates without error
    - Checkpoint saving succeeds (including the ELECTRA contiguity fix)
    - Evaluation metrics are computed and returned
    - Post-training inference via predict_proba produces valid probabilities

    It does NOT assert that the model learns a specific pattern — that is
    unreliable on a 60-example toy dataset with random initialisation and
    varies across hardware (CPU vs MPS vs CUDA).
    """
    from egh490.models import TransformerClassifier, Trainer, TrainingConfig

    clf = TransformerClassifier(
        SMOKE_CHECKPOINT,
        num_labels=2,
        max_length=32,
        device="cpu",
    )

    train_texts = [
        "this answer is valid and correct",
        "the reasoning is valid here",
        "a valid explanation of the concept",
        "this is wrong and confused",
        "totally wrong answer here",
        "a wrong explanation of the concept",
    ] * 3  # 18 examples — just enough to run a few batches

    train_labels = ([1] * 3 + [0] * 3) * 3

    eval_texts = ["valid answer", "wrong answer"]
    eval_labels = [1, 0]

    cfg = TrainingConfig(
        epochs=2,
        batch_size=4,
        eval_batch_size=4,
        learning_rate=2e-4,
        early_stopping_patience=0,
        warmup_ratio=0.0,
        output_dir=str(tmp_path / "checkpoints"),
        fp16=False,
    )
    trainer = Trainer(clf, cfg)
    metrics = trainer.fit(train_texts, train_labels, eval_texts, eval_labels)

    # ---- Training completed and produced all expected metrics ----
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "loss" in metrics
    assert "runtime" in metrics

    # ---- Post-training inference produces valid probability distributions ----
    probs = clf.predict_proba(eval_texts)
    assert probs.shape == (2, 2), f"Expected (2, 2), got {probs.shape}"
    assert probs.dtype in (np.float32, np.float64)
    assert (probs >= 0).all(), "Probabilities must be non-negative"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1"

    # ---- predict returns integer class indices ----
    preds = clf.predict(eval_texts)
    assert preds.shape == (2,)
    assert preds.dtype.kind in "iu"
    assert ((preds >= 0) & (preds < 2)).all()


@pytest.mark.slow
def test_fit_runs_without_eval_set(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Training should work even when no eval set is provided."""
    from egh490.models import TransformerClassifier, Trainer, TrainingConfig

    clf = TransformerClassifier(
        SMOKE_CHECKPOINT,
        num_labels=2,
        max_length=32,
        device="cpu",
    )

    cfg = TrainingConfig(
        epochs=1,
        batch_size=4,
        warmup_ratio=0.0,
        output_dir=str(tmp_path / "checkpoints_no_eval"),
        fp16=False,
    )
    trainer = Trainer(clf, cfg)
    metrics = trainer.fit(
        train_texts=["hello world", "foo bar", "baz qux", "another one"],
        train_labels=[0, 1, 0, 1],
    )

    # No eval set => empty metrics dict
    assert metrics == {}