"""Tests for ``egh490.data``.

These tests run against the synthetic CSV in ``data/synthetic/`` and
verify that the DataModule correctly loads, validates, encodes labels,
and produces stratified k-fold splits. No model downloads needed — all
tests are fast.

Run with::

    pytest tests/test_data.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egh490.data import DataModule, get_task_config, VALIDITY_LABELS, CONFIDENCE_LABELS

# Path to synthetic data — relative to the repo root
SYNTHETIC_CSV = Path("data/synthetic/synthetic_responses.csv")


@pytest.fixture
def skip_if_no_synthetic():
    """Skip tests if synthetic CSV doesn't exist (e.g. in CI without data)."""
    if not SYNTHETIC_CSV.is_file():
        pytest.skip(f"Synthetic data not found at {SYNTHETIC_CSV}")


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_get_task_config_validity():
    cfg = get_task_config("validity")
    assert cfg["label_column"] == "validity"
    assert cfg["num_labels"] == 2
    assert cfg["label_map"] == {"incorrect": 0, "correct": 1}


def test_get_task_config_confidence():
    cfg = get_task_config("confidence")
    assert cfg["label_column"] == "confidence"
    assert cfg["num_labels"] == 2
    assert cfg["label_map"] == {"low": 0, "high": 1}


def test_get_task_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task_config("nonexistent")


# ---------------------------------------------------------------------------
# DataModule loading tests
# ---------------------------------------------------------------------------


def test_datamodule_loads_synthetic(skip_if_no_synthetic):
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity")

    assert len(dm) > 0
    assert len(dm.texts) == len(dm.labels)
    assert all(isinstance(t, str) for t in dm.texts)
    assert all(label in (0, 1) for label in dm.labels)


def test_datamodule_loads_confidence_task(skip_if_no_synthetic):
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="confidence")

    assert len(dm) > 0
    assert all(label in (0, 1) for label in dm.labels)


def test_datamodule_removes_single_word_responses(skip_if_no_synthetic):
    """Single-word responses should be filtered out per Somers et al."""
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity")

    # Every remaining text should have at least one space (multi-word)
    for text in dm.texts:
        assert " " in text.strip(), f"Single-word response found: {text!r}"


def test_datamodule_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        DataModule(csv_path="does/not/exist.csv", task="validity")


def test_datamodule_rejects_unknown_task(skip_if_no_synthetic):
    with pytest.raises(ValueError, match="Unknown task"):
        DataModule(csv_path=SYNTHETIC_CSV, task="nonexistent")


# ---------------------------------------------------------------------------
# K-fold split tests
# ---------------------------------------------------------------------------


def test_kfold_produces_correct_number_of_folds(skip_if_no_synthetic):
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity", n_folds=5)
    folds = list(dm.kfold_splits())

    assert len(folds) == 5


def test_kfold_indices_are_disjoint_and_complete(skip_if_no_synthetic):
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity", n_folds=5)
    all_test_indices: list[int] = []

    for train_idx, test_idx in dm.kfold_splits():
        # Train and test must not overlap within a fold
        assert len(set(train_idx) & set(test_idx)) == 0
        # Together they must cover the full dataset
        assert len(train_idx) + len(test_idx) == len(dm)
        all_test_indices.extend(test_idx)

    # Across all folds, every index appears exactly once as test
    assert sorted(all_test_indices) == list(range(len(dm)))


def test_kfold_is_stratified(skip_if_no_synthetic):
    """Each fold's test set should have roughly the same class balance."""
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity", n_folds=5)
    labels_array = np.array(dm.labels)
    overall_positive_rate = labels_array.mean()

    for train_idx, test_idx in dm.kfold_splits():
        test_labels = labels_array[test_idx]
        fold_positive_rate = test_labels.mean()
        # Stratified split should keep positive rate within 15% of overall
        assert abs(fold_positive_rate - overall_positive_rate) < 0.15, (
            f"Fold positive rate {fold_positive_rate:.2f} too far from "
            f"overall {overall_positive_rate:.2f}"
        )


def test_kfold_is_deterministic(skip_if_no_synthetic):
    """Same seed must produce identical splits."""
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity", seed=42)
    folds_a = [(t.copy(), v.copy()) for t, v in dm.kfold_splits()]

    dm2 = DataModule(csv_path=SYNTHETIC_CSV, task="validity", seed=42)
    folds_b = [(t.copy(), v.copy()) for t, v in dm2.kfold_splits()]

    for (ta, va), (tb, vb) in zip(folds_a, folds_b):
        assert ta == tb
        assert va == vb


def test_kfold_different_seed_gives_different_splits(skip_if_no_synthetic):
    dm1 = DataModule(csv_path=SYNTHETIC_CSV, task="validity", seed=1)
    dm2 = DataModule(csv_path=SYNTHETIC_CSV, task="validity", seed=999)

    folds1 = list(dm1.kfold_splits())
    folds2 = list(dm2.kfold_splits())

    # At least one fold should differ
    any_different = any(
        f1[1] != f2[1] for f1, f2 in zip(folds1, folds2)
    )
    assert any_different, "Different seeds produced identical splits"


# ---------------------------------------------------------------------------
# Convenience accessor tests
# ---------------------------------------------------------------------------


def test_get_texts_and_labels(skip_if_no_synthetic):
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity")
    indices = [0, 1, 2]
    texts, labels = dm.get_texts_and_labels(indices)

    assert len(texts) == 3
    assert len(labels) == 3
    assert texts[0] == dm.texts[0]
    assert labels[0] == dm.labels[0]


def test_get_dataframe_returns_copy(skip_if_no_synthetic):
    dm = DataModule(csv_path=SYNTHETIC_CSV, task="validity")
    df = dm.get_dataframe()

    # Mutating the returned dataframe should not affect the DataModule
    df.drop(df.index, inplace=True)
    assert len(dm) > 0