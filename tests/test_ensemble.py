"""Tests for ``egh490.models.ensemble``.

Three groups:

1. **Unit tests** for voting logic — use a mock classifier that returns
   pre-set probabilities, so no model downloads or torch required.
2. **Validation tests** for construction errors (too few models, mismatched
   num_labels, bad strategy).
3. **Integration test** marked ``@slow`` that builds an ensemble from two
   real ELECTRA instances and runs prediction.

Run the fast subset::

    pytest tests/test_ensemble.py -v -m "not slow"

Run everything::

    pytest tests/test_ensemble.py -v
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from egh490.models.ensemble import Ensemble


# ---------------------------------------------------------------------------
# Mock classifier for unit tests
# ---------------------------------------------------------------------------


class _MockClassifier:
    """Tiny stand-in for TransformerClassifier that returns fixed probs."""

    def __init__(self, probs: np.ndarray, num_labels: int = 2) -> None:
        """``probs`` shape: ``(n_texts, num_labels)``."""
        self.num_labels = num_labels
        self._probs = probs

    def predict_proba(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        return self._probs


# ---------------------------------------------------------------------------
# Unit: construction validation
# ---------------------------------------------------------------------------


def test_ensemble_rejects_fewer_than_two_classifiers() -> None:
    mock = _MockClassifier(np.array([[0.4, 0.6]]))
    with pytest.raises(ValueError, match="at least 2"):
        Ensemble([mock])


def test_ensemble_rejects_mismatched_num_labels() -> None:
    a = _MockClassifier(np.array([[0.4, 0.6]]), num_labels=2)
    b = _MockClassifier(np.array([[0.3, 0.3, 0.4]]), num_labels=3)
    with pytest.raises(ValueError, match="num_labels"):
        Ensemble([a, b])


def test_ensemble_rejects_invalid_strategy() -> None:
    a = _MockClassifier(np.array([[0.4, 0.6]]))
    b = _MockClassifier(np.array([[0.3, 0.7]]))
    with pytest.raises(ValueError, match="strategy"):
        Ensemble([a, b], strategy="majority")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit: soft voting
# ---------------------------------------------------------------------------


def test_soft_voting_averages_probabilities() -> None:
    """Soft vote = mean of per-model probability distributions."""
    a = _MockClassifier(np.array([[0.2, 0.8], [0.9, 0.1]]))
    b = _MockClassifier(np.array([[0.4, 0.6], [0.7, 0.3]]))

    ens = Ensemble([a, b], strategy="soft")
    probs = ens.predict_proba(["x", "y"])

    expected = np.array([[0.3, 0.7], [0.8, 0.2]])
    np.testing.assert_allclose(probs, expected, atol=1e-6)


def test_soft_voting_predict_returns_argmax() -> None:
    a = _MockClassifier(np.array([[0.2, 0.8]]))
    b = _MockClassifier(np.array([[0.4, 0.6]]))

    ens = Ensemble([a, b], strategy="soft")
    preds = ens.predict(["x"])

    assert preds.tolist() == [1]


# ---------------------------------------------------------------------------
# Unit: hard voting
# ---------------------------------------------------------------------------


def test_hard_voting_clear_majority() -> None:
    """3 of 4 models say class 1 → ensemble predicts class 1."""
    a = _MockClassifier(np.array([[0.3, 0.7]]))  # votes 1
    b = _MockClassifier(np.array([[0.4, 0.6]]))  # votes 1
    c = _MockClassifier(np.array([[0.8, 0.2]]))  # votes 0
    d = _MockClassifier(np.array([[0.2, 0.8]]))  # votes 1

    ens = Ensemble([a, b, c, d], strategy="hard")
    preds = ens.predict(["x"])

    assert preds.tolist() == [1]


def test_hard_voting_proportions_sum_to_one() -> None:
    a = _MockClassifier(np.array([[0.3, 0.7]]))
    b = _MockClassifier(np.array([[0.6, 0.4]]))

    ens = Ensemble([a, b], strategy="hard")
    probs = ens.predict_proba(["x"])

    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_hard_voting_tie_broken_by_confidence() -> None:
    """With 2 models split 1-1, the model with higher confidence wins."""
    # Model A: votes class 0 with 0.60 confidence
    a = _MockClassifier(np.array([[0.6, 0.4]]))
    # Model B: votes class 1 with 0.95 confidence
    b = _MockClassifier(np.array([[0.05, 0.95]]))

    ens = Ensemble([a, b], strategy="hard")
    preds = ens.predict(["x"])

    # Model B's 0.95 confidence beats Model A's 0.60 → class 1 wins.
    assert preds.tolist() == [1]


def test_hard_voting_tie_broken_other_direction() -> None:
    """Verify tie-breaking works in the other direction too."""
    # Model A: votes class 0 with 0.99 confidence
    a = _MockClassifier(np.array([[0.99, 0.01]]))
    # Model B: votes class 1 with 0.55 confidence
    b = _MockClassifier(np.array([[0.45, 0.55]]))

    ens = Ensemble([a, b], strategy="hard")
    preds = ens.predict(["x"])

    # Model A's 0.99 confidence beats Model B's 0.55 → class 0 wins.
    assert preds.tolist() == [0]


def test_hard_voting_multiple_texts() -> None:
    """Ensemble handles batches of multiple texts correctly."""
    a = _MockClassifier(np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]]))
    b = _MockClassifier(np.array([[0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]))
    c = _MockClassifier(np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]))

    ens = Ensemble([a, b, c], strategy="hard")
    preds = ens.predict(["x", "y", "z"])

    # text 0: all 3 vote class 1 → 1
    # text 1: all 3 vote class 0 → 0
    # text 2: a=1, b=0, c=1 → majority 1
    assert preds.tolist() == [1, 0, 1]


def test_empty_input_returns_empty_array() -> None:
    a = _MockClassifier(np.array([[0.3, 0.7]]))
    b = _MockClassifier(np.array([[0.4, 0.6]]))

    ens = Ensemble([a, b], strategy="soft")
    probs = ens.predict_proba([])

    assert probs.shape == (0, 2)


# ---------------------------------------------------------------------------
# Integration: real ELECTRA instances
# ---------------------------------------------------------------------------


SMOKE_CHECKPOINT = "google/electra-small-discriminator"


@pytest.mark.slow
def test_ensemble_with_real_models() -> None:
    """Build an ensemble from two real ELECTRA instances and run prediction.

    Uses two copies of the same checkpoint (not four different architectures)
    to keep the download small. The test proves the ensemble correctly wires
    into real TransformerClassifier instances — the voting logic is already
    proven by the unit tests above.
    """
    from egh490.models import TransformerClassifier

    clf_a = TransformerClassifier(
        SMOKE_CHECKPOINT, num_labels=2, max_length=32, device="cpu"
    )
    clf_b = TransformerClassifier(
        SMOKE_CHECKPOINT, num_labels=2, max_length=32, device="cpu"
    )

    ens = Ensemble([clf_a, clf_b], strategy="hard")
    texts = ["The signal aliases because fs is too low", "I have no idea"]

    probs = ens.predict_proba(texts)
    assert probs.shape == (2, 2)
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = ens.predict(texts)
    assert preds.shape == (2,)
    assert ((preds >= 0) & (preds < 2)).all()

    # Soft voting should also work on the same ensemble objects.
    ens_soft = Ensemble([clf_a, clf_b], strategy="soft")
    probs_soft = ens_soft.predict_proba(texts)
    assert probs_soft.shape == (2, 2)
    assert np.allclose(probs_soft.sum(axis=1), 1.0, atol=1e-5)