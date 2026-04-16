"""Tests for ``egh490.models.base``.

These tests split into two categories:

1. **Unit tests** that don't download anything — they check device selection
   and empty-input handling.
2. **Integration tests** marked ``@pytest.mark.slow`` that download ELECTRA-small
   (~50 MB, one-time) and run real predictions through it. These take longer
   to run but prove the wrapper actually works end-to-end.

Run just the fast tests::

    pytest tests/test_base_model.py -v -m "not slow"

Run everything including the download::

    pytest tests/test_base_model.py -v

The first full run takes ~30 seconds (download + one forward pass).
Subsequent runs use the Hugging Face cache and are much faster.
"""

from __future__ import annotations

import numpy as np
import pytest

from egh490.utils.device import get_device


# ---------------------------------------------------------------------------
# Unit tests — no network, no model download
# ---------------------------------------------------------------------------


def test_get_device_returns_known_value() -> None:
    """Whatever device is picked, it must be one of the three supported."""
    assert get_device() in {"cpu", "cuda", "mps"}


def test_get_device_respects_cpu_preference() -> None:
    """Forcing CPU should always return CPU regardless of hardware."""
    assert get_device(prefer="cpu") == "cpu"


# ---------------------------------------------------------------------------
# Integration tests — download ELECTRA-small and run it
# ---------------------------------------------------------------------------

# The smallest model in the Somers et al. (2021) ensemble. ~50 MB on disk,
# runs in <1s per forward pass on a laptop CPU. Used as the canonical
# "does the whole stack work" check throughout the test suite.
SMOKE_CHECKPOINT = "google/electra-small-discriminator"


@pytest.fixture(scope="module")
def smoke_classifier():  # type: ignore[no-untyped-def]
    """Load ELECTRA-small once and share it across tests in this module."""
    from egh490.models import TransformerClassifier

    return TransformerClassifier(
        SMOKE_CHECKPOINT,
        num_labels=2,
        max_length=64,
        device="cpu",  # force CPU so the test runs identically everywhere
    )


@pytest.mark.slow
def test_predict_proba_returns_valid_distribution(smoke_classifier) -> None:  # type: ignore[no-untyped-def]
    """Each row must be a valid probability distribution over 2 classes."""
    texts = [
        "The sampled signal aliases because the sampling frequency is too low.",
        "I think the answer is maybe correct.",
    ]
    probs = smoke_classifier.predict_proba(texts)

    assert probs.shape == (2, 2), f"Expected (2, 2), got {probs.shape}"
    assert probs.dtype in (np.float32, np.float64)
    # Probabilities are non-negative and sum to 1 along the class axis.
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.slow
def test_predict_returns_class_indices(smoke_classifier) -> None:  # type: ignore[no-untyped-def]
    """``predict`` should return integers in ``[0, num_labels)``."""
    labels = smoke_classifier.predict(["hello world", "aliasing and convolution"])
    assert labels.shape == (2,)
    assert labels.dtype.kind in "iu"  # signed or unsigned int
    assert ((labels >= 0) & (labels < 2)).all()


@pytest.mark.slow
def test_predict_proba_handles_empty_input(smoke_classifier) -> None:  # type: ignore[no-untyped-def]
    """An empty input should return an empty (0, num_labels) array."""
    probs = smoke_classifier.predict_proba([])
    assert probs.shape == (0, 2)


@pytest.mark.slow
def test_predict_proba_batches_correctly(smoke_classifier) -> None:  # type: ignore[no-untyped-def]
    """Results must not depend on batch boundaries."""
    texts = [f"example response number {i}" for i in range(5)]

    # Force all-in-one vs. split batches — should produce identical output.
    one_batch = smoke_classifier.predict_proba(texts, batch_size=16)
    split_batch = smoke_classifier.predict_proba(texts, batch_size=2)

    np.testing.assert_allclose(one_batch, split_batch, atol=1e-5)


@pytest.mark.slow
def test_save_and_load_roundtrip(smoke_classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A saved and reloaded model must produce identical predictions."""
    from egh490.models import TransformerClassifier

    target = tmp_path / "electra_small_snapshot"
    smoke_classifier.save(target)

    reloaded = TransformerClassifier.load(target, num_labels=2, max_length=64, device="cpu")

    text = ["convolution is commutative for LTI systems"]
    np.testing.assert_allclose(
        smoke_classifier.predict_proba(text),
        reloaded.predict_proba(text),
        atol=1e-5,
    )