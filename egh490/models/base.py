"""Thin wrapper around HuggingFace transformer classifiers.

One class, ``TransformerClassifier``, covers all four base models from
Somers et al. (2021) — ELECTRA, RoBERTa, XLNet, ALBERT — plus any other
HuggingFace sequence-classification model. The wrapper hides the tokeniser
and model boilerplate behind two methods::

    clf = TransformerClassifier("google/electra-small-discriminator", num_labels=2)
    labels = clf.predict(["The signal aliases because fs < 2fmax"])

Why a wrapper rather than calling ``transformers`` directly everywhere?

1. The four base models have slightly different tokeniser quirks (XLNet uses
   SentencePiece, ALBERT needs ``sentencepiece`` installed). ``AutoTokenizer``
   + ``AutoModelForSequenceClassification`` handle those differences uniformly,
   and having that choice made in one place prevents subtle cross-model bugs.
2. The ensemble in ``egh490/models/ensemble.py`` treats all four models as
   interchangeable. That's only true if they share an interface, which this
   class provides.
3. LIME and SHAP need a function that takes ``list[str]`` and returns class
   probabilities as a numpy array. ``predict_proba`` does exactly that —
   the XAI layer wires into it with no transformer-specific knowledge.

The wrapper is inference-only at this stage. Fine-tuning lives in
``egh490/models/trainer.py`` and accepts a ``TransformerClassifier`` to train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from egh490.utils.device import get_device
from egh490.utils.logging import get_logger

logger = get_logger(__name__)


class TransformerClassifier:
    """Inference-ready wrapper around a HuggingFace sequence classifier.

    Parameters
    ----------
    checkpoint
        Either a HuggingFace hub identifier (``"roberta-base"``) or a local
        directory containing a saved model.
    num_labels
        Number of output classes. For validity and confidence tasks this is 2.
    max_length
        Maximum tokenised sequence length. Longer inputs are truncated.
    device
        Override auto-detection. Accepts ``"cpu"``, ``"cuda"``, ``"mps"``,
        or ``None`` to pick the best available.
    revision
        Pin to a specific commit SHA on the HuggingFace hub for reproducibility.
        Defaults to ``"main"`` for initial development; should be pinned to
        an exact SHA before reporting results.

    Notes
    -----
    The PyTorch and transformers imports are deferred to the constructor so
    this module can be imported at collection time even if torch is absent
    (e.g. during smoke tests that stub out the model).
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        num_labels: int = 2,
        max_length: int = 256,
        device: str | None = None,
        revision: str = "main",
    ) -> None:
        # Deferred imports — keep the module importable in lightweight contexts.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self.checkpoint = checkpoint
        self.num_labels = num_labels
        self.max_length = max_length
        self._initial_device = device or get_device()

        logger.info(
            "Loading %s (num_labels=%d, device=%s)",
            checkpoint,
            num_labels,
            self._initial_device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            revision=revision,
        )
        self.model.to(self._initial_device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Device handling
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> str:
        """Return the device the model currently resides on.

        Read dynamically from the model's parameters rather than cached from
        construction time. HuggingFace's ``Trainer`` may silently move the
        model between devices (e.g. CPU → MPS on Apple Silicon) during
        training, and we need ``predict_proba`` to follow the model wherever
        it ended up.
        """
        # ``next(...)`` grabs the first parameter; all parameters of a
        # loaded HF model share a device, so this is a reliable probe.
        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:  # pragma: no cover — models always have params
            return self._initial_device

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_proba(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return class probabilities for a batch of texts.

        Returns an ``(n_texts, num_labels)`` numpy array of probabilities
        summing to 1 along axis 1. This shape is what ``sklearn``, ``lime``,
        and ``shap`` all expect from a classifier.
        """
        if not texts:
            return np.empty((0, self.num_labels), dtype=np.float32)

        torch = self._torch
        device = self.device  # resolve current device once per call
        probs: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = self.model(**encoded).logits
                batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(batch_probs)

        return np.concatenate(probs, axis=0)

    def predict(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return predicted class indices for a batch of texts."""
        return self.predict_proba(texts, batch_size=batch_size).argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save model + tokeniser to a directory for later reloading.

        Some architectures (notably ELECTRA) hold non-contiguous tensors as
        a memory optimisation — the ``embeddings_project.weight`` matrix
        shares storage with another tensor. The safetensors format used by
        ``save_pretrained`` refuses to serialise non-contiguous tensors, so
        we make every parameter contiguous in-place first. This is a no-op
        for parameters that are already contiguous (RoBERTa, XLNet, ALBERT),
        and costs one extra memcpy per affected tensor for ELECTRA.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for parameter in self.model.parameters():
            if not parameter.is_contiguous():
                parameter.data = parameter.data.contiguous()

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved classifier to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        num_labels: int = 2,
        max_length: int = 256,
        device: str | None = None,
    ) -> "TransformerClassifier":
        """Load a previously saved classifier from disk."""
        return cls(
            checkpoint=str(path),
            num_labels=num_labels,
            max_length=max_length,
            device=device,
            revision="main",
        )