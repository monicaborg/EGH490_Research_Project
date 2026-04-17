"""Majority-vote ensemble of ``TransformerClassifier`` instances.

Somers et al. (2021) combined ELECTRA-small, RoBERTa-base, XLNet-base, and
ALBERT-base-v2 via majority vote to achieve 96.72% validity accuracy and
97.21% confidence accuracy. This module replicates that strategy and also
supports soft voting (averaged probabilities) as a comparison.

The ``Ensemble`` class exposes the same ``predict`` / ``predict_proba``
interface as ``TransformerClassifier``, so the XAI layer (LIME, SHAP) can
explain either a single model or the full ensemble without code changes.

Example
-------
>>> from egh490.models import TransformerClassifier, Ensemble
>>> models = [
...     TransformerClassifier("google/electra-small-discriminator"),
...     TransformerClassifier("roberta-base"),
...     TransformerClassifier("xlnet-base-cased"),
...     TransformerClassifier("albert-base-v2"),
... ]
>>> ensemble = Ensemble(models, strategy="hard")
>>> ensemble.predict(["The signal aliases because fs < 2fmax"])
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from egh490.models.base import TransformerClassifier
from egh490.utils.logging import get_logger

logger = get_logger(__name__)


class Ensemble:
    """Combine multiple classifiers into a single voting classifier.

    Parameters
    ----------
    classifiers
        Two or more trained ``TransformerClassifier`` instances. All must
        share the same ``num_labels``.
    strategy
        ``"hard"`` — each model votes for a class, majority wins. Ties are
        broken by the model with the highest confidence on the disputed
        instance (i.e. the single model whose winning-class probability is
        highest).

        ``"soft"`` — average the probability distributions across all models,
        then take the argmax. Often slightly better because it uses
        confidence information rather than discarding it into a binary vote.

    Raises
    ------
    ValueError
        If fewer than two classifiers are provided or if ``num_labels``
        differs across classifiers.
    """

    def __init__(
        self,
        classifiers: Sequence[TransformerClassifier],
        *,
        strategy: Literal["hard", "soft"] = "hard",
    ) -> None:
        if len(classifiers) < 2:
            raise ValueError(
                f"Ensemble requires at least 2 classifiers, got {len(classifiers)}"
            )

        num_labels = {clf.num_labels for clf in classifiers}
        if len(num_labels) > 1:
            raise ValueError(
                f"All classifiers must share num_labels, got {num_labels}"
            )

        if strategy not in ("hard", "soft"):
            raise ValueError(f"strategy must be 'hard' or 'soft', got {strategy!r}")

        self.classifiers = list(classifiers)
        self.num_labels = classifiers[0].num_labels
        self.strategy = strategy

        logger.info(
            "Built ensemble with %d classifiers, strategy=%s",
            len(self.classifiers),
            self.strategy,
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_proba(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return class probabilities for a batch of texts.

        For **soft** voting, this is the mean of per-model probability
        distributions — a proper probability distribution that sums to 1.

        For **hard** voting, this returns the proportion of models that
        voted for each class (e.g. 3 of 4 voted "valid" → ``[0.25, 0.75]``).
        This is not a calibrated probability but maintains the same shape
        and summing-to-1 property, so LIME/SHAP can consume it identically.
        """
        if not texts:
            return np.empty((0, self.num_labels), dtype=np.float32)

        # Collect predictions from all classifiers: (n_models, n_texts, n_labels)
        all_probs = np.stack(
            [clf.predict_proba(texts, batch_size=batch_size) for clf in self.classifiers],
            axis=0,
        )

        if self.strategy == "soft":
            return all_probs.mean(axis=0)

        # Hard voting: count votes per class, break ties by max confidence.
        return self._hard_vote_probs(all_probs)

    def predict(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return predicted class indices for a batch of texts."""
        return self.predict_proba(texts, batch_size=batch_size).argmax(axis=1)

    # ------------------------------------------------------------------ #
    # Internal: hard voting with confidence-based tie breaking
    # ------------------------------------------------------------------ #

    def _hard_vote_probs(self, all_probs: np.ndarray) -> np.ndarray:
        """Convert per-model probabilities into hard-vote proportions.

        Parameters
        ----------
        all_probs
            Shape ``(n_models, n_texts, n_labels)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_texts, n_labels)`` — vote proportions per class.
            In the case of a tie, the tied class with the highest single-model
            confidence wins (gets a fractionally higher proportion).
        """
        n_models, n_texts, n_labels = all_probs.shape

        # Per-model hard predictions: (n_models, n_texts)
        hard_preds = all_probs.argmax(axis=2)

        # Count votes for each class: (n_texts, n_labels)
        vote_counts = np.zeros((n_texts, n_labels), dtype=np.float32)
        for model_idx in range(n_models):
            for text_idx in range(n_texts):
                vote_counts[text_idx, hard_preds[model_idx, text_idx]] += 1

        # Check for ties and break them using max confidence.
        for text_idx in range(n_texts):
            max_votes = vote_counts[text_idx].max()
            tied_classes = np.where(vote_counts[text_idx] == max_votes)[0]

            if len(tied_classes) > 1:
                # For each tied class, find the highest confidence any model
                # assigned to it. The class with the highest such confidence
                # gets a tiny boost to break the tie.
                best_class = -1
                best_confidence = -1.0
                for cls in tied_classes:
                    # Models that voted for this class
                    for model_idx in range(n_models):
                        if hard_preds[model_idx, text_idx] == cls:
                            conf = all_probs[model_idx, text_idx, cls]
                            if conf > best_confidence:
                                best_confidence = conf
                                best_class = cls

                # Tiny nudge so argmax picks the winner.
                vote_counts[text_idx, best_class] += 0.01

        # Normalise to proportions that sum to 1.
        row_sums = vote_counts.sum(axis=1, keepdims=True)
        return vote_counts / row_sums