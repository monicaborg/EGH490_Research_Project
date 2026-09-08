"""LIME explanations for transformer classifiers and ensembles.

LIME (Local Interpretable Model-agnostic Explanations; Ribeiro, Singh &
Guestrin, 2016) explains a single prediction by perturbing the input —
here, by randomly removing words from the response — and observing how the
model's output probability changes. It then fits a simple linear model to
those perturbations, and the linear model's coefficients become the
per-word importance scores.

Because LIME only needs a function mapping ``list[str] -> probability array``,
it works identically on a single ``TransformerClassifier`` or the full
``Ensemble`` — both expose ``predict_proba``. Nothing in this module is
transformer-specific; it treats the model as a black box, which is exactly
what makes the explanation model-agnostic.

Example
-------
>>> from egh490.models import TransformerClassifier
>>> from egh490.xai import LimeExplainer
>>> clf = TransformerClassifier.load("outputs/checkpoints/roberta/final")
>>> explainer = LimeExplainer(clf, class_names=["incorrect", "correct"])
>>> result = explainer.explain("the signal aliases because fs is below 2 fmax")
>>> result.top_features(5)
[('aliases', 0.42), ('fmax', 0.31), ('below', 0.18), ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from egh490.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LimeExplanation:
    """Result of explaining one response with LIME.

    Attributes
    ----------
    text
        The response that was explained.
    predicted_label
        The class index the model predicted (argmax of probabilities).
    predicted_proba
        The full probability vector for the response.
    class_names
        Human-readable class names, indexed by label.
    feature_weights
        List of ``(word, weight)`` pairs. Positive weight pushes toward the
        explained class; negative pushes away. Ordered as LIME returned them.
    explained_class
        The class index the feature weights are explaining (defaults to the
        predicted class).
    """

    text: str
    predicted_label: int
    predicted_proba: np.ndarray
    class_names: list[str]
    feature_weights: list[tuple[str, float]]
    explained_class: int

    def top_features(self, n: int = 10, *, by_magnitude: bool = True) -> list[tuple[str, float]]:
        """Return the ``n`` most influential words.

        Parameters
        ----------
        n
            How many features to return.
        by_magnitude
            If True (default), rank by absolute weight so the strongest
            evidence in *either* direction surfaces. If False, return the
            most positive weights only (strongest evidence *for* the class).
        """
        if by_magnitude:
            ranked = sorted(self.feature_weights, key=lambda kv: abs(kv[1]), reverse=True)
        else:
            ranked = sorted(self.feature_weights, key=lambda kv: kv[1], reverse=True)
        return ranked[:n]

    def as_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "text": self.text,
            "predicted_label": int(self.predicted_label),
            "predicted_class_name": self.class_names[self.predicted_label],
            "predicted_proba": [float(p) for p in self.predicted_proba],
            "explained_class": int(self.explained_class),
            "class_names": self.class_names,
            "feature_weights": [[w, float(s)] for w, s in self.feature_weights],
        }


class LimeExplainer:
    """Generate LIME explanations for a classifier or ensemble.

    Parameters
    ----------
    model
        Any object exposing ``predict_proba(list[str]) -> np.ndarray`` of
        shape ``(n, num_labels)``. Both ``TransformerClassifier`` and
        ``Ensemble`` qualify.
    class_names
        Human-readable names indexed by label, e.g. ``["incorrect", "correct"]``.
    num_samples
        Number of perturbed samples LIME generates per explanation. More
        samples give a more stable explanation at higher compute cost.
        Ribeiro et al. use ~5000 for text; we default to 1000 to keep
        interactive use responsive and raise it for final reporting.
    bow
        Whether LIME treats the text as a bag of words (removing a word
        everywhere it occurs) or as a positional sequence. Bag-of-words is
        the standard choice for text and is more stable.
    random_state
        Seed for LIME's internal perturbation sampler, for reproducibility.
    """

    def __init__(
        self,
        model,
        *,
        class_names: Sequence[str],
        num_samples: int = 1000,
        bow: bool = True,
        random_state: int = 20260413,
        ngram: int = 1,
    ) -> None:
        # Deferred import so the package is importable without lime installed.
        from lime.lime_text import LimeTextExplainer

        self.model = model
        self.class_names = list(class_names)
        self.num_samples = num_samples
        self.random_state = random_state
        self.ngram = ngram

        # Bigram mode uses a join/unjoin scheme. LIME can only perturb
        # contiguous, removable tokens, so we present it overlapping bigrams
        # joined with a rare separator (e.g. "shortest~period period~highest").
        # LIME treats each joined bigram as one removable feature; our predict
        # wrapper reconstructs readable text before the model sees it. This
        # gives genuine phrase-level attribution without breaking LIME's
        # indexed-string reconstruction (the naive unigram+bigram token list
        # approach fails because bigram tokens don't map to source spans).
        if ngram >= 2:
            self._explainer = LimeTextExplainer(
                class_names=self.class_names,
                bow=False,
                random_state=random_state,
                split_expression=r"\s+",
            )
        else:
            self._explainer = LimeTextExplainer(
                class_names=self.class_names,
                bow=bow,
                random_state=random_state,
            )

    # ------------------------------------------------------------------ #
    # Prediction function passed to LIME
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Bigram join/unjoin helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_bigram_form(text: str) -> str:
        """Convert 'a b c' -> 'a~b b~c' (overlapping bigrams, ~-joined).

        LIME sees each ~-joined bigram as one removable token; the model
        never sees this form (see _from_bigram_form).
        """
        import re
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        if len(words) < 2:
            return text
        return " ".join(f"{words[i]}~{words[i+1]}" for i in range(len(words) - 1))

    @staticmethod
    def _from_bigram_form(text: str) -> str:
        """Reconstruct readable text from ~-joined overlapping bigrams.

        'a~b b~c' -> 'a b c'. When LIME removes a bigram token during
        perturbation, the reconstruction naturally drops those words.
        """
        toks = text.split()
        words: list[str] = []
        for t in toks:
            parts = t.split("~")
            if not words:
                words.extend(parts)
            else:
                words.append(parts[-1])
        return " ".join(words)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Wrap the model's predict_proba for LIME.

        LIME hands us a list of perturbed strings and expects an
        ``(n, num_labels)`` probability array back. In bigram mode the
        strings are in ~-joined form, so we reconstruct readable text
        before the model sees them.
        """
        if self.ngram >= 2:
            texts = [self._from_bigram_form(t) for t in texts]
        return self.model.predict_proba(texts)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def explain(
        self,
        text: str,
        *,
        explained_class: int | None = None,
        num_features: int = 20,
    ) -> LimeExplanation:
        """Explain a single response.

        Parameters
        ----------
        text
            The student response to explain.
        explained_class
            Which class to explain the evidence for. Defaults to the model's
            predicted class, which is usually what you want ("why did the
            model call this correct?").
        num_features
            Maximum number of words LIME assigns weights to.
        """
        proba = self.model.predict_proba([text])[0]
        predicted_label = int(np.argmax(proba))
        target = predicted_label if explained_class is None else explained_class

        # In bigram mode, present LIME the ~-joined overlapping-bigram form.
        lime_input = self._to_bigram_form(text) if self.ngram >= 2 else text

        # LIME cannot perturb a document with fewer than 2 tokens (it samples
        # from randint(1, doc_size+1), which fails when doc_size < 2). Short
        # responses (empty, single word, or a single bigram) are returned with
        # empty feature weights rather than crashing the batch. These are
        # inherently unexplainable and correctly excluded by coverage anyway.
        token_count = len(lime_input.split())
        if token_count < 2:
            return LimeExplanation(
                text=text,
                predicted_label=predicted_label,
                predicted_proba=proba,
                class_names=self.class_names,
                feature_weights=[],
                explained_class=target,
            )

        explanation = self._explainer.explain_instance(
            lime_input,
            self._predict_proba,
            labels=(target,),
            num_features=num_features,
            num_samples=self.num_samples,
        )

        feature_weights = explanation.as_list(label=target)

        # In bigram mode, convert the ~-joined feature labels back to readable
        # space-separated phrases (e.g. "shortest~period" -> "shortest period").
        if self.ngram >= 2:
            feature_weights = [
                (feat.replace("~", " "), weight) for feat, weight in feature_weights
            ]

        return LimeExplanation(
            text=text,
            predicted_label=predicted_label,
            predicted_proba=proba,
            class_names=self.class_names,
            feature_weights=feature_weights,
            explained_class=target,
        )

    def explain_batch(
        self,
        texts: Sequence[str],
        *,
        num_features: int = 20,
    ) -> list[LimeExplanation]:
        """Explain multiple responses sequentially.

        LIME has no batch mode — each explanation is independent — so this
        simply loops. Progress is logged every 10 explanations because each
        one triggers ``num_samples`` model calls and the total can be slow.
        """
        results: list[LimeExplanation] = []
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                logger.info("LIME progress: %d / %d", i, len(texts))
            results.append(self.explain(text, num_features=num_features))
        return results