"""SHAP explanations for transformer classifiers and ensembles.

SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) attributes a
model's prediction to its input features using Shapley values from
cooperative game theory. Each word receives a "fair share" of the credit
(or blame) for the prediction, with a theoretical guarantee that the word
attributions sum to the difference between the prediction and a baseline
expectation. This additive consistency is SHAP's key advantage over LIME,
whose linear surrogate is only a local approximation.

For text transformers, SHAP's ``Explainer`` with a ``Text`` masker perturbs
the input by masking tokens and measures the marginal contribution of each.
As with LIME, the model is treated as a black box through its
``predict_proba`` interface, so this works identically on a single
``TransformerClassifier`` or the full ``Ensemble``.

SHAP is heavier than LIME per explanation but produces attributions that are
consistent across a corpus, which makes it the better choice for the
global / corpus-wide pattern analysis in a later project phase.

Example
-------
>>> from egh490.models import TransformerClassifier
>>> from egh490.xai import ShapExplainer
>>> clf = TransformerClassifier.load("outputs/checkpoints/roberta/final")
>>> explainer = ShapExplainer(clf, class_names=["incorrect", "correct"])
>>> result = explainer.explain("the signal aliases because fs is below 2 fmax")
>>> result.top_features(5)
[('aliases', 0.38), ('fmax', 0.29), ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from egh490.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ShapExplanation:
    """Result of explaining one response with SHAP.

    Attributes
    ----------
    text
        The response that was explained.
    predicted_label
        The class index the model predicted.
    predicted_proba
        The full probability vector.
    class_names
        Human-readable class names, indexed by label.
    tokens
        The tokens SHAP attributed values to (as segmented by the masker).
    shap_values
        Per-token SHAP values for the explained class, aligned with ``tokens``.
    base_value
        The model's expected output over the background — the value from
        which the SHAP contributions are measured.
    explained_class
        The class index the SHAP values explain.
    """

    text: str
    predicted_label: int
    predicted_proba: np.ndarray
    class_names: list[str]
    tokens: list[str]
    shap_values: np.ndarray
    base_value: float
    explained_class: int

    def feature_weights(self) -> list[tuple[str, float]]:
        """Return ``(token, shap_value)`` pairs aligned by position."""
        return list(zip(self.tokens, [float(v) for v in self.shap_values]))

    def top_features(self, n: int = 10, *, by_magnitude: bool = True) -> list[tuple[str, float]]:
        """Return the ``n`` most influential tokens.

        Whitespace-only tokens are dropped since they carry no linguistic
        meaning even when the masker assigns them a small value.
        """
        pairs = [(t, s) for t, s in self.feature_weights() if t.strip()]
        if by_magnitude:
            ranked = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)
        else:
            ranked = sorted(pairs, key=lambda kv: kv[1], reverse=True)
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
            "base_value": float(self.base_value),
            "tokens": self.tokens,
            "shap_values": [float(v) for v in self.shap_values],
        }


class ShapExplainer:
    """Generate SHAP explanations for a classifier or ensemble.

    Parameters
    ----------
    model
        Any object exposing ``predict_proba(list[str]) -> np.ndarray``.
    class_names
        Human-readable names indexed by label.
    max_evals
        Maximum number of model evaluations per explanation. SHAP uses this
        to bound the Shapley value approximation — higher is more accurate
        but slower. The default of 500 balances quality and speed for short
        responses; raise for final reporting.
    batch_size
        How many masked variants to score per model call.
    """

    def __init__(
        self,
        model,
        *,
        class_names: Sequence[str],
        max_evals: int = 500,
        batch_size: int = 16,
        ngram: int = 1,
    ) -> None:
        # Deferred imports so the package imports without shap installed.
        import shap

        self._shap = shap
        self.model = model
        self.class_names = list(class_names)
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.ngram = ngram

        # A Text masker segments on a regex and masks whole words/tokens.
        # In bigram mode the text is presented in ~-joined overlapping-bigram
        # form (see _to_bigram_form); splitting on whitespace then makes each
        # ~-joined bigram one maskable unit, giving phrase-level attribution.
        # _predict_proba reconstructs readable text before the model sees it.
        if ngram >= 2:
            self._masker = shap.maskers.Text(r"\s+")
        else:
            self._masker = shap.maskers.Text(r"\W+")
        self._explainer = shap.Explainer(
            self._predict_proba,
            self._masker,
            output_names=self.class_names,
        )

    @staticmethod
    def _to_bigram_form(text: str) -> str:
        """Convert 'a b c' -> 'a~b b~c' (overlapping bigrams, ~-joined)."""
        import re
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        if len(words) < 2:
            return text
        return " ".join(f"{words[i]}~{words[i+1]}" for i in range(len(words) - 1))

    @staticmethod
    def _from_bigram_form(text: str) -> str:
        """Reconstruct readable text from ~-joined overlapping bigrams."""
        toks = text.split()
        words: list[str] = []
        for t in toks:
            parts = t.split("~")
            if not words:
                words.extend(parts)
            else:
                words.append(parts[-1])
        return " ".join(words)

    # ------------------------------------------------------------------ #
    # Prediction function passed to SHAP
    # ------------------------------------------------------------------ #

    def _predict_proba(self, texts) -> np.ndarray:
        """Wrap the model's predict_proba for SHAP.

        SHAP may pass a numpy array of strings rather than a list, so coerce
        to a plain list of str before calling the model. In bigram mode the
        strings are in ~-joined form, so reconstruct readable text first.
        """
        texts = [str(t) for t in texts]
        if self.ngram >= 2:
            texts = [self._from_bigram_form(t) for t in texts]
        return self.model.predict_proba(texts)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _token_count(self, text: str) -> int:
        """Count tokens the way the active masker will segment them.

        Unigram mode splits on \\W+ (word characters); bigram mode splits on
        whitespace (the ~-joined bigrams). Used to guard against too-short
        responses that crash SHAP's clustering.
        """
        import re
        if self.ngram >= 2:
            return len([t for t in re.split(r"\s+", text.strip()) if t])
        return len([t for t in re.split(r"\W+", text.strip()) if t])

    def explain(
        self,
        text: str,
        *,
        explained_class: int | None = None,
    ) -> ShapExplanation:
        """Explain a single response.

        Parameters
        ----------
        text
            The student response to explain.
        explained_class
            Which class to explain. Defaults to the predicted class.
        """
        proba = self.model.predict_proba([text])[0]
        predicted_label = int(np.argmax(proba))
        target = predicted_label if explained_class is None else explained_class

        # SHAP expects a batch; pass a single-element list.
        # In bigram mode, present SHAP the ~-joined overlapping-bigram form.
        shap_input = self._to_bigram_form(text) if self.ngram >= 2 else text

        # SHAP's partition masker builds a token clustering that fails on
        # responses with fewer than 2 tokens (empty clustering array →
        # "zero-size array to reduction" error). Return an empty explanation
        # for these rather than crashing the batch; they are inherently
        # unexplainable and excluded by coverage anyway.
        if self._token_count(shap_input) < 2:
            return ShapExplanation(
                text=text,
                predicted_label=predicted_label,
                predicted_proba=proba,
                class_names=self.class_names,
                tokens=[text] if text else [],
                shap_values=np.zeros(1 if text else 0),
                base_value=float(proba[target]),
                explained_class=target,
            )

        shap_values = self._explainer(
            [shap_input],
            max_evals=self.max_evals,
            batch_size=self.batch_size,
            silent=True,
        )

        # shap_values has shape (n_texts, n_tokens, n_classes).
        # Extract the single text and the target class column.
        tokens = list(shap_values.data[0])
        values_for_class = np.array(shap_values.values[0][:, target])
        base = float(np.array(shap_values.base_values[0]).reshape(-1)[target])

        # In bigram mode, convert ~-joined token labels to readable phrases.
        if self.ngram >= 2:
            tokens = [str(t).replace("~", " ") for t in tokens]

        return ShapExplanation(
            text=text,
            predicted_label=predicted_label,
            predicted_proba=proba,
            class_names=self.class_names,
            tokens=tokens,
            shap_values=values_for_class,
            base_value=base,
            explained_class=target,
        )

    def explain_batch(self, texts: Sequence[str]) -> list[ShapExplanation]:
        """Explain multiple responses sequentially.

        Each explanation is independent. Progress is logged every 10 items
        since SHAP is compute-heavy.
        """
        results: list[ShapExplanation] = []
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                logger.info("SHAP progress: %d / %d", i, len(texts))
            results.append(self.explain(text))
        return results