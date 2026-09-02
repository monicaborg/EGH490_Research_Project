"""Transformer attention weight extraction for qualitative XAI.

Attention weights show which tokens each transformer layer attended to when
forming its representation of a response. Unlike LIME and SHAP — which
perturb the input and are model-agnostic — attention is read directly from
inside a single transformer, so this module works on a
``TransformerClassifier`` rather than the ensemble.

An important caveat, and one worth stating explicitly in the report:
attention weights are **not** a faithful explanation of a model's decision.
High attention on a token does not guarantee that token drove the prediction
(Jain & Wallace, 2019, "Attention is not Explanation"). Attention is included
here as a *qualitative complement* to LIME and SHAP — useful for visualising
what the model looked at, and for comparing against the perturbation-based
attributions, but not as a standalone faithfulness claim.

This module extracts the attention from the final layer, averaged across
heads, for the [CLS]-equivalent pooling position, and maps it back onto the
input tokens for visualisation as a heatmap over the response text.

Example
-------
>>> from egh490.models import TransformerClassifier
>>> from egh490.xai import AttentionExtractor
>>> clf = TransformerClassifier.load("outputs/checkpoints/roberta/final")
>>> extractor = AttentionExtractor(clf)
>>> result = extractor.extract("the signal aliases because fs is below 2 fmax")
>>> result.token_weights()[:5]
[('the', 0.02), ('signal', 0.09), ('aliases', 0.21), ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from egh490.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionResult:
    """Attention weights mapped onto the input tokens for one response.

    Attributes
    ----------
    text
        The response that was analysed.
    tokens
        The tokens as segmented by the model's tokeniser, with special
        tokens ([CLS], [SEP], <s>, </s>, etc.) already removed.
    weights
        Attention weight per token, aligned with ``tokens``, normalised to
        sum to 1 across the response.
    layer
        Which transformer layer the attention was taken from (-1 = last).
    """

    text: str
    tokens: list[str]
    weights: np.ndarray
    layer: int

    def token_weights(self) -> list[tuple[str, float]]:
        """Return ``(token, weight)`` pairs aligned by position."""
        return list(zip(self.tokens, [float(w) for w in self.weights]))

    def top_tokens(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the ``n`` most attended tokens."""
        pairs = self.token_weights()
        return sorted(pairs, key=lambda kv: kv[1], reverse=True)[:n]

    def as_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "text": self.text,
            "tokens": self.tokens,
            "weights": [float(w) for w in self.weights],
            "layer": self.layer,
        }


class AttentionExtractor:
    """Extract attention weights from a single transformer classifier.

    Parameters
    ----------
    classifier
        A ``TransformerClassifier``. The ensemble is not supported because
        attention is an internal property of one transformer, not of a vote.
    layer
        Which layer's attention to extract. ``-1`` (default) takes the final
        layer, which is closest to the classification decision.
    """

    def __init__(self, classifier, *, layer: int = -1) -> None:
        self.clf = classifier
        self.layer = layer

    def extract(self, text: str) -> AttentionResult:
        """Extract token-level attention for one response.

        Runs a forward pass with ``output_attentions=True`` and averages the
        attention matrix across heads, then takes the row corresponding to
        the pooling token (position 0 for BERT-style models) to get a single
        weight per input token.
        """
        torch = self.clf._torch
        tokenizer = self.clf.tokenizer
        model = self.clf.model
        device = self.clf.device

        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.clf.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)

        # attentions: tuple of (n_layers) tensors, each
        # (batch, n_heads, seq_len, seq_len). Take requested layer, drop batch.
        attn = outputs.attentions[self.layer][0]        # (n_heads, seq, seq)
        attn = attn.mean(dim=0)                          # average heads -> (seq, seq)

        # Row 0 is how the pooling/[CLS] position attends to every token.
        cls_attention = attn[0].cpu().numpy()            # (seq,)

        input_ids = encoded["input_ids"][0].cpu().numpy()
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Drop special tokens and their attention weights.
        special_ids = set(tokenizer.all_special_ids)
        keep = [i for i, tid in enumerate(input_ids) if tid not in special_ids]

        tokens = [self._clean_token(raw_tokens[i]) for i in keep]
        weights = cls_attention[keep]

        # Renormalise across the retained tokens so weights sum to 1.
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return AttentionResult(
            text=text,
            tokens=tokens,
            weights=weights,
            layer=self.layer,
        )

    def extract_batch(self, texts: Sequence[str]) -> list[AttentionResult]:
        """Extract attention for multiple responses sequentially."""
        return [self.extract(t) for t in texts]

    @staticmethod
    def _clean_token(token: str) -> str:
        """Strip tokeniser artefacts for readable display.

        Different tokenisers mark word continuations differently:
          - RoBERTa/GPT-2 BPE uses a leading 'Ġ' for word-initial tokens
          - ALBERT/XLNet SentencePiece uses a leading '▁'
          - BERT WordPiece uses '##' for continuations
        Normalising these makes the heatmap readable.
        """
        return (
            token.replace("Ġ", "")
            .replace("▁", "")
            .replace("##", "")
        )
