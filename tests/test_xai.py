"""Tests for the XAI layer.

These use a deterministic mock model that mimics the ``predict_proba``
interface of ``TransformerClassifier`` / ``Ensemble`` without loading any
transformer weights. This keeps the tests fast and torch-independent while
still exercising the real LIME, SHAP, and evaluation code paths.

The mock keys its output on a few Signals & Systems marker words so the
explanations have a known "correct answer" we can assert against.
"""

from __future__ import annotations

import numpy as np
import pytest


class MockModel:
    """Deterministic classifier for XAI testing.

    Pushes probability toward class 1 ("correct") when conceptual marker
    words are present, and toward class 0 when misconception words appear.
    Exposes the same predict_proba / predict interface the explainers rely on.
    """

    POSITIVE = {"aliases", "nyquist", "convolution", "sinc"}
    NEGATIVE = {"amplitude", "multiply", "louder"}

    def predict_proba(self, texts):
        out = []
        for t in texts:
            tokens = str(t).lower().split()
            score = 0.5
            score += 0.15 * sum(w in self.POSITIVE for w in tokens)
            score -= 0.15 * sum(w in self.NEGATIVE for w in tokens)
            score = min(max(score, 0.01), 0.99)
            out.append([1 - score, score])
        return np.array(out)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)


CLASS_NAMES = ["incorrect", "correct"]

SAMPLE_TEXTS = [
    "the signal aliases because fs is below the nyquist rate",
    "you just multiply the two signals to get louder output",
    "convolution of the input with the impulse response gives the output",
    "the amplitude is higher so the frequency must be higher too",
]


@pytest.fixture
def model():
    return MockModel()


# ------------------------------------------------------------------ #
# LIME
# ------------------------------------------------------------------ #

def test_lime_identifies_marker_word(model):
    from egh490.xai import LimeExplainer

    lime = LimeExplainer(model, class_names=CLASS_NAMES, num_samples=300)
    exp = lime.explain("the signal aliases because fs is below the nyquist rate")

    # The model predicts "correct" for this response.
    assert exp.class_names[exp.predicted_label] == "correct"

    # "aliases" or "nyquist" should be among the strongest positive features.
    top_words = {w for w, _ in exp.top_features(5)}
    assert "aliases" in top_words or "nyquist" in top_words


def test_lime_explanation_serialises(model):
    from egh490.xai import LimeExplainer

    lime = LimeExplainer(model, class_names=CLASS_NAMES, num_samples=200)
    exp = lime.explain(SAMPLE_TEXTS[0])
    d = exp.as_dict()

    assert set(d) >= {
        "text", "predicted_label", "predicted_proba",
        "class_names", "feature_weights", "explained_class",
    }
    assert d["class_names"] == CLASS_NAMES


def test_lime_batch(model):
    from egh490.xai import LimeExplainer

    lime = LimeExplainer(model, class_names=CLASS_NAMES, num_samples=150)
    results = lime.explain_batch(SAMPLE_TEXTS[:2])
    assert len(results) == 2


# ------------------------------------------------------------------ #
# SHAP
# ------------------------------------------------------------------ #

def test_shap_identifies_marker_word(model):
    from egh490.xai import ShapExplainer

    shap_exp = ShapExplainer(model, class_names=CLASS_NAMES, max_evals=100)
    exp = shap_exp.explain("the signal aliases because fs is below the nyquist rate")

    assert exp.class_names[exp.predicted_label] == "correct"
    top_words = {w.strip() for w, _ in exp.top_features(5)}
    assert "aliases" in top_words or "nyquist" in top_words


def test_shap_values_have_base(model):
    from egh490.xai import ShapExplainer

    shap_exp = ShapExplainer(model, class_names=CLASS_NAMES, max_evals=100)
    exp = shap_exp.explain(SAMPLE_TEXTS[0])
    # Base value should be a finite float around the model's average output.
    assert np.isfinite(exp.base_value)
    assert len(exp.tokens) == len(exp.shap_values)


# ------------------------------------------------------------------ #
# Evaluation — fidelity, stability, coverage
# ------------------------------------------------------------------ #

def test_fidelity_positive_for_good_explanations(model):
    from egh490.xai import LimeExplainer, compute_fidelity

    lime = LimeExplainer(model, class_names=CLASS_NAMES, num_samples=300)
    exps = lime.explain_batch(SAMPLE_TEXTS)
    fid = compute_fidelity(model, exps, k=3)

    # Removing the marker words should reduce confidence, so
    # comprehensiveness should be positive.
    assert fid.comprehensiveness > 0
    assert 0.0 <= fid.sufficiency <= 1.0
    assert fid.n_instances == len(SAMPLE_TEXTS)


def test_coverage_range(model):
    from egh490.xai import LimeExplainer, compute_coverage

    lime = LimeExplainer(model, class_names=CLASS_NAMES, num_samples=200)
    exps = lime.explain_batch(SAMPLE_TEXTS)
    cov = compute_coverage(exps)

    assert 0.0 <= cov.coverage <= 1.0
    assert cov.n_total == len(SAMPLE_TEXTS)


def test_stability_range(model):
    from egh490.xai import LimeExplainer, compute_stability

    lime = LimeExplainer(model, class_names=CLASS_NAMES, num_samples=200)
    stab = compute_stability(
        lambda t: lime.explain(t),
        SAMPLE_TEXTS[:2],
        n_repeats=3,
    )
    # Rank correlation is in [-1, 1]; for a deterministic model it should be
    # reasonably high, but we only assert the valid range to avoid flakiness.
    assert -1.0 <= stab.mean_rank_correlation <= 1.0
    assert stab.n_repeats == 3


# ------------------------------------------------------------------ #
# Explainers accept an ensemble-like interface
# ------------------------------------------------------------------ #

def test_explainer_works_on_any_predict_proba():
    """Confirm the explainer only depends on predict_proba, not model type."""
    from egh490.xai import LimeExplainer

    class MinimalModel:
        def predict_proba(self, texts):
            return np.array([[0.3, 0.7] for _ in texts])

    lime = LimeExplainer(MinimalModel(), class_names=CLASS_NAMES, num_samples=100)
    exp = lime.explain("any text at all")
    assert exp.predicted_label == 1
