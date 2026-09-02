"""Explainable AI layer for the EGH490 project.

Provides post-hoc explanation techniques for the transformer classifiers and
ensemble, plus quantitative metrics for evaluating explanation quality.

Techniques
----------
LimeExplainer        Local per-response explanations via input perturbation.
ShapExplainer        Shapley-value attributions with additive consistency.
AttentionExtractor   Transformer attention weights (qualitative complement).

Evaluation
----------
compute_fidelity     Comprehensiveness & sufficiency — does the explanation
                     reflect the model's actual decision?
compute_stability    Rank-correlation of explanations across repeats.
compute_coverage     Fraction of responses with a meaningful explanation.

All explainers accept any object exposing ``predict_proba(list[str])`` —
both ``TransformerClassifier`` and ``Ensemble`` qualify — except
``AttentionExtractor``, which needs a single ``TransformerClassifier``
because attention is internal to one transformer.
"""

from egh490.xai.attention import AttentionExtractor, AttentionResult
from egh490.xai.evaluation import (
    CoverageResult,
    FidelityResult,
    StabilityResult,
    compute_coverage,
    compute_fidelity,
    compute_stability,
)
from egh490.xai.lime_explainer import LimeExplainer, LimeExplanation
from egh490.xai.shap_explainer import ShapExplainer, ShapExplanation
from egh490.xai.visualise import (
    plot_feature_importance,
    plot_lime_vs_shap,
    plot_token_heatmap,
)

__all__ = [
    "LimeExplainer",
    "LimeExplanation",
    "ShapExplainer",
    "ShapExplanation",
    "AttentionExtractor",
    "AttentionResult",
    "compute_fidelity",
    "compute_stability",
    "compute_coverage",
    "FidelityResult",
    "StabilityResult",
    "CoverageResult",
    "plot_token_heatmap",
    "plot_feature_importance",
    "plot_lime_vs_shap",
]
