"""Quantitative evaluation of explanation quality.

An explanation that looks plausible is not necessarily a good one. This
module implements three metrics for assessing XAI output objectively,
following the framework in Gunasekara & Saarela (2025) and the explanation
coverage idea from Villegas-Ch et al. (2025):

1. **Fidelity** — does the explanation reflect what the model actually did?
   Measured by comprehensiveness and sufficiency (DeYoung et al., 2020):
     - *Comprehensiveness*: remove the top-k important words. If they truly
       drove the prediction, the model's confidence in its class should drop
       substantially. A large drop = high comprehensiveness.
     - *Sufficiency*: keep ONLY the top-k important words. If they are
       sufficient, the model's confidence should stay close to the original.
       A small drop = high sufficiency.

2. **Stability** — does the same input produce consistent explanations?
   LIME and SHAP both involve random sampling, so re-explaining the same
   response can give different weights. Stability measures the average rank
   correlation of feature importances across repeated explanations. High
   correlation = stable, trustworthy explanations.

3. **Coverage** — for what fraction of responses can a meaningful
   explanation be produced at all? An explanation with no non-trivial
   feature weights (e.g. every weight ~0) does not explain anything.
   Coverage is the proportion of responses that clear a minimum-signal bar.

All three return values in [0, 1] where higher is better, so they can be
reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from egh490.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Fidelity — comprehensiveness & sufficiency
# ------------------------------------------------------------------ #

@dataclass
class FidelityResult:
    """Fidelity metrics for one or more explanations.

    Attributes
    ----------
    comprehensiveness
        Mean probability drop when the top-k features are removed. Higher is
        better (the important words really mattered). Range [0, 1].
    sufficiency
        Mean probability drop when only the top-k features are kept. Lower is
        better, so we report ``1 - drop`` so that higher is better and it is
        comparable with comprehensiveness. Range [0, 1].
    n_instances
        How many explanations were evaluated.
    k
        How many top features were used.
    """

    comprehensiveness: float
    sufficiency: float
    n_instances: int
    k: int

    def as_dict(self) -> dict:
        return {
            "comprehensiveness": float(self.comprehensiveness),
            "sufficiency": float(self.sufficiency),
            "n_instances": int(self.n_instances),
            "k": int(self.k),
        }


def _tokenise(text: str) -> list[str]:
    """Split a response into whitespace-delimited words.

    Fidelity works by rebuilding the response with certain words removed or
    kept, so we need a simple, reversible tokenisation. Whitespace splitting
    matches how LIME's bag-of-words treats text, keeping the two consistent.
    """
    return text.split()


def _rebuild(words: Sequence[str]) -> str:
    return " ".join(words)


def compute_fidelity(
    model,
    explanations,
    *,
    k: int = 5,
) -> FidelityResult:
    """Compute comprehensiveness and sufficiency over a set of explanations.

    Parameters
    ----------
    model
        The classifier/ensemble that produced the predictions, exposing
        ``predict_proba``.
    explanations
        An iterable of explanation objects (LimeExplanation or
        ShapExplanation). Each must expose ``text``, ``predicted_label``,
        ``predicted_proba``, and ``top_features(n)``.
    k
        Number of top features to remove / keep.

    Returns
    -------
    FidelityResult
    """
    comp_drops: list[float] = []
    suff_scores: list[float] = []

    for exp in explanations:
        words = _tokenise(exp.text)
        if not words:
            continue

        label = exp.predicted_label
        original_conf = float(exp.predicted_proba[label])

        # Identify the top-k important words (by magnitude) as a set.
        top_words = {w for w, _ in exp.top_features(k, by_magnitude=True)}

        # Comprehensiveness — remove the important words.
        reduced = [w for w in words if w not in top_words]
        if reduced:
            reduced_conf = float(model.predict_proba([_rebuild(reduced)])[0][label])
        else:
            # Everything was important; removing all leaves nothing.
            reduced_conf = 0.0
        comp_drops.append(max(0.0, original_conf - reduced_conf))

        # Sufficiency — keep only the important words.
        kept = [w for w in words if w in top_words]
        if kept:
            kept_conf = float(model.predict_proba([_rebuild(kept)])[0][label])
        else:
            kept_conf = 0.0
        # Small drop = high sufficiency, so report 1 - drop.
        suff_scores.append(1.0 - max(0.0, original_conf - kept_conf))

    n = len(comp_drops)
    if n == 0:
        return FidelityResult(0.0, 0.0, 0, k)

    return FidelityResult(
        comprehensiveness=float(np.mean(comp_drops)),
        sufficiency=float(np.mean(suff_scores)),
        n_instances=n,
        k=k,
    )


# ------------------------------------------------------------------ #
# Stability — rank correlation across repeated explanations
# ------------------------------------------------------------------ #

@dataclass
class StabilityResult:
    """Stability metric across repeated explanations of the same inputs.

    Attributes
    ----------
    mean_rank_correlation
        Mean Spearman rank correlation of feature-importance orderings across
        repeated explanations of the same response, averaged over all
        responses. Range roughly [-1, 1]; higher (closer to 1) is better.
    n_instances
        Number of responses evaluated.
    n_repeats
        How many times each response was re-explained.
    """

    mean_rank_correlation: float
    n_instances: int
    n_repeats: int

    def as_dict(self) -> dict:
        return {
            "mean_rank_correlation": float(self.mean_rank_correlation),
            "n_instances": int(self.n_instances),
            "n_repeats": int(self.n_repeats),
        }


def compute_stability(
    explain_fn: Callable[[str], object],
    texts: Sequence[str],
    *,
    n_repeats: int = 5,
    top_k: int = 10,
) -> StabilityResult:
    """Measure explanation stability by re-explaining each response.

    Parameters
    ----------
    explain_fn
        A callable that takes a response string and returns an explanation
        object exposing ``top_features(n)``. Typically
        ``lambda t: lime_explainer.explain(t)``.
    texts
        Responses to evaluate.
    n_repeats
        How many times to re-explain each response.
    top_k
        How many top features to compare across repeats.

    Returns
    -------
    StabilityResult

    Notes
    -----
    Uses Spearman rank correlation on the union of top-k features across
    each pair of repeats. Features absent from one repeat are assigned a rank
    below all present features, so a feature that appears in one explanation
    but vanishes in another is penalised.
    """
    from itertools import combinations

    from scipy.stats import spearmanr

    per_text_corrs: list[float] = []

    for text in texts:
        # Collect feature -> weight dicts across repeats.
        repeats: list[dict[str, float]] = []
        for _ in range(n_repeats):
            exp = explain_fn(text)
            weights = dict(exp.top_features(top_k, by_magnitude=True))
            repeats.append(weights)

        # For each pair of repeats, build aligned weight vectors over the
        # union of their features and compute Spearman correlation.
        pair_corrs: list[float] = []
        for a, b in combinations(range(n_repeats), 2):
            feats = set(repeats[a]) | set(repeats[b])
            if len(feats) < 2:
                continue
            vec_a = [repeats[a].get(f, 0.0) for f in feats]
            vec_b = [repeats[b].get(f, 0.0) for f in feats]
            corr, _ = spearmanr(vec_a, vec_b)
            if not np.isnan(corr):
                pair_corrs.append(corr)

        if pair_corrs:
            per_text_corrs.append(float(np.mean(pair_corrs)))

    n = len(per_text_corrs)
    if n == 0:
        return StabilityResult(0.0, 0, n_repeats)

    return StabilityResult(
        mean_rank_correlation=float(np.mean(per_text_corrs)),
        n_instances=n,
        n_repeats=n_repeats,
    )


# ------------------------------------------------------------------ #
# Coverage — fraction of responses with a meaningful explanation
# ------------------------------------------------------------------ #

@dataclass
class CoverageResult:
    """Explanation coverage across a set of responses.

    Attributes
    ----------
    coverage
        Fraction of responses for which the explanation contains at least one
        feature whose absolute weight exceeds ``min_weight``. Range [0, 1].
    n_covered
        Number of responses with a meaningful explanation.
    n_total
        Total responses evaluated.
    min_weight
        The threshold a feature weight had to exceed to count as meaningful.
    """

    coverage: float
    n_covered: int
    n_total: int
    min_weight: float

    def as_dict(self) -> dict:
        return {
            "coverage": float(self.coverage),
            "n_covered": int(self.n_covered),
            "n_total": int(self.n_total),
            "min_weight": float(self.min_weight),
        }


def compute_coverage(
    explanations,
    *,
    min_weight: float = 0.01,
) -> CoverageResult:
    """Compute the fraction of explanations that carry meaningful signal.

    Parameters
    ----------
    explanations
        Iterable of explanation objects exposing ``top_features(n)``.
    min_weight
        Minimum absolute feature weight for an explanation to count as
        meaningful. An explanation whose strongest feature is below this bar
        is treated as uninformative.
    """
    n_total = 0
    n_covered = 0

    for exp in explanations:
        n_total += 1
        top = exp.top_features(1, by_magnitude=True)
        if top and abs(top[0][1]) >= min_weight:
            n_covered += 1

    coverage = (n_covered / n_total) if n_total else 0.0
    return CoverageResult(
        coverage=coverage,
        n_covered=n_covered,
        n_total=n_total,
        min_weight=min_weight,
    )
