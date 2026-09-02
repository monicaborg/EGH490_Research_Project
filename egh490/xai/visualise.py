"""Visualisation helpers for XAI explanations.

Renders explanation results as figures suitable for the report and educator
dashboard:

- ``plot_token_heatmap`` — colours each word of a response by its importance
  weight, producing the "highlighted text" view that makes an explanation
  legible to a non-technical educator.
- ``plot_feature_importance`` — a horizontal bar chart of the top features
  for a single explanation.
- ``plot_lime_vs_shap`` — compares LIME and SHAP attributions for the same
  response side by side, useful for the methods discussion.

All figures use the same transparent-background, slide-safe styling as
plot_results.py so they drop cleanly into the report and presentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Slide-safe palette consistent with plot_results.py
POSITIVE_COLOR = "#0f7558"   # green — evidence toward the explained class
NEGATIVE_COLOR = "#c04f20"   # orange — evidence against
TEXT_COLOR     = "#1a1a2e"
MUTED_COLOR    = "#555566"

SAVEFIG_KW = dict(dpi=150, bbox_inches="tight", transparent=True)


def _normalise_weights(weights: Sequence[float]) -> np.ndarray:
    """Scale weights to [-1, 1] by the largest magnitude for colour mapping."""
    w = np.array(weights, dtype=float)
    peak = np.abs(w).max()
    if peak > 0:
        w = w / peak
    return w


def plot_token_heatmap(
    tokens: Sequence[str],
    weights: Sequence[float],
    *,
    title: str = "",
    out_path: str | Path | None = None,
):
    """Render a response as coloured tokens weighted by importance.

    Green tokens push toward the explained class, orange tokens push against,
    and colour intensity reflects magnitude. This is the primary
    educator-facing explanation view.

    Parameters
    ----------
    tokens
        The words/tokens of the response, in order.
    weights
        Importance weight per token, aligned with ``tokens``.
    title
        Optional title (e.g. the predicted class and confidence).
    out_path
        If given, save the figure there; otherwise return the figure.
    """
    norm = _normalise_weights(weights)

    fig, ax = plt.subplots(figsize=(min(12, 0.28 * len(tokens) + 2), 1.6))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=10, loc="left")

    # Lay tokens left to right, wrapping to new lines as needed.
    x, y = 0.0, 0.9
    line_height = 0.28
    max_width = 1.0

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for token, w in zip(tokens, norm):
        colour = POSITIVE_COLOR if w >= 0 else NEGATIVE_COLOR
        alpha = float(min(0.85, 0.15 + abs(w) * 0.7))
        # White text on strong fills, dark text on faint ones, for contrast.
        txt_colour = "white" if abs(w) > 0.55 else TEXT_COLOR

        txt = ax.text(
            x, y, token + " ",
            fontsize=13, color=txt_colour,
            bbox=dict(boxstyle="round,pad=0.25", fc=colour, ec="none", alpha=alpha),
            transform=ax.transAxes,
        )
        bbox = txt.get_window_extent(renderer=renderer)
        width = bbox.width / fig.bbox.width
        x += width + 0.008
        if x > max_width:
            x = 0.0
            y -= line_height

    # Legend
    ax.text(0.0, y - line_height, "■", color=POSITIVE_COLOR, fontsize=12,
            transform=ax.transAxes)
    ax.text(0.03, y - line_height, "supports prediction", color=MUTED_COLOR,
            fontsize=9, transform=ax.transAxes)
    ax.text(0.45, y - line_height, "■", color=NEGATIVE_COLOR, fontsize=12,
            transform=ax.transAxes)
    ax.text(0.48, y - line_height, "counts against", color=MUTED_COLOR,
            fontsize=9, transform=ax.transAxes)

    if out_path:
        fig.savefig(out_path, **SAVEFIG_KW)
        plt.close(fig)
        return None
    return fig


def plot_feature_importance(
    feature_weights: Sequence[tuple[str, float]],
    *,
    title: str = "Feature importance",
    out_path: str | Path | None = None,
):
    """Horizontal bar chart of the top features for one explanation.

    Positive weights extend right in green, negative extend left in orange,
    sorted by magnitude so the strongest evidence sits at the top.
    """
    pairs = sorted(feature_weights, key=lambda kv: abs(kv[1]), reverse=True)
    words = [w for w, _ in pairs][::-1]
    values = [v for _, v in pairs][::-1]
    colours = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(words))))
    ax.barh(words, values, color=colours, alpha=0.88, edgecolor="none")
    ax.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=10)
    ax.set_xlabel("Importance weight", color=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(MUTED_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, **SAVEFIG_KW)
        plt.close(fig)
        return None
    return fig


def plot_lime_vs_shap(
    lime_explanation,
    shap_explanation,
    *,
    top_k: int = 10,
    out_path: str | Path | None = None,
):
    """Compare LIME and SHAP attributions for the same response.

    Aligns the two methods on their combined top-k features and plots them as
    grouped horizontal bars, so agreement and disagreement between the two
    explanation techniques is visible at a glance.
    """
    lime_weights = dict(lime_explanation.top_features(top_k, by_magnitude=True))
    shap_weights = dict(
        (w.strip(), s) for w, s in shap_explanation.top_features(top_k, by_magnitude=True)
    )

    feats = list(dict.fromkeys(list(lime_weights) + list(shap_weights)))
    feats = sorted(
        feats,
        key=lambda f: max(abs(lime_weights.get(f, 0)), abs(shap_weights.get(f, 0))),
    )

    lime_vals = [lime_weights.get(f, 0.0) for f in feats]
    shap_vals = [shap_weights.get(f, 0.0) for f in feats]

    y = np.arange(len(feats))
    h = 0.38

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(feats))))
    ax.barh(y + h / 2, lime_vals, h, label="LIME", color="#6a44c0", alpha=0.88, edgecolor="none")
    ax.barh(y - h / 2, shap_vals, h, label="SHAP", color="#2a5fb8", alpha=0.88, edgecolor="none")
    ax.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Importance weight", color=TEXT_COLOR)
    ax.set_title("LIME vs SHAP attributions", fontsize=12, color=TEXT_COLOR, pad=10)
    ax.legend(fontsize=9, frameon=False)
    for spine in ax.spines.values():
        spine.set_color(MUTED_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, **SAVEFIG_KW)
        plt.close(fig)
        return None
    return fig
