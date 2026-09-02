"""Build report-ready figures from the JSON saved by scripts/explain.py.

Reads lime_explanations.json, shap_explanations.json, and attention.json from
an explanations directory and produces, for hand-picked or auto-selected
responses:

  1. token_heatmap_<i>.png     — LIME / SHAP / attention side by side over the
                                 same response text (the "money figure": shows
                                 LIME+SHAP agreeing on concept words while
                                 attention bleeds onto punctuation).
  2. lime_vs_shap_<i>.png      — grouped bars comparing the two perturbation
                                 methods' top features for one response.
  3. lime_shap_agreement.png   — corpus-level scatter of LIME vs SHAP weights
                                 for shared tokens across all explained
                                 responses (mutual-validation evidence).

Everything is rebuilt from JSON — no live explainer objects needed — so it
runs after training/explanation without re-computing anything.

Usage
-----
    python scripts/plot_xai.py \\
        --explanations-dir outputs/explanations/validity_roberta_fold1_Monica_Data_CCUS_converted_ccu2 \\
        --out-dir outputs/figures/xai_ccu2

    # pick specific responses by index (0-based, order as in the JSON):
    python scripts/plot_xai.py --explanations-dir <dir> --out-dir <dir> --indices 3 20 33

    # otherwise it auto-selects the most strongly-attributed substantive responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Slide/report-safe palette, consistent with the rest of the project's figures.
POS_COLOR = "#0f7558"   # green — pushes toward the explained class
NEG_COLOR = "#c04f20"   # orange — pushes away
ATTN_COLOR = "#6a44c0"  # purple — attention magnitude (unsigned)
TEXT_COLOR = "#1a1a2e"
MUTED = "#666677"
SAVEKW = dict(dpi=150, bbox_inches="tight", facecolor="white")


# ------------------------------------------------------------------ #
# Loading & normalising
# ------------------------------------------------------------------ #

def _load(dir_: Path):
    def rd(name):
        p = dir_ / name
        return json.loads(p.read_text()) if p.exists() else []
    return rd("lime_explanations.json"), rd("shap_explanations.json"), rd("attention.json")


def _lime_pairs(entry):
    """[(word, signed_weight), ...] from a LIME entry."""
    out = []
    for item in entry.get("feature_weights", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]).strip(), float(item[1])))
    return out


def _shap_pairs(entry):
    """[(token, signed_value), ...] from a SHAP entry."""
    toks = entry.get("tokens", [])
    vals = entry.get("shap_values", [])
    return [(str(t).strip(), float(v)) for t, v in zip(toks, vals)]


def _attn_pairs(entry):
    """[(token, unsigned_weight), ...] from an attention entry."""
    toks = entry.get("tokens", [])
    ws = entry.get("weights", [])
    return [(str(t).strip(), float(w)) for t, w in zip(toks, ws)]


def _index_by_text(entries):
    return {e.get("text", ""): e for e in entries}


# ------------------------------------------------------------------ #
# Figure 1 — token heatmap (LIME / SHAP / attention stacked)
# ------------------------------------------------------------------ #

def _norm(vals):
    v = np.array(vals, dtype=float)
    peak = np.abs(v).max()
    return v / peak if peak > 0 else v


def _draw_token_row(ax, tokens, weights, signed, title):
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=TEXT_COLOR, loc="left", pad=6)
    norm = _norm(weights)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x, y = 0.0, 0.55
    line_h = 0.42
    for tok, w in zip(tokens, norm):
        if not tok:
            continue
        if signed:
            color = POS_COLOR if w >= 0 else NEG_COLOR
            alpha = float(min(0.9, 0.12 + abs(w) * 0.8))
        else:
            color = ATTN_COLOR
            alpha = float(min(0.9, 0.12 + abs(w) * 0.8))
        txtcol = "white" if abs(w) > 0.55 else TEXT_COLOR
        t = ax.text(x, y, tok + " ", fontsize=12.5, color=txtcol,
                    bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=alpha),
                    transform=ax.transAxes)
        bb = t.get_window_extent(renderer=renderer)
        x += bb.width / fig.bbox.width + 0.006
        if x > 0.98:
            x = 0.0
            y -= line_h


def plot_token_heatmap(text, lime_e, shap_e, attn_e, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.2))
    if lime_e:
        pairs = _lime_pairs(lime_e)
        _draw_token_row(axes[0], [w for w, _ in pairs], [s for _, s in pairs],
                        signed=True, title="LIME  (green = supports prediction, orange = against)")
    if shap_e:
        pairs = _shap_pairs(shap_e)
        _draw_token_row(axes[1], [w for w, _ in pairs], [s for _, s in pairs],
                        signed=True, title="SHAP  (green = supports prediction, orange = against)")
    if attn_e:
        pairs = _attn_pairs(attn_e)
        _draw_token_row(axes[2], [w for w, _ in pairs], [s for _, s in pairs],
                        signed=False, title="Attention  (purple = attention magnitude, unsigned)")
    pred = (lime_e or shap_e or {}).get("predicted_class_name", "?")
    proba = (lime_e or shap_e or {}).get("predicted_proba", [])
    conf = f"{max(proba):.2f}" if proba else "?"
    fig.suptitle(f'"{text[:80]}"    →  predicted: {pred} ({conf})',
                 fontsize=12, color=TEXT_COLOR, y=1.02, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, **SAVEKW)
    plt.close(fig)


# ------------------------------------------------------------------ #
# Figure 2 — LIME vs SHAP grouped bars for one response
# ------------------------------------------------------------------ #

def plot_lime_vs_shap(lime_e, shap_e, out_path, top_k=8):
    lime_w = {w.lower(): s for w, s in _lime_pairs(lime_e)}
    shap_w = {}
    for t, v in _shap_pairs(shap_e):
        tl = t.lower()
        # keep the largest-magnitude value if a token repeats
        if tl and (tl not in shap_w or abs(v) > abs(shap_w[tl])):
            shap_w[tl] = v
    feats = list(dict.fromkeys(list(lime_w) + list(shap_w)))
    feats = [f for f in feats if f]
    feats.sort(key=lambda f: max(abs(lime_w.get(f, 0)), abs(shap_w.get(f, 0))), reverse=True)
    feats = feats[:top_k][::-1]
    if not feats:
        return False
    lvals = [lime_w.get(f, 0.0) for f in feats]
    svals = [shap_w.get(f, 0.0) for f in feats]
    y = np.arange(len(feats))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(feats))))
    ax.barh(y + h/2, lvals, h, label="LIME", color="#2a5fb8", alpha=0.88)
    ax.barh(y - h/2, svals, h, label="SHAP", color="#0f7558", alpha=0.88)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(feats)
    ax.set_xlabel("Attribution weight", color=TEXT_COLOR)
    ax.set_title("LIME vs SHAP — same response", color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color(MUTED)
    ax.tick_params(colors=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, **SAVEKW)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ #
# Figure 3 — corpus-level LIME vs SHAP agreement scatter
# ------------------------------------------------------------------ #

def plot_agreement(lime, shap, out_path):
    shap_by_text = _index_by_text(shap)
    xs, ys = [], []
    for le in lime:
        se = shap_by_text.get(le.get("text", ""))
        if not se:
            continue
        lw = {w.lower(): s for w, s in _lime_pairs(le)}
        sw = {}
        for t, v in _shap_pairs(se):
            tl = t.lower()
            if tl and (tl not in sw or abs(v) > abs(sw[tl])):
                sw[tl] = v
        for tok in set(lw) & set(sw):
            xs.append(lw[tok]); ys.append(sw[tok])
    if not xs:
        return False
    xs, ys = np.array(xs), np.array(ys)
    # correlation
    if xs.std() > 0 and ys.std() > 0:
        r = float(np.corrcoef(xs, ys)[0, 1])
    else:
        r = float("nan")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=18, alpha=0.45, color="#2a5fb8", edgecolors="none")
    lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--", label="perfect agreement")
    ax.axhline(0, color=MUTED, lw=0.5); ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("LIME weight", color=TEXT_COLOR)
    ax.set_ylabel("SHAP value", color=TEXT_COLOR)
    title = "LIME vs SHAP agreement across shared tokens"
    if r == r:
        title += f"\nPearson r = {r:.3f}  (n = {len(xs)} tokens)"
    ax.set_title(title, color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color(MUTED)
    ax.tick_params(colors=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, **SAVEKW)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ #
# Auto-selection of interesting responses
# ------------------------------------------------------------------ #

def _auto_indices(lime, n=3):
    """Pick substantive responses with the strongest single attribution."""
    scored = []
    for i, e in enumerate(lime):
        text = e.get("text", "")
        if len(text.split()) < 6:
            continue
        pairs = _lime_pairs(e)
        peak = max((abs(s) for _, s in pairs), default=0.0)
        scored.append((peak, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:n]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--explanations-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="0-based response indices to plot (else auto-selects strongest)")
    p.add_argument("--n-auto", type=int, default=3)
    args = p.parse_args(argv)

    exp_dir = Path(args.explanations_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lime, shap, attn = _load(exp_dir)
    if not lime:
        raise SystemExit(f"No lime_explanations.json in {exp_dir}")

    shap_by_text = _index_by_text(shap)
    attn_by_text = _index_by_text(attn)

    indices = args.indices if args.indices else _auto_indices(lime, args.n_auto)
    print(f"Plotting responses at indices: {indices}")

    for i in indices:
        le = lime[i]
        text = le.get("text", "")
        se = shap_by_text.get(text)
        ae = attn_by_text.get(text)
        plot_token_heatmap(text, le, se, ae, out_dir / f"token_heatmap_{i}.png")
        if se:
            plot_lime_vs_shap(le, se, out_dir / f"lime_vs_shap_{i}.png")
        print(f"  response {i}: '{text[:50]}' -> heatmap + bars")

    if plot_agreement(lime, shap, out_dir / "lime_shap_agreement.png"):
        print("  corpus LIME-vs-SHAP agreement scatter saved")

    print(f"\nAll figures in {out_dir}/")


if __name__ == "__main__":
    main()
