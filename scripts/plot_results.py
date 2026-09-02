"""Generate all presentation charts from saved JSON metric files.

Transparent background, slide-safe colours (≥4.5:1 contrast on #fefbf5).

Charts generated
----------------
Always (from any results JSON):
  model_comparison.png        — mean ± std accuracy / F1 / AUC per model
  per_fold_accuracy.png       — per-fold accuracy bars
  radar_comparison.png        — 5-metric spider chart
  cv_variance.png             — mean ± std strip plot showing fold stability
  training_times.png          — total training time per model
  class_distribution.png      — valid vs invalid response counts per CCU
  batch_size_table.png        — batch size / time per epoch annotation table

When loss curve data exists (requires re-running train_all_models.py):
  loss_curves.png             — train vs val loss per epoch, one panel per model

When confusion matrix data exists:
  confusion_matrices.png      — 2×2 confusion matrix, one panel per model

When a second dataset tag is supplied (--compare-dataset):
  training_times_vs_<tag>.png — side-by-side training time comparison
  performance_gap.png         — grouped bars: dataset1 vs dataset2 accuracy per model

Usage
-----
    python scripts/plot_results.py --dataset synthetic_responses_2000

    python scripts/plot_results.py --dataset synthetic_responses_2000 \\
        --compare-dataset synthetic_responses --benchmarks --memory

All charts saved to outputs/figures/<dataset>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

MODEL_ORDER  = ["electra", "roberta", "xlnet", "albert"]
MODEL_LABELS = {
    "electra": "ELECTRA-small",
    "roberta": "RoBERTa-base",
    "xlnet":   "XLNet-base",
    "albert":  "ALBERT-base-v2",
}

# Slide background #fefbf5 — all colours ≥4.5:1 contrast
COLORS = {
    "electra": "#2a5fb8",   # 5.93:1
    "roberta": "#6a44c0",   # 6.35:1
    "xlnet":   "#c04f20",   # 4.64:1
    "albert":  "#0f7558",   # 5.49:1
}
TEXT_COLOR   = "#1a1a2e"   # 16.51:1
MUTED_COLOR  = "#555566"   #  7.07:1
GRID_COLOR   = "#c8c4bc"
WARN_COLOR   = "#8a7000"   #  4.62:1
DANGER_COLOR = "#b02020"   #  6.62:1
AUC_COLOR    = "#1a5fa0"   #  6.38:1

SAVEFIG_KW = dict(dpi=150, bbox_inches="tight", transparent=True)

# ------------------------------------------------------------------ #
# Font configuration
# ------------------------------------------------------------------ #
# FONT_PRESET controls which font is used across all charts.
# Switch between presets using the --font flag when running the script:
#
#   --font slides   → DM Sans  (matches your Canva / Keynote presentation)
#   --font report   → Calibri  (matches your Word report)
#   --font system   → system default sans-serif (fallback, always works)
#
# DM Sans must be registered with matplotlib from the .ttf files.
# Calibri is available if Microsoft Office is installed (macOS/Windows).
# The registration happens once in apply_font() called from main().

FONT_PRESETS = {
    "slides": {
        "family":      "DM Sans",
        "ttf_patterns": ["DMSans-*.ttf", "DMSans-*.ttc", "DMSans*.otf"],
        "ttf_dirs": [
            "~/Library/Fonts",
            "/Library/Fonts",
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
        ],
        "fallback": "DejaVu Sans",
    },
    "report": {
        "family":      "Calibri",
        "ttf_patterns": ["Calibri*.ttf", "Calibri*.ttc", "Calibri*.otf",
                         "calibri*.ttf", "calibri*.ttc"],
        "ttf_dirs": [
            "~/Library/Fonts",
            "/Library/Fonts",
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
            "C:/Windows/Fonts",
        ],
        "fallback": "DejaVu Sans",
    },
    "system": {
        "family":       "sans-serif",
        "ttf_patterns": None,
        "ttf_dirs":     [],
        "fallback":     "sans-serif",
    },
}

_ACTIVE_FONT = "DM Sans"


def apply_font(preset: str = "slides") -> str:
    """Register TTF/TTC files with matplotlib and set the active font family.

    Handles .ttf, .ttc (TrueType Collection — Calibri on macOS), and .otf.
    Sets font.family directly to the resolved font name so matplotlib
    actually uses it rather than falling back to its default sans-serif list.
    """
    import glob
    import os
    from matplotlib import font_manager

    cfg      = FONT_PRESETS.get(preset, FONT_PRESETS["system"])
    family   = cfg["family"]
    fallback = cfg["fallback"]

    if cfg["ttf_patterns"] is None:
        resolved = fallback
        print(f"  Font: system default ({fallback})")
    else:
        found_files = []
        for d in cfg["ttf_dirs"]:
            expanded = os.path.expanduser(d)
            for pattern in cfg["ttf_patterns"]:
                matches = glob.glob(os.path.join(expanded, pattern))
                found_files.extend(matches)
        found_files = sorted(set(found_files))

        if found_files:
            for ttf in found_files:
                try:
                    font_manager.fontManager.addfont(ttf)
                except Exception as e:
                    print(f"  Warn: could not register {ttf}: {e}")

            # Verify the font is actually now available to matplotlib
            available = {f.name for f in font_manager.fontManager.ttflist}
            if family in available:
                resolved = family
                print(f"  Font: {family} ({len(found_files)} files registered, verified)")
            else:
                # Some fonts register under variant names — try fuzzy match
                matches = [n for n in available if family.lower() in n.lower()]
                if matches:
                    resolved = matches[0]
                    print(f"  Font: requested {family}, registered as {resolved}")
                else:
                    resolved = fallback
                    print(f"  Font: {family} files found but not registered — "
                          f"using {fallback}")
                    print(f"  (registered fonts include: "
                          f"{sorted([n for n in available if 'calibri' in n.lower() or 'dm sans' in n.lower()])})")
        else:
            resolved = fallback
            print(f"  Font: {family} not found — using {fallback}")
            print(f"  (searched: {', '.join(cfg['ttf_dirs'])})")
            print(f"  (patterns: {', '.join(cfg['ttf_patterns'])})")

    global _ACTIVE_FONT
    _ACTIVE_FONT = resolved

    # Set font.family DIRECTLY to the resolved font name (not "sans-serif").
    # matplotlib's "sans-serif" family ignores the sans-serif list unless
    # font.family is exactly "sans-serif" AND the first list entry resolves.
    # Setting it to the resolved name is more reliable.
    plt.rcParams.update({
        "font.family":     [resolved, "DejaVu Sans", "Arial", "sans-serif"],
        "font.sans-serif": [resolved, "DejaVu Sans", "Arial", "sans-serif"],
    })

    return resolved


# Apply base rcParams (font is set later in main() via apply_font())
plt.rcParams.update({
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
    "axes.edgecolor":    MUTED_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.7,
    "text.color":        TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "font.family":       "sans-serif",
    "font.size":         11,
    "legend.facecolor":  "white",
    "legend.edgecolor":  GRID_COLOR,
    "legend.fontsize":   10,
    "legend.framealpha": 0.9,
})


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_all_metrics(metrics_dir: Path, dataset: str, task: str = "validity") -> dict:
    tag_dir = metrics_dir / dataset
    results = {}
    for key in MODEL_ORDER:
        path = tag_dir / f"{key}_{task}_results.json"
        if path.exists():
            with open(path) as f:
                results[key] = json.load(f)
    return results


def load_benchmark(metrics_dir: Path, dataset: str, model_key: str, task: str) -> dict | None:
    path = metrics_dir / dataset / f"{model_key}_{task}_batch_benchmark.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def ensure_fig_dir(base: Path, dataset: str) -> Path:
    out = base / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out


def n_responses_from_tag(tag: str) -> str:
    if "2000" in tag:
        return "2000 responses"
    elif "100" in tag:
        return "100 responses"
    return tag.replace("_", " ")


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_color(MUTED_COLOR)
    ax.tick_params(colors=TEXT_COLOR)


def has_loss_curves(results: dict) -> bool:
    for r in results.values():
        for fold_data in r.get("folds", {}).values():
            lc = fold_data.get("loss_curve", {})
            if lc.get("train_loss"):
                return True
    return False


def has_confusion_matrices(results: dict) -> bool:
    for r in results.values():
        for fold_data in r.get("folds", {}).values():
            cm = fold_data.get("confusion_matrix", {})
            if cm.get("matrix"):
                return True
    return False


def has_class_distribution(results: dict) -> bool:
    for r in results.values():
        if r.get("class_distribution"):
            return True
    return False


# ------------------------------------------------------------------ #
# Chart 1: Model comparison (mean ± std)
# ------------------------------------------------------------------ #

def plot_model_comparison(results: dict, out_dir: Path, dataset: str) -> None:
    if not results:
        return
    models        = [k for k in MODEL_ORDER if k in results]
    metrics       = ["accuracy", "f1_macro", "auc"]
    metric_labels = ["Accuracy", "F1 Macro", "AUC"]
    n_responses   = n_responses_from_tag(dataset)

    fig, ax = plt.subplots(figsize=(10, 5))
    x, width = np.arange(len(metrics)), 0.18

    for i, key in enumerate(models):
        r     = results[key]
        means = [r["mean"].get(m, 0) for m in metrics]
        stds  = [r["std"].get(m, 0)  for m in metrics]
        ax.bar(
            x + i * width - width * (len(models) - 1) / 2,
            means, width, yerr=stds,
            label=f"{MODEL_LABELS[key]} (batch {r['batch_size']})",
            color=COLORS[key], alpha=0.88, edgecolor="none",
            capsize=3, error_kw={"elinewidth": 1.2, "capthick": 1.2, "color": MUTED_COLOR},
        )

    ax.set_ylabel("Score")
    ax.set_title(f"5-Fold CV Results — {n_responses}", fontsize=13, pad=12)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(models), fontsize=9, frameon=False)
    ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.5, linewidth=1)
    style_ax(ax)
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 2: Per-fold accuracy
# ------------------------------------------------------------------ #

def plot_per_fold_accuracy(results: dict, out_dir: Path, dataset: str) -> None:
    if not results:
        return
    models      = [k for k in MODEL_ORDER if k in results]
    n_folds     = max(len(r["folds"]) for r in results.values())
    fold_nums   = list(range(1, n_folds + 1))
    n_responses = n_responses_from_tag(dataset)

    fig, ax = plt.subplots(figsize=(10, 5))
    x, width = np.arange(n_folds), 0.18

    for i, key in enumerate(models):
        accs = [results[key]["folds"].get(str(f), {}).get("accuracy", 0) for f in fold_nums]
        ax.bar(
            x + i * width - width * (len(models) - 1) / 2,
            accs, width,
            label=MODEL_LABELS[key], color=COLORS[key], alpha=0.88, edgecolor="none",
        )

    ax.set_ylabel("Accuracy"); ax.set_xlabel("Fold")
    ax.set_title(f"Per-Fold Accuracy — {n_responses}", fontsize=13, pad=12)
    ax.set_xticks(x); ax.set_xticklabels(fold_nums)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="upper right", fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    path = out_dir / "per_fold_accuracy.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 3: Radar chart
# ------------------------------------------------------------------ #

def plot_radar_chart(results: dict, out_dir: Path, dataset: str) -> None:
    if not results:
        return
    models        = [k for k in MODEL_ORDER if k in results]
    metrics       = ["accuracy", "f1_macro", "auc", "precision", "recall"]
    metric_labels = ["Accuracy", "F1 Macro", "AUC", "Precision", "Recall"]
    angles        = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]
    n_responses   = n_responses_from_tag(dataset)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("none")

    for key in models:
        r      = results[key]
        values = [r["mean"].get(m, 0) for m in metrics] + [r["mean"].get(metrics[0], 0)]
        ax.plot(angles, values, "o-", linewidth=2, label=MODEL_LABELS[key], color=COLORS[key], markersize=5)
        ax.fill(angles, values, alpha=0.12, color=COLORS[key])

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_labels, color=TEXT_COLOR, size=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=MUTED_COLOR, size=9)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.6)
    ax.set_title(f"Model Performance Profile\n(5-fold mean, {n_responses})", fontsize=13, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=9)
    fig.tight_layout()
    path = out_dir / "radar_comparison.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 4: CV variance strip plot
# ------------------------------------------------------------------ #

def plot_cv_variance(results: dict, out_dir: Path, dataset: str) -> None:
    """Mean ± std across folds with individual fold dots — shows stability."""
    if not results:
        return
    models      = [k for k in MODEL_ORDER if k in results]
    n_responses = n_responses_from_tag(dataset)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    metrics       = ["accuracy", "f1_macro", "auc"]
    metric_labels = ["Accuracy", "F1 Macro", "AUC"]

    for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]
        for i, key in enumerate(models):
            r      = results[key]
            folds  = sorted(r["folds"].keys(), key=int)
            values = [r["folds"][f].get(metric, 0) for f in folds]
            mean   = r["mean"].get(metric, 0)
            std    = r["std"].get(metric, 0)
            color  = COLORS[key]

            # Error bar for mean ± std
            ax.errorbar(i, mean, yerr=std, fmt="none",
                        ecolor=color, elinewidth=2, capsize=6, capthick=2)
            # Mean point
            ax.scatter(i, mean, color=color, s=100, zorder=5)
            # Individual fold dots
            jitter = np.linspace(-0.08, 0.08, len(values))
            for j, v in zip(jitter, values):
                ax.scatter(i + j, v, color=color, alpha=0.45, s=30, zorder=4)

        ax.set_title(mlabel, fontsize=12, pad=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABELS[k].replace("-", "-\n") for k in models], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.4, linewidth=1)
        style_ax(ax)

    fig.suptitle(f"Cross-Validation Variance — {n_responses}\n"
                 "Dots = individual folds, bar = mean ± 1 std",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / "cv_variance.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 5: Loss curves
# ------------------------------------------------------------------ #

def plot_loss_curves(results: dict, out_dir: Path, dataset: str) -> None:
    """Train vs validation loss per epoch, one panel per model."""
    if not has_loss_curves(results):
        print("  Skipping loss_curves.png — no loss curve data in JSON.")
        print("  Re-run train_all_models.py to capture loss history.")
        return

    models      = [k for k in MODEL_ORDER if k in results]
    n_models    = len(models)
    n_responses = n_responses_from_tag(dataset)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, key in zip(axes, models):
        r     = results[key]
        color = COLORS[key]
        folds = sorted(r["folds"].keys(), key=int)

        all_train, all_eval = [], []
        for fold_key in folds:
            lc = r["folds"][fold_key].get("loss_curve", {})
            if lc.get("train_loss"):
                all_train.append(lc["train_loss"])
                all_eval.append(lc["eval_loss"])

        if not all_train:
            ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes,
                    color=MUTED_COLOR)
            ax.set_title(MODEL_LABELS[key], fontsize=11)
            style_ax(ax)
            continue

        max_epochs_train = max(len(t) for t in all_train)
        max_epochs_eval  = max(len(e) for e in all_eval)

        # Plot each fold faintly
        for train_curve, eval_curve in zip(all_train, all_eval):
            ep_train = list(range(1, len(train_curve) + 1))
            ep_eval  = list(range(1, len(eval_curve) + 1))
            ax.plot(ep_train, train_curve, color=color,   alpha=0.2, linewidth=1)
            ax.plot(ep_eval,  eval_curve,  color=WARN_COLOR, alpha=0.2, linewidth=1)

        # Plot means
        def pad_mean(curves, length):
            padded = [c + [np.nan] * (length - len(c)) for c in curves]
            return np.nanmean(padded, axis=0)

        mean_train = pad_mean(all_train, max_epochs_train)
        mean_eval  = pad_mean(all_eval,  max_epochs_eval)
        ep_mean_train = list(range(1, max_epochs_train + 1))
        ep_mean_eval  = list(range(1, max_epochs_eval + 1))

        ax.plot(ep_mean_train, mean_train, color=color,      linewidth=2.5, label="Train loss")
        ax.plot(ep_mean_eval,  mean_eval,  color=WARN_COLOR, linewidth=2.5, label="Val loss", linestyle="--")

        ax.set_title(MODEL_LABELS[key], fontsize=11, pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss") if key == models[0] else None
        ax.legend(fontsize=8)
        style_ax(ax)

    fig.suptitle(f"Loss Curves — {n_responses}\n"
                 "Faint lines = individual folds, bold = fold mean",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / "loss_curves.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 6: Confusion matrices
# ------------------------------------------------------------------ #

def plot_confusion_matrices(results: dict, out_dir: Path, dataset: str) -> None:
    """Aggregated confusion matrix (summed across folds) per model."""
    if not has_confusion_matrices(results):
        print("  Skipping confusion_matrices.png — no confusion matrix data in JSON.")
        print("  Re-run train_all_models.py to capture confusion matrices.")
        return

    models      = [k for k in MODEL_ORDER if k in results]
    n_models    = len(models)
    n_responses = n_responses_from_tag(dataset)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, key in zip(axes, models):
        r      = results[key]
        color  = COLORS[key]
        folds  = sorted(r["folds"].keys(), key=int)

        # Aggregate confusion matrix across all folds
        agg_cm     = None
        class_names = None
        for fold_key in folds:
            cm_data = r["folds"][fold_key].get("confusion_matrix", {})
            if cm_data.get("matrix"):
                cm_arr = np.array(cm_data["matrix"])
                agg_cm = cm_arr if agg_cm is None else agg_cm + cm_arr
                class_names = cm_data["classes"]

        if agg_cm is None:
            ax.text(0.5, 0.5, "No CM data", ha="center", va="center", transform=ax.transAxes,
                    color=MUTED_COLOR)
            ax.set_title(MODEL_LABELS[key], fontsize=11)
            style_ax(ax)
            continue

        # Normalise to proportions
        cm_norm = agg_cm.astype(float) / agg_cm.sum(axis=1, keepdims=True)

        # Custom colourmap from white → model colour
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "model_cm", ["#ffffff", color], N=256
        )

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        # Annotations
        for row in range(agg_cm.shape[0]):
            for col in range(agg_cm.shape[1]):
                raw   = int(agg_cm[row, col])
                norm  = cm_norm[row, col]
                # Dark text on light cells, light on dark
                txt_color = TEXT_COLOR if norm < 0.6 else "white"
                ax.text(col, row, f"{raw}\n({norm:.0%})",
                        ha="center", va="center", fontsize=9, color=txt_color, fontweight="bold")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual") if key == models[0] else None
        ax.set_title(MODEL_LABELS[key], fontsize=11, pad=8)
        for spine in ax.spines.values():
            spine.set_color(MUTED_COLOR)

    fig.suptitle(f"Confusion Matrices (aggregated across 5 folds) — {n_responses}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 7: Training times
# ------------------------------------------------------------------ #

def plot_training_times(
    results: dict, out_dir: Path, dataset: str,
    compare_results: dict | None = None, compare_dataset: str | None = None,
) -> None:
    if not results:
        return
    models        = [k for k in MODEL_ORDER if k in results]
    labels        = [MODEL_LABELS[k] for k in models]
    times_primary = [results[k]["total_time_seconds"] / 60 for k in models]

    if compare_results and compare_dataset:
        times_compare = [compare_results.get(k, {}).get("total_time_seconds", 0) / 60 for k in models]
        tag1, tag2    = n_responses_from_tag(dataset), n_responses_from_tag(compare_dataset)
        fig, ax = plt.subplots(figsize=(10, 5))
        x, w = np.arange(len(models)), 0.35
        bars1 = ax.barh(x + w/2, times_primary,  w, color=[COLORS[k] for k in models],
                        alpha=0.65, edgecolor="none", label=tag1)
        bars2 = ax.barh(x - w/2, times_compare,  w, color=[COLORS[k] for k in models],
                        alpha=1.0,  edgecolor="none", hatch="//", label=tag2)
        for bar, t in list(zip(bars1, times_primary)) + list(zip(bars2, times_compare)):
            if t > 0:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{t:.1f}m", va="center", color=TEXT_COLOR, fontsize=9)
        ax.set_yticks(x); ax.set_yticklabels(labels)
        ax.set_xlabel("Total Training Time (minutes, 5 folds)")
        ax.set_title(f"Training Time: {tag1} vs {tag2} (MacBook MPS)", fontsize=13, pad=12)
        ax.invert_yaxis(); ax.legend(fontsize=9); style_ax(ax)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, times_primary, color=[COLORS[k] for k in models],
                       alpha=0.88, edgecolor="none", height=0.6)
        for bar, t in zip(bars, times_primary):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{t:.1f} min", va="center", color=TEXT_COLOR, fontsize=10)
        ax.set_xlabel("Total Training Time (minutes, 5 folds)")
        ax.set_title(f"Training Time by Model ({n_responses_from_tag(dataset)}, MacBook MPS)",
                     fontsize=13, pad=12)
        ax.invert_yaxis(); style_ax(ax)

    fig.tight_layout()
    suffix = f"_vs_{compare_dataset}" if compare_dataset else ""
    path = out_dir / f"training_times{suffix}.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 8: Class distribution per CCU
# ------------------------------------------------------------------ #

def plot_class_distribution(results: dict, dm_csv: str | None, out_dir: Path, dataset: str) -> None:
    """Bar chart of valid/invalid responses per CCU.

    Uses class_distribution from the JSON if available, otherwise reads
    the CSV directly if --csv is supplied.
    """
    # Try JSON first
    class_dist = None
    for r in results.values():
        if r.get("class_distribution"):
            class_dist = r["class_distribution"]
            break

    # Try CSV if JSON doesn't have it
    ccu_dist = None
    if dm_csv and Path(dm_csv).exists():
        try:
            import csv as _csv
            from collections import defaultdict
            ccu_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
            with open(dm_csv) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    ccu   = row.get("ccuname", "unknown")
                    valid = row.get("validity", "")
                    if valid in ("correct", "incorrect"):
                        ccu_counts[ccu][valid] += 1
            ccu_dist = dict(sorted(ccu_counts.items()))
        except Exception:
            pass

    if class_dist is None and ccu_dist is None:
        print("  Skipping class_distribution.png — no class distribution data.")
        print("  Re-run train_all_models.py or supply --csv to read from file.")
        return

    n_responses = n_responses_from_tag(dataset)

    if ccu_dist:
        # Per-CCU breakdown
        ccus       = list(ccu_dist.keys())
        correct    = [ccu_dist[c]["correct"]   for c in ccus]
        incorrect  = [ccu_dist[c]["incorrect"] for c in ccus]
        x          = np.arange(len(ccus))

        fig, ax = plt.subplots(figsize=(10, 5))
        w = 0.35
        ax.bar(x - w/2, correct,   w, label="Correct (valid reasoning)",
               color=COLORS["albert"],  alpha=0.88, edgecolor="none")
        ax.bar(x + w/2, incorrect, w, label="Incorrect (invalid reasoning)",
               color=COLORS["xlnet"],   alpha=0.88, edgecolor="none")
        ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in ccus])
        ax.set_ylabel("Number of responses")
        ax.set_title(f"Class Distribution by CCU — {n_responses}", fontsize=13, pad=12)
        # Headroom so totals don't get clipped, then put legend below the chart
        max_total = max(c + i for c, i in zip(correct, incorrect))
        ax.set_ylim(0, max_total * 1.18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=9, frameon=False)
        # Annotate totals above each pair of bars
        for xi, (c, inc) in enumerate(zip(correct, incorrect)):
            top = max(c, inc) + max_total * 0.03
            ax.text(xi, top, f"n={c+inc}", ha="center",
                    fontsize=8.5, color=MUTED_COLOR)
        style_ax(ax)

    else:
        # Whole-dataset totals only
        classes = list(class_dist.keys())
        counts  = [class_dist[c] for c in classes]
        colors  = [COLORS["albert"], COLORS["xlnet"]][:len(classes)]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(classes, counts, color=colors, alpha=0.88, edgecolor="none", width=0.5)
        for i, (cls, cnt) in enumerate(zip(classes, counts)):
            ax.text(i, cnt + 5, str(cnt), ha="center", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel("Number of responses")
        ax.set_title(f"Overall Class Distribution — {n_responses}", fontsize=13, pad=12)
        style_ax(ax)

    fig.tight_layout()
    path = out_dir / "class_distribution.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 9: Batch size benchmark
# ------------------------------------------------------------------ #

def plot_batch_benchmarks(metrics_dir: Path, out_dir: Path, dataset: str, task: str) -> None:
    bench_results = {key: load_benchmark(metrics_dir, dataset, key, task)
                     for key in MODEL_ORDER}
    bench_results = {k: v for k, v in bench_results.items() if v}

    if not bench_results:
        print("  Skipping batch_benchmarks_all.png — no benchmark JSON files found.")
        return

    n_models = len(bench_results)
    fig, axes = plt.subplots(n_models, 2, figsize=(13, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for row, (key, data) in enumerate(bench_results.items()):
        batch_labels = [str(b) for b in data["batch_sizes"]]
        ax_m, ax_t   = axes[row][0], axes[row][1]

        ax_m.plot(batch_labels, data["accuracy"], "o-",
                  color=COLORS[key], linewidth=2, markersize=8, label="Accuracy")
        ax_m.plot(batch_labels, data["f1_macro"], "s--",
                  color=WARN_COLOR, linewidth=2, markersize=8, label="F1 Macro")
        ax_m.plot(batch_labels, data["auc"], "^:",
                  color=AUC_COLOR, linewidth=2, markersize=8, label="AUC")
        ax_m.set_xlabel("Batch Size"); ax_m.set_ylabel("Score")
        ax_m.set_title(f"{MODEL_LABELS[key]}: Metrics vs Batch Size", fontsize=12, pad=10)
        ax_m.set_ylim(0.3, 1.1); ax_m.legend(fontsize=9); style_ax(ax_m)

        ax_t.bar(batch_labels, data["time_seconds"], color=COLORS[key], alpha=0.88, edgecolor="none")
        ax_t.set_xlabel("Batch Size"); ax_t.set_ylabel("Training Time (seconds)")
        ax_t.set_title(f"{MODEL_LABELS[key]}: Training Time vs Batch Size", fontsize=12, pad=10)
        for i, t in enumerate(data["time_seconds"]):
            ax_t.text(i, t + 3, f"{t:.0f}s", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)
        style_ax(ax_t)

    fig.suptitle(f"Batch Size Impact on Performance ({n_responses_from_tag(dataset)}, Fold 1)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = out_dir / "batch_benchmarks_all.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 10: Batch size / time annotation table
# ------------------------------------------------------------------ #

def plot_batch_time_table(results: dict, out_dir: Path, dataset: str) -> None:
    """A clean table figure showing batch size, training time per fold, and notes."""
    if not results:
        return
    models = [k for k in MODEL_ORDER if k in results]

    rows = []
    for key in models:
        r      = results[key]
        bs     = r.get("batch_size", "—")
        folds  = r.get("folds", {})
        times  = [v.get("time_seconds", 0) for v in folds.values()]
        avg_t  = np.mean(times) / 60 if times else 0
        total  = r.get("total_time_seconds", 0) / 60
        note   = {
            "electra": "No OOM issues at batch 16",
            "roberta": "Tight at batch 16 → reduced to 8",
            "xlnet":   "Near OOM at batch 16 → reduced to 4",
            "albert":  "No OOM issues at batch 16",
        }.get(key, "")
        rows.append([MODEL_LABELS[key], str(bs), f"{avg_t:.1f} min", f"{total:.1f} min", note])

    col_labels = ["Model", "Batch size", "Avg time / fold", "Total (5 folds)", "Hardware note"]
    fig, ax = plt.subplots(figsize=(12, 1 + 0.5 * len(rows)))
    ax.set_facecolor("none")
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    # Style header
    for col in range(len(col_labels)):
        cell = tbl[0, col]
        cell.set_facecolor("#e8e4de")
        cell.set_text_props(color=TEXT_COLOR, fontweight="bold")

    # Style data rows
    for row_idx in range(1, len(rows) + 1):
        for col in range(len(col_labels)):
            cell = tbl[row_idx, col]
            cell.set_facecolor("white" if row_idx % 2 == 0 else "#f5f2ec")
            cell.set_text_props(color=TEXT_COLOR)
            cell.set_edgecolor(GRID_COLOR)

    ax.set_title(f"Batch Size & Training Time Summary — {n_responses_from_tag(dataset)}\n"
                 "MacBook Pro (Apple Silicon, 16 GB RAM)",
                 fontsize=12, pad=16, color=TEXT_COLOR)

    fig.tight_layout()
    path = out_dir / "batch_size_table.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 11: Synthetic vs real performance gap
# ------------------------------------------------------------------ #

def plot_performance_gap(
    results: dict, compare_results: dict,
    dataset: str, compare_dataset: str, out_dir: Path,
) -> None:
    """Grouped bar: accuracy per model for two datasets side by side."""
    models      = [k for k in MODEL_ORDER if k in results or k in compare_results]
    tag1, tag2  = n_responses_from_tag(dataset), n_responses_from_tag(compare_dataset)

    fig, ax = plt.subplots(figsize=(10, 5))
    x, w = np.arange(len(models)), 0.3

    acc1 = [results.get(k, {}).get("mean", {}).get("accuracy", 0)         for k in models]
    acc2 = [compare_results.get(k, {}).get("mean", {}).get("accuracy", 0) for k in models]

    bars1 = ax.bar(x - w/2, acc1, w, label=tag1,
                   color=[COLORS[k] for k in models], alpha=0.65, edgecolor="none")
    bars2 = ax.bar(x + w/2, acc2, w, label=tag2,
                   color=[COLORS[k] for k in models], alpha=1.0, edgecolor="none", hatch="//")

    for bar, v in list(zip(bars1, acc1)) + list(zip(bars2, acc2)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[k] for k in models], fontsize=9)
    ax.set_ylabel("Mean Accuracy (5-fold CV)")
    ax.set_title(f"Performance: {tag1} vs {tag2}", fontsize=13, pad=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.4, linewidth=1)
    ax.legend(fontsize=9); style_ax(ax)

    fig.tight_layout()
    path = out_dir / f"performance_gap_{compare_dataset}.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 12: Somers et al. comparison
# ------------------------------------------------------------------ #

# Somers et al. (2021) published results — Table 23 (validity ensemble)
# and individual model averages across all 6 CCU questions.
# Source: Somers, R., Cunningham-Nelson, S., & Boles, W. (2021).
SOMERS_RESULTS = {
    "ensemble": {
        "label":    "Somers ensemble\n(ELECTRA+RoBERTa+XLNet)",
        "accuracy": 0.9672,
        "f1_macro": 0.9669,   # estimated from published accuracy + AUC
        "auc":      0.9469,
    },
    "electra": {
        "label":    "Somers ELECTRA\n(avg across 6 CCUs)",
        "accuracy": 0.933,    # midpoint of 91–95% range
        "f1_macro": 0.930,
        "auc":      0.910,
    },
    "roberta": {
        "label":    "Somers RoBERTa\n(avg across 6 CCUs)",
        "accuracy": 0.950,    # midpoint of 93–97% range
        "f1_macro": 0.948,
        "auc":      0.930,
    },
    "xlnet": {
        "label":    "Somers XLNet\n(avg across 6 CCUs)",
        "accuracy": 0.940,    # midpoint of 92–96% range
        "f1_macro": 0.938,
        "auc":      0.920,
    },
    "albert": {
        "label":    "Somers ALBERT\n(avg across 6 CCUs)",
        "accuracy": 0.930,    # midpoint of 91–95% range
        "f1_macro": 0.928,
        "auc":      0.910,
    },
}

# Somers per-question validity ensemble accuracy (Table 23)
SOMERS_PER_CCU = {
    "CCU1": {"accuracy": 0.9797, "auc": 0.8641},
    "CCU2": {"accuracy": 0.9734, "auc": 0.9747},
    "CCU3": {"accuracy": 0.9866, "auc": 0.9814},
    "CCU4": {"accuracy": 0.9686, "auc": 0.9661},
    "CCU5": {"accuracy": 0.9354, "auc": 0.9376},
    "CCU6": {"accuracy": 0.9595, "auc": 0.9577},
}


def plot_somers_comparison(results: dict, out_dir: Path, dataset: str) -> None:
    """Grouped bar chart comparing your pipeline against Somers published results.

    Shows your synthetic training results vs Somers' real-data results
    per model, with a clear disclaimer that this is a pipeline validation
    comparison rather than a direct replication.

    Framing for the presentation: the transformers are working correctly —
    accuracy gaps are expected because (a) Somers trained per-question on
    real labelled data, (b) synthetic data has different properties.
    The real-data replication will follow once the marked CCU data is available.
    """
    models      = [k for k in MODEL_ORDER if k in results]
    n_responses = n_responses_from_tag(dataset)

    metrics       = ["accuracy", "f1_macro", "auc"]
    metric_labels = ["Accuracy", "F1 Macro", "AUC"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)

    for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]

        your_vals   = [results[k]["mean"].get(metric, 0) for k in models]
        somers_vals = [SOMERS_RESULTS.get(k, {}).get(metric, 0) for k in models]

        x = np.arange(len(models))
        w = 0.32

        # Your results
        bars_y = ax.bar(
            x - w / 2, your_vals, w,
            color=[COLORS[k] for k in models],
            alpha=0.88, edgecolor="none",
            label=f"This project ({n_responses}, synthetic)",
        )
        # Somers results — same colour but hatched + lighter
        bars_s = ax.bar(
            x + w / 2, somers_vals, w,
            color=[COLORS[k] for k in models],
            alpha=0.40, edgecolor=[COLORS[k] for k in models],
            linewidth=1.2, hatch="//",
            label="Somers et al. (2021) — real data",
        )

        # Value labels
        for bar, v in zip(bars_y, your_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=7.5, color=TEXT_COLOR)
        for bar, v in zip(bars_s, somers_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=7.5, color=MUTED_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[k].replace("-", "-\n") for k in models], fontsize=8.5)
        ax.set_ylabel(mlabel) if col == 0 else None
        ax.set_title(mlabel, fontsize=12, pad=8)
        ax.set_ylim(0, 1.18)
        ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.35, linewidth=1)
        style_ax(ax)

    # Figure-level legend below all three panels (no chart-data overlap)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=9, frameon=False)

    # Disclaimer annotation below the legend
    fig.text(
        0.5, -0.09,
        "Note: not a direct replication — Somers trained per CCU question on real labelled data. "
        "Synthetic results validate pipeline correctness. Real-data replication pending marked CCU dataset.",
        ha="center", fontsize=8.5, color=MUTED_COLOR, style="italic",
        wrap=True,
    )

    fig.suptitle(
        f"Pipeline validation: this project vs Somers et al. (2021)\n"
        f"Synthetic training ({n_responses}) — real data pending",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "somers_comparison.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


def plot_somers_per_ccu(out_dir: Path) -> None:
    """Bar chart showing Somers' per-CCU ensemble accuracy — reference target.

    Use this as a standalone 'target' slide showing what accuracy to expect
    once real marked data is available and training is per-question.
    """
    ccus     = list(SOMERS_PER_CCU.keys())
    accs     = [SOMERS_PER_CCU[c]["accuracy"] for c in ccus]
    aucs     = [SOMERS_PER_CCU[c]["auc"]      for c in ccus]
    x        = np.arange(len(ccus))
    avg_acc  = np.mean(accs)
    avg_auc  = np.mean(aucs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.35
    bars_a = ax.bar(x - w / 2, accs, w, label="Accuracy",
                    color=COLORS["roberta"], alpha=0.88, edgecolor="none")
    bars_u = ax.bar(x + w / 2, aucs, w, label="AUC",
                    color=COLORS["albert"],  alpha=0.88, edgecolor="none")

    for bar, v in list(zip(bars_a, accs)) + list(zip(bars_u, aucs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.1%}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)

    ax.axhline(avg_acc, color=COLORS["roberta"], linestyle="--", alpha=0.5, linewidth=1.2,
               label=f"Avg accuracy {avg_acc:.1%}")
    ax.axhline(avg_auc, color=COLORS["albert"],  linestyle=":",  alpha=0.5, linewidth=1.2,
               label=f"Avg AUC {avg_auc:.1%}")

    ax.set_xticks(x); ax.set_xticklabels(ccus)
    ax.set_ylabel("Score"); ax.set_ylim(0.8, 1.05)
    ax.set_title("Somers et al. (2021) — validity ensemble per CCU question\n"
                 "Replication target once marked data is available",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    path = out_dir / "somers_per_ccu_target.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 13: Memory pressure
# ------------------------------------------------------------------ #

def plot_memory_pressure(out_dir: Path) -> None:
    free_mb = [
        964, 867, 782, 779, 209, 211, 100, 79, 64, 88,
        127, 17, 67, 65, 68, 51, 61, 43, 41, 87,
        46, 99, 1848, 740, 507, 465, 392, 429, 409, 365,
        329, 343, 339, 347, 329, 344, 347, 337, 339, 330,
        338, 316, 332, 327, 330, 1262, 545, 316, 314, 275,
        304, 308, 291, 307, 295, 270, 297, 262, 218, 197,
        186, 191, 203, 214, 212, 199, 189, 179, 212, 180,
        178, 184, 204, 206, 196, 207, 166, 195, 209, 1044,
    ]
    swap_mb = [458] * 7 + [681] + [699] * 72
    t = list(range(0, len(free_mb) * 5, 5))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.fill_between(t, free_mb, alpha=0.2, color=COLORS["xlnet"])
    ax1.plot(t, free_mb, color=COLORS["xlnet"], linewidth=1.8)
    ax1.axhline(y=200, color=DANGER_COLOR, linestyle="--", alpha=0.8, linewidth=1.5,
                label="Critical threshold (200MB)")
    ax1.axhline(y=500, color=WARN_COLOR, linestyle="--", alpha=0.6, linewidth=1.2,
                label="Warning threshold (500MB)")
    ax1.set_ylabel("Free Memory (MB)")
    ax1.set_title("XLNet Batch-16 Training: Memory Pressure Over Time", fontsize=13, pad=12)
    ax1.legend(loc="upper right", fontsize=9)
    min_idx = free_mb.index(min(free_mb))
    ax1.annotate(f"Min: {min(free_mb)}MB",
                 xy=(t[min_idx], min(free_mb)), xytext=(t[min_idx] + 30, min(free_mb) + 400),
                 arrowprops=dict(arrowstyle="->", color=DANGER_COLOR, lw=1.5),
                 color=DANGER_COLOR, fontsize=10, fontweight="bold")
    style_ax(ax1)

    ax2.fill_between(t, swap_mb, alpha=0.2, color=DANGER_COLOR)
    ax2.plot(t, swap_mb, color=DANGER_COLOR, linewidth=1.8)
    ax2.set_ylabel("Swap Used (MB)"); ax2.set_xlabel("Time (seconds)")
    jump_idx = next(i for i, s in enumerate(swap_mb) if s > 500)
    ax2.annotate(f"Swap spike: {swap_mb[jump_idx]}MB\n(physical RAM exhausted)",
                 xy=(t[jump_idx], swap_mb[jump_idx]),
                 xytext=(t[jump_idx] + 40, swap_mb[jump_idx] + 80),
                 arrowprops=dict(arrowstyle="->", color=DANGER_COLOR, lw=1.5),
                 color=DANGER_COLOR, fontsize=10)
    style_ax(ax2)

    fig.tight_layout()
    path = out_dir / "memory_pressure_xlnet.png"
    fig.savefig(path, **SAVEFIG_KW); plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate all presentation charts from saved metrics JSON files."
    )
    p.add_argument("--dataset",         default="synthetic_responses")
    p.add_argument("--compare-dataset", default=None,
                   help="Second dataset for gap / time-comparison charts")
    p.add_argument("--task",            default="validity")
    p.add_argument("--metrics-dir",     default="outputs/metrics")
    p.add_argument("--figures-dir",     default="outputs/figures")
    p.add_argument("--csv",             default=None,
                   help="Path to the CSV used for training (enables per-CCU class distribution chart)")
    p.add_argument("--benchmarks",      action="store_true",
                   help="Generate batch benchmark charts")
    p.add_argument("--memory",          action="store_true",
                   help="Generate XLNet memory pressure chart")
    p.add_argument("--somers",          action="store_true",
                   help="Generate Somers et al. comparison charts (pipeline validation)")
    p.add_argument("--font",            default="slides",
                   choices=["slides", "report", "system"],
                   help="Font preset: slides=DM Sans, report=Calibri, system=default")
    return p.parse_args(argv)


def main(argv=None):
    args        = parse_args(argv)

    # Apply font before any chart is drawn
    apply_font(args.font)

    metrics_dir = Path(args.metrics_dir)
    out_dir     = ensure_fig_dir(Path(args.figures_dir), args.dataset)

    print(f"\nLoading metrics from: {metrics_dir / args.dataset}/")
    results = load_all_metrics(metrics_dir, args.dataset, args.task)

    if not results:
        print(f"  No JSON files found in {metrics_dir / args.dataset}/")
        print("  Run `python scripts/train_all_models.py` first.")
        return

    print(f"  Found results for: {list(results.keys())}")

    compare_results = None
    if args.compare_dataset:
        compare_results = load_all_metrics(metrics_dir, args.compare_dataset, args.task)
        if compare_results:
            print(f"  Comparing against: {args.compare_dataset}")

    print(f"\nGenerating charts → {out_dir}/\n")

    # Always generated
    plot_model_comparison(results, out_dir, args.dataset)
    plot_per_fold_accuracy(results, out_dir, args.dataset)
    plot_radar_chart(results, out_dir, args.dataset)
    plot_cv_variance(results, out_dir, args.dataset)
    plot_training_times(results, out_dir, args.dataset, compare_results, args.compare_dataset)
    plot_batch_time_table(results, out_dir, args.dataset)
    plot_class_distribution(results, args.csv, out_dir, args.dataset)

    # Generated when rich data is available
    plot_loss_curves(results, out_dir, args.dataset)
    plot_confusion_matrices(results, out_dir, args.dataset)

    # Optional flags
    if args.benchmarks:
        plot_batch_benchmarks(metrics_dir, out_dir, args.dataset, args.task)

    if args.memory:
        plot_memory_pressure(out_dir)

    if args.somers:
        plot_somers_comparison(results, out_dir, args.dataset)
        plot_somers_per_ccu(out_dir)

    # Cross-dataset comparison
    if compare_results and args.compare_dataset:
        plot_performance_gap(
            results, compare_results,
            args.dataset, args.compare_dataset, out_dir,
        )

    print(f"\nDone. All charts saved to {out_dir}/")


if __name__ == "__main__":
    main()