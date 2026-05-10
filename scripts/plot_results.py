"""Generate all comparison charts from saved JSON metric files.

Reads results produced by train_all_models.py and benchmark_batch_sizes.py.
No manual data entry required — just point at a dataset tag and it builds
all charts automatically from whatever models have JSON results.

Usage
-----
Plot results for the 100-row synthetic dataset::

    python scripts/plot_results.py --dataset synthetic_responses

Plot results for the 2000-row dataset::

    python scripts/plot_results.py --dataset synthetic_responses_2000

Compare training times across both dataset sizes::

    python scripts/plot_results.py --dataset synthetic_responses --compare-dataset synthetic_responses_2000

Plot all benchmark results (batch size charts)::

    python scripts/plot_results.py --dataset synthetic_responses --benchmarks

All charts saved to outputs/figures/<dataset>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------ #
# Model display config (order, labels, colors)
# ------------------------------------------------------------------ #

MODEL_ORDER = ["electra", "roberta", "xlnet", "albert"]
MODEL_LABELS = {
    "electra": "ELECTRA-small",
    "roberta": "RoBERTa-base",
    "xlnet": "XLNet-base",
    "albert": "ALBERT-base-v2",
}
COLORS = {
    "electra": "#4682dc",
    "roberta": "#966ee6",
    "xlnet": "#e67850",
    "albert": "#3cc8aa",
}

BG_COLOR = "#0c1020"
SURFACE_COLOR = "#141828"
TEXT_COLOR = "#d0d6e4"
GRID_COLOR = "#1e2438"
MUTED_COLOR = "#707890"
DANGER_COLOR = "#e05050"
WARN_COLOR = "#e6be46"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": SURFACE_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "text.color": TEXT_COLOR,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "font.family": "sans-serif",
    "font.size": 11,
    "legend.facecolor": SURFACE_COLOR,
    "legend.edgecolor": GRID_COLOR,
    "legend.fontsize": 10,
})


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_all_metrics(metrics_dir: Path, dataset: str, task: str = "validity") -> dict:
    """Load all model JSON results for a given dataset tag."""
    tag_dir = metrics_dir / dataset
    results = {}
    for key in MODEL_ORDER:
        path = tag_dir / f"{key}_{task}_results.json"
        if path.exists():
            with open(path) as f:
                results[key] = json.load(f)
    return results


def load_benchmark(metrics_dir: Path, dataset: str, model_key: str, task: str = "validity") -> dict | None:
    """Load batch benchmark JSON for a model."""
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
    """Extract a human-readable dataset size label from the filename tag."""
    if "2000" in tag:
        return "2000 responses"
    elif "100" in tag:
        return "100 responses"
    else:
        return tag.replace("_", " ")


# ------------------------------------------------------------------ #
# Chart 1: Model comparison (mean ± std)
# ------------------------------------------------------------------ #

def plot_model_comparison(results: dict, out_dir: Path, dataset: str) -> None:
    if not results:
        print("  No results to plot for model_comparison")
        return

    models = [k for k in MODEL_ORDER if k in results]
    metrics = ["accuracy", "f1_macro", "auc"]
    metric_labels = ["Accuracy", "F1 Macro", "AUC"]
    n_responses = n_responses_from_tag(dataset)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.18

    for i, key in enumerate(models):
        r = results[key]
        means = [r["mean"].get(m, 0) for m in metrics]
        stds = [r["std"].get(m, 0) for m in metrics]
        label = f"{MODEL_LABELS[key]} (batch {r['batch_size']})"
        ax.bar(
            x + i * width - width * (len(models) - 1) / 2,
            means, width, yerr=stds,
            label=label, color=COLORS[key], alpha=0.85, edgecolor="none",
            capsize=3, error_kw={"elinewidth": 1, "capthick": 1, "color": MUTED_COLOR},
        )

    ax.set_ylabel("Score")
    ax.set_title(f"5-Fold CV Results — {n_responses}", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 2: Per-fold accuracy
# ------------------------------------------------------------------ #

def plot_per_fold_accuracy(results: dict, out_dir: Path, dataset: str) -> None:
    if not results:
        return

    models = [k for k in MODEL_ORDER if k in results]
    n_folds = max(len(r["folds"]) for r in results.values())
    fold_nums = list(range(1, n_folds + 1))
    n_responses = n_responses_from_tag(dataset)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_folds)
    width = 0.18

    for i, key in enumerate(models):
        r = results[key]
        accs = [r["folds"].get(str(f), {}).get("accuracy", 0) for f in fold_nums]
        ax.bar(
            x + i * width - width * (len(models) - 1) / 2,
            accs, width,
            label=MODEL_LABELS[key], color=COLORS[key], alpha=0.85, edgecolor="none",
        )

    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Fold Accuracy — {n_responses}", fontsize=13, pad=12)
    ax.set_xlabel("Fold")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_nums)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color=MUTED_COLOR, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    path = out_dir / "per_fold_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 3: Radar chart
# ------------------------------------------------------------------ #

def plot_radar_chart(results: dict, out_dir: Path, dataset: str) -> None:
    if not results:
        return

    models = [k for k in MODEL_ORDER if k in results]
    metrics = ["accuracy", "f1_macro", "auc", "precision", "recall"]
    metric_labels = ["Accuracy", "F1 Macro", "AUC", "Precision", "Recall"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    n_responses = n_responses_from_tag(dataset)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor(SURFACE_COLOR)

    for key in models:
        r = results[key]
        values = [r["mean"].get(m, 0) for m in metrics] + [r["mean"].get(metrics[0], 0)]
        ax.plot(angles, values, "o-", linewidth=2, label=MODEL_LABELS[key], color=COLORS[key], markersize=5)
        ax.fill(angles, values, alpha=0.1, color=COLORS[key])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, color=TEXT_COLOR, size=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=MUTED_COLOR, size=9)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.4)
    ax.set_title(f"Model Performance Profile\n(5-fold mean, {n_responses})", fontsize=13, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=9)

    fig.tight_layout()
    path = out_dir / "radar_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 4: Training time — single dataset or comparison
# ------------------------------------------------------------------ #

def plot_training_times(
    results: dict,
    out_dir: Path,
    dataset: str,
    compare_results: dict | None = None,
    compare_dataset: str | None = None,
) -> None:
    if not results:
        return

    models = [k for k in MODEL_ORDER if k in results]
    labels = [MODEL_LABELS[k] for k in models]
    times_primary = [results[k]["total_time_seconds"] / 60 for k in models]

    if compare_results and compare_dataset:
        # Side-by-side comparison of two datasets
        times_compare = [compare_results.get(k, {}).get("total_time_seconds", 0) / 60 for k in models]
        tag1 = n_responses_from_tag(dataset)
        tag2 = n_responses_from_tag(compare_dataset)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(models))
        w = 0.35

        bars1 = ax.barh(x + w / 2, times_primary, w, label=tag1,
                        color=[COLORS[k] for k in models], alpha=0.7, edgecolor="none")
        bars2 = ax.barh(x - w / 2, times_compare, w, label=tag2,
                        color=[COLORS[k] for k in models], alpha=1.0, edgecolor="none",
                        hatch="//")

        for bar, t in zip(bars1, times_primary):
            if t > 0:
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        f"{t:.1f}m", va="center", color=TEXT_COLOR, fontsize=9)
        for bar, t in zip(bars2, times_compare):
            if t > 0:
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        f"{t:.1f}m", va="center", color=TEXT_COLOR, fontsize=9)

        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Total Training Time (minutes, 5 folds)")
        ax.set_title(f"Training Time: {tag1} vs {tag2} (MacBook MPS)", fontsize=13, pad=12)
        ax.invert_yaxis()
        ax.legend(fontsize=9)

    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, times_primary,
                       color=[COLORS[k] for k in models], alpha=0.85, edgecolor="none", height=0.6)
        for bar, t in zip(bars, times_primary):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{t:.1f} min", va="center", color=TEXT_COLOR, fontsize=10)
        ax.set_xlabel("Total Training Time (minutes, 5 folds)")
        n_responses = n_responses_from_tag(dataset)
        ax.set_title(f"Training Time by Model ({n_responses}, MacBook MPS)", fontsize=13, pad=12)
        ax.invert_yaxis()

    fig.tight_layout()
    suffix = f"_vs_{compare_dataset}" if compare_dataset else ""
    path = out_dir / f"training_times{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 5: Batch size benchmarks for all models
# ------------------------------------------------------------------ #

def plot_batch_benchmarks(metrics_dir: Path, out_dir: Path, dataset: str, task: str = "validity") -> None:
    """Plot batch benchmarks for all models that have benchmark JSON files."""
    bench_results = {}
    for key in MODEL_ORDER:
        data = load_benchmark(metrics_dir, dataset, key, task)
        if data:
            bench_results[key] = data

    if not bench_results:
        print("  No batch benchmark JSON files found. Run benchmark_batch_sizes.py --save-json first.")
        return

    n_models = len(bench_results)
    fig, axes = plt.subplots(n_models, 2, figsize=(13, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for row, (key, data) in enumerate(bench_results.items()):
        batch_labels = [str(b) for b in data["batch_sizes"]]
        ax_m = axes[row][0]
        ax_t = axes[row][1]

        ax_m.plot(batch_labels, data["accuracy"], "o-", color=COLORS[key], linewidth=2, markersize=8, label="Accuracy")
        ax_m.plot(batch_labels, data["f1_macro"], "s--", color="#e6be46", linewidth=2, markersize=8, label="F1 Macro")
        ax_m.plot(batch_labels, data["auc"], "^:", color="#5aaaf0", linewidth=2, markersize=8, label="AUC")
        ax_m.set_xlabel("Batch Size")
        ax_m.set_ylabel("Score")
        ax_m.set_title(f"{MODEL_LABELS[key]}: Metrics vs Batch Size", fontsize=12, pad=10)
        ax_m.set_ylim(0.3, 1.1)
        ax_m.legend(fontsize=9)

        ax_t.bar(batch_labels, data["time_seconds"], color=COLORS[key], alpha=0.8, edgecolor="none")
        ax_t.set_xlabel("Batch Size")
        ax_t.set_ylabel("Training Time (seconds)")
        ax_t.set_title(f"{MODEL_LABELS[key]}: Training Time vs Batch Size", fontsize=12, pad=10)
        for i, t in enumerate(data["time_seconds"]):
            ax_t.text(i, t + 3, f"{t}s", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)

    WARN_COLOR = "#e6be46"
    n_responses = n_responses_from_tag(dataset)
    fig.suptitle(f"Batch Size Impact on Performance ({n_responses}, Fold 1)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = out_dir / "batch_benchmarks_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Chart 6: Memory pressure (hardcoded — from your terminal log)
# ------------------------------------------------------------------ #

def plot_memory_pressure(out_dir: Path) -> None:
    """Memory pressure during XLNet batch-16 training (from captured log)."""
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

    ax1.fill_between(t, free_mb, alpha=0.3, color=COLORS["xlnet"])
    ax1.plot(t, free_mb, color=COLORS["xlnet"], linewidth=1.5)
    ax1.axhline(y=200, color=DANGER_COLOR, linestyle="--", alpha=0.6, label="Critical threshold (200MB)")
    ax1.axhline(y=500, color=WARN_COLOR, linestyle="--", alpha=0.4, label="Warning threshold (500MB)")
    ax1.set_ylabel("Free Memory (MB)")
    ax1.set_title("XLNet Batch-16 Training: Memory Pressure Over Time", fontsize=13, pad=12)
    ax1.legend(loc="upper right", fontsize=9)

    min_free = min(free_mb)
    min_idx = free_mb.index(min_free)
    ax1.annotate(f"Min: {min_free}MB",
                 xy=(t[min_idx], min_free), xytext=(t[min_idx] + 30, min_free + 400),
                 arrowprops=dict(arrowstyle="->", color=DANGER_COLOR, lw=1.5),
                 color=DANGER_COLOR, fontsize=10, fontweight="bold")

    ax2.fill_between(t, swap_mb, alpha=0.3, color=DANGER_COLOR)
    ax2.plot(t, swap_mb, color=DANGER_COLOR, linewidth=1.5)
    ax2.set_ylabel("Swap Used (MB)")
    ax2.set_xlabel("Time (seconds)")

    swap_jump_idx = next(i for i, s in enumerate(swap_mb) if s > 500)
    ax2.annotate(f"Swap spike: {swap_mb[swap_jump_idx]}MB\n(physical RAM exhausted)",
                 xy=(t[swap_jump_idx], swap_mb[swap_jump_idx]),
                 xytext=(t[swap_jump_idx] + 40, swap_mb[swap_jump_idx] + 100),
                 arrowprops=dict(arrowstyle="->", color=DANGER_COLOR, lw=1.5),
                 color=DANGER_COLOR, fontsize=10)

    fig.tight_layout()
    path = out_dir / "memory_pressure_xlnet.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
# Parse args & main
# ------------------------------------------------------------------ #

WARN_COLOR = "#e6be46"

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate charts from saved metrics JSON files.")
    p.add_argument("--dataset", default="synthetic_responses",
                   help="Dataset tag (stem of CSV filename, e.g. synthetic_responses)")
    p.add_argument("--compare-dataset", default=None,
                   help="Second dataset tag to compare training times against")
    p.add_argument("--task", default="validity")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--benchmarks", action="store_true",
                   help="Also plot batch size benchmark charts")
    p.add_argument("--memory", action="store_true",
                   help="Also plot memory pressure chart (XLNet batch-16 log)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    out_dir = ensure_fig_dir(figures_dir, args.dataset)

    print(f"\nLoading metrics from: {metrics_dir / args.dataset}/")
    results = load_all_metrics(metrics_dir, args.dataset, args.task)

    if not results:
        print(f"  No JSON metric files found in {metrics_dir / args.dataset}/")
        print("  Run `python scripts/train_all_models.py` first to generate metrics.")
        return

    print(f"  Found results for: {list(results.keys())}")
    print(f"\nGenerating charts → {out_dir}/\n")

    compare_results = None
    if args.compare_dataset:
        compare_results = load_all_metrics(metrics_dir, args.compare_dataset, args.task)
        if compare_results:
            print(f"  Comparing against: {args.compare_dataset} ({list(compare_results.keys())})")

    plot_model_comparison(results, out_dir, args.dataset)
    plot_per_fold_accuracy(results, out_dir, args.dataset)
    plot_radar_chart(results, out_dir, args.dataset)
    plot_training_times(results, out_dir, args.dataset, compare_results, args.compare_dataset)

    if args.benchmarks:
        plot_batch_benchmarks(metrics_dir, out_dir, args.dataset, args.task)

    if args.memory:
        plot_memory_pressure(out_dir)

    print(f"\nDone. All charts saved to {out_dir}/")


if __name__ == "__main__":
    main()