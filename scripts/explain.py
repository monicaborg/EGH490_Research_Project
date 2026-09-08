"""Run the XAI layer over a trained model and save explanations + metrics.

Loads a trained classifier (or builds an ensemble from four trained models),
generates LIME and SHAP explanations for a sample of responses, optionally
extracts attention weights, evaluates explanation quality (fidelity,
stability, coverage), and writes everything to JSON for later visualisation.

Usage
-----
Explain with a single trained model::

    python scripts/explain.py \\
        --checkpoint outputs/checkpoints/validity_roberta_fold1_synthetic_responses_2000/final \\
        --csv data/synthetic/synthetic_responses_2000.csv \\
        --n-samples 20

Explain the full four-model ensemble::

    python scripts/explain.py \\
        --ensemble \\
        --checkpoint-dir outputs/checkpoints \\
        --dataset-tag synthetic_responses_2000 \\
        --csv data/synthetic/synthetic_responses_2000.csv

Skip SHAP (faster, LIME only)::

    python scripts/explain.py --checkpoint <path> --csv <path> --no-shap

Output
------
    outputs/explanations/<tag>/lime_explanations.json
    outputs/explanations/<tag>/shap_explanations.json
    outputs/explanations/<tag>/attention.json
    outputs/explanations/<tag>/xai_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the XAI layer over a trained model.")
    # Model source
    p.add_argument("--checkpoint", default=None,
                   help="Path to a single trained model directory")
    p.add_argument("--ensemble", action="store_true",
                   help="Build the four-model ensemble instead of a single model")
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints",
                   help="Directory containing per-model checkpoints (for --ensemble)")
    p.add_argument("--dataset-tag", default="synthetic_responses_2000",
                   help="Dataset tag used to locate ensemble checkpoints")
    p.add_argument("--fold", type=int, default=1,
                   help="Which fold's trained models to load for the ensemble")
    # Data
    p.add_argument("--csv", required=True, help="CSV of responses to explain")
    p.add_argument("--task", default="validity", choices=["validity", "confidence"])
    p.add_argument("--n-samples", type=int, default=20,
                   help="How many responses to explain (sampled from the CSV)")
    p.add_argument("--ccu", default=None,
                   help="Restrict explained responses to a single CCU (e.g. ccu3). "
                        "Without this, responses are sampled from the whole dataset, "
                        "which will not match the CCU-specific ensemble being loaded.")
    p.add_argument("--seed", type=int, default=20260413)
    # Which techniques to run
    p.add_argument("--no-lime", action="store_true", help="Skip LIME")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP")
    p.add_argument("--no-attention", action="store_true", help="Skip attention extraction")
    p.add_argument("--no-eval", action="store_true", help="Skip fidelity/stability/coverage")
    # XAI parameters
    p.add_argument("--lime-samples", type=int, default=1000,
                   help="LIME perturbation samples per explanation")
    p.add_argument("--shap-max-evals", type=int, default=500,
                   help="SHAP max evaluations per explanation")
    p.add_argument("--fidelity-k", type=int, default=5,
                   help="Top-k features used in fidelity evaluation")
    p.add_argument("--stability-repeats", type=int, default=5,
                   help="Repeats per response for stability (LIME only)")
    # Output
    p.add_argument("--output-dir", default="outputs/explanations")
    p.add_argument("--exclude-models", nargs="*", default=["electra"],
                   choices=["electra", "roberta", "xlnet", "albert"],
                   help="Models to exclude from the ensemble (default: electra, "
                        "which underperforms). Pass with no values to include all.")
    p.add_argument("--ngram", type=int, default=1, choices=[1, 2],
                   help="Attribution granularity: 1=unigram (words), 2=bigram "
                        "(overlapping word pairs). Run both separately to compare.")
    return p.parse_args(argv)


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_single(checkpoint: str, num_labels: int, logger):
    from egh490.models import TransformerClassifier
    logger.info("Loading single model from %s", checkpoint)
    return TransformerClassifier.load(checkpoint, num_labels=num_labels)


def load_ensemble(checkpoint_dir: str, dataset_tag: str, task: str, fold: int,
                  num_labels: int, logger, exclude_models=None):
    """Build the ensemble from per-fold checkpoints.

    Expects the directory layout produced by train_all_models.py with
    --save-models, i.e.:
        <checkpoint_dir>/<task>_<model>_fold<fold>_<tag>/final/

    exclude_models: optional list of model keys to leave out of the ensemble
    (e.g. ["electra"] to exclude the underperforming ELECTRA model).
    """
    from egh490.models import Ensemble, TransformerClassifier

    exclude = set(exclude_models or [])
    model_keys = [k for k in ["electra", "roberta", "xlnet", "albert"] if k not in exclude]
    if exclude:
        logger.info("Excluding from ensemble: %s", sorted(exclude))
    classifiers = []
    for key in model_keys:
        run_name = f"{task}_{key}_fold{fold}_{dataset_tag}"
        path = Path(checkpoint_dir) / run_name / "final"
        if not path.exists():
            logger.warning("Missing checkpoint for %s at %s — skipping", key, path)
            continue
        logger.info("Loading %s from %s", key, path)
        classifiers.append(TransformerClassifier.load(str(path), num_labels=num_labels))

    if len(classifiers) < 2:
        raise RuntimeError(
            f"Ensemble needs at least 2 models; found {len(classifiers)}. "
            "Did you train with --save-models?"
        )
    return Ensemble(classifiers, strategy="soft")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv=None):
    args = parse_args(argv)

    from egh490.data import DataModule
    from egh490.data.schema import get_task_config
    from egh490.utils import get_logger, set_global_seed

    set_global_seed(args.seed)
    logger = get_logger("explain")

    task_cfg = get_task_config(args.task)
    num_labels = task_cfg["num_labels"]
    class_names = [task_cfg["label_names"][i] for i in range(num_labels)]

    # ── Load responses ───────────────────────────────────────────────
    dm = DataModule(csv_path=args.csv, task=args.task, seed=args.seed)
    all_indices = list(range(len(dm.texts)))

    # Restrict to a single CCU when requested. Without this the sample is
    # drawn from the whole corpus, which would not correspond to the
    # CCU-specific ensemble being loaded from --dataset-tag.
    if args.ccu:
        df = dm.get_dataframe()
        if "ccuname" not in df.columns:
            raise ValueError("--ccu given but the CSV has no 'ccuname' column")
        mask = df["ccuname"].astype(str).str.strip().str.lower() == args.ccu.strip().lower()
        all_indices = [i for i, keep in enumerate(mask.tolist()) if keep]
        if not all_indices:
            raise ValueError(f"No responses found for --ccu {args.ccu!r}")
        logger.info("CCU filter: %s — %d responses available", args.ccu, len(all_indices))

    all_texts, all_labels = dm.get_texts_and_labels(all_indices)

    # Sample a subset for explanation (XAI is slow; explaining the full
    # corpus is rarely necessary for analysis).
    import numpy as np
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(all_texts))
    idx = rng.choice(len(all_texts), size=n, replace=False)
    texts = [all_texts[i] for i in idx]
    logger.info("Explaining %d responses (task=%s)", len(texts), args.task)

    # ── Load model ───────────────────────────────────────────────────
    if args.ensemble:
        model = load_ensemble(
            args.checkpoint_dir, args.dataset_tag, args.task, args.fold,
            num_labels, logger, exclude_models=args.exclude_models,
        )
        tag = f"{args.dataset_tag}_ensemble"
    else:
        if not args.checkpoint:
            raise SystemExit("Provide --checkpoint <path> or use --ensemble")
        model = load_single(args.checkpoint, num_labels, logger)
        tag = Path(args.checkpoint).parent.name or "single_model"

    out_dir = Path(args.output_dir) / tag
    if args.ngram >= 2:
        out_dir = out_dir.parent / f"{out_dir.name}_bigram"
    out_dir.mkdir(parents=True, exist_ok=True)

    lime_explanations = []
    shap_explanations = []

    # ── LIME ─────────────────────────────────────────────────────────
    if not args.no_lime:
        from egh490.xai import LimeExplainer
        logger.info("Running LIME (%d samples per explanation)...", args.lime_samples)
        lime_exp = LimeExplainer(
            model, class_names=class_names,
            num_samples=args.lime_samples, random_state=args.seed, ngram=args.ngram,
        )
        lime_explanations = lime_exp.explain_batch(texts)
        with open(out_dir / "lime_explanations.json", "w") as f:
            json.dump([e.as_dict() for e in lime_explanations], f, indent=2)
        logger.info("Saved LIME → %s", out_dir / "lime_explanations.json")

    # ── SHAP ─────────────────────────────────────────────────────────
    if not args.no_shap:
        from egh490.xai import ShapExplainer
        logger.info("Running SHAP (max_evals=%d)...", args.shap_max_evals)
        shap_exp = ShapExplainer(
            model, class_names=class_names, max_evals=args.shap_max_evals, ngram=args.ngram,
        )
        shap_explanations = shap_exp.explain_batch(texts)
        with open(out_dir / "shap_explanations.json", "w") as f:
            json.dump([e.as_dict() for e in shap_explanations], f, indent=2)
        logger.info("Saved SHAP → %s", out_dir / "shap_explanations.json")

    # ── Attention (single model only) ────────────────────────────────
    if not args.no_attention and not args.ensemble:
        from egh490.xai import AttentionExtractor
        logger.info("Extracting attention weights...")
        attn = AttentionExtractor(model)
        attn_results = attn.extract_batch(texts)
        with open(out_dir / "attention.json", "w") as f:
            json.dump([a.as_dict() for a in attn_results], f, indent=2)
        logger.info("Saved attention → %s", out_dir / "attention.json")
    elif not args.no_attention and args.ensemble:
        logger.info("Skipping attention — not defined for an ensemble")

    # ── Evaluation ───────────────────────────────────────────────────
    if not args.no_eval:
        from egh490.xai import compute_coverage, compute_fidelity, compute_stability

        evaluation = {}

        # Compute faithfulness (comprehensiveness/sufficiency) and coverage
        # separately for each explanation method that was run, so LIME and
        # SHAP can be reported side by side rather than only the primary one.
        method_explanations = {}
        if lime_explanations:
            method_explanations["lime"] = lime_explanations
        if shap_explanations:
            method_explanations["shap"] = shap_explanations

        for method_name, expls in method_explanations.items():
            logger.info("Computing fidelity for %s (k=%d)...", method_name.upper(), args.fidelity_k)
            fid = compute_fidelity(model, expls, k=args.fidelity_k)
            logger.info("Computing coverage for %s...", method_name.upper())
            cov = compute_coverage(expls)
            evaluation[method_name] = {
                "fidelity": fid.as_dict(),
                "coverage": cov.as_dict(),
            }

        # Stability re-runs the explainer, so compute it for whichever
        # perturbation methods are enabled. LIME's random sampling makes this
        # the most important case; SHAP is near-deterministic but included for
        # completeness and direct comparability.
        stab_texts = texts[: min(5, len(texts))]
        if not args.no_lime and lime_explanations:
            from egh490.xai import LimeExplainer
            logger.info("Computing stability for LIME (%d repeats)...", args.stability_repeats)
            lime_exp = LimeExplainer(
                model, class_names=class_names,
                num_samples=args.lime_samples, random_state=args.seed,
                ngram=args.ngram,
            )
            stab = compute_stability(
                lambda t: lime_exp.explain(t),
                stab_texts,
                n_repeats=args.stability_repeats,
            )
            evaluation.setdefault("lime", {})["stability"] = stab.as_dict()

        if not args.no_shap and shap_explanations:
            from egh490.xai import ShapExplainer
            logger.info("Computing stability for SHAP (%d repeats)...", args.stability_repeats)
            shap_exp = ShapExplainer(
                model, class_names=class_names, max_evals=args.shap_max_evals,
                ngram=args.ngram,
            )
            stab_shap = compute_stability(
                lambda t: shap_exp.explain(t),
                stab_texts,
                n_repeats=args.stability_repeats,
            )
            evaluation.setdefault("shap", {})["stability"] = stab_shap.as_dict()

        with open(out_dir / "xai_evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)
        logger.info("Saved evaluation → %s", out_dir / "xai_evaluation.json")

        # Console summary — one block per method
        logger.info("")
        logger.info("=" * 60)
        logger.info("XAI EVALUATION SUMMARY — %s", tag)
        logger.info("=" * 60)
        for method_name in ("lime", "shap"):
            m = evaluation.get(method_name)
            if not m:
                continue
            logger.info("--- %s ---", method_name.upper())
            if "fidelity" in m:
                logger.info("  Comprehensiveness: %.3f (higher = important words mattered)",
                            m["fidelity"]["comprehensiveness"])
                logger.info("  Sufficiency:       %.3f (higher = top words are enough)",
                            m["fidelity"]["sufficiency"])
            if "stability" in m:
                logger.info("  Stability (rank r): %.3f (higher = consistent explanations)",
                            m["stability"]["mean_rank_correlation"])
            if "coverage" in m:
                logger.info("  Coverage:          %.3f (%d / %d responses explained)",
                            m["coverage"]["coverage"],
                            m["coverage"]["n_covered"],
                            m["coverage"]["n_total"])
        logger.info("=" * 60)

    logger.info("Done. All outputs in %s/", out_dir)


if __name__ == "__main__":
    main()