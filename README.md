# EGH490_Research_Project

Codebase for the EGH490 project *Explainable Automated Scoring of Conceptual
Reasoning in Signals & Systems* (Monica Borg, n9802045, supervisors Dr Sam
Cunningham-Nelson and Dr Wageeh Boles, QUT Faculty of Engineering, 2026).

This project replicates the transformer ensemble from Somers, Cunningham-Nelson
& Boles (2021) on Signals & Systems MCQ free-text explanations and layers
explainable-AI techniques (LIME, SHAP, attention visualisation) on top, with
fidelity and stability metrics adapted from Gunasekara & Saarela (2025).

The Python package is named `egh490`, so imports stay short and clean:

```python
from egh490.models import Ensemble
from egh490.xai import lime_explainer, shap_explainer
from egh490.evaluation import fidelity, stability
```

## Scope at a glance

Two classification tasks over student-written explanations:

1. **Validity** — is the conceptual reasoning in the explanation correct?
2. **Confidence** — does the student express high or low confidence?

Four fine-tuned base models combined by majority vote:
`electra-small-discriminator`, `roberta-base`, `xlnet-base-cased`,
`albert-base-v2`.

Explanation pipelines:

- **LIME** — text perturbation + local linear surrogate.
- **SHAP** — Partition Explainer over token groups.
- **Attention** — supplementary qualitative visualisation only
  (Wiegreffe & Pinter 2019: plausible, not faithful).

## Repository layout

```
EGH490_Research_Project/
├── configs/            Hydra-style YAML configs (data, model, training, xai)
├── egh490/             Python package (flat layout)
│   ├── data/           Loading, synthetic generation, preprocessing, splits
│   ├── models/         Base model wrappers, ensemble, training loop
│   ├── xai/            LIME, SHAP, attention, fidelity & stability metrics
│   ├── evaluation/     CV harness, classification metrics, benchmarking
│   ├── analysis/       Linguistic / conceptual pattern analysis, fairness
│   └── utils/          Seeding, logging, I/O, device selection
├── scripts/            CLI entry points (train, explain, evaluate, report)
├── notebooks/          EDA, exploratory XAI, figure generation
├── tests/              Smoke tests on synthetic data
├── data/               Local data root (gitignored except README)
│   ├── raw/            Real corpus (SSCI/CCU) — only after ethics approval
│   ├── synthetic/      Generated placeholder responses for pipeline dev
│   ├── processed/      Tokenised/cleaned artefacts
│   └── splits/         Stratified 5-fold CV indices
└── outputs/            Checkpoints, predictions, explanations, figures, logs
```

## Two-phase development strategy

Because ethics approval for the real corpus is pending (Section 6.2 of the
proposal), the pipeline is developed and validated first against **synthetic
data** that mirrors the schema of the real corpus:

| Phase | Data | Goal | Proposal milestone |
|-------|------|------|-------------------|
| 1 | `data/synthetic/` | End-to-end pipeline functional (item 5) | W7–W12 |
| 2 | `data/raw/` (SSCI/CCU) | Replicate Somers et al. (2021) | W12–W14 |
| 3 | either | XAI layer + fidelity/stability eval | W14–W28 |

Switching phases is a **config change only** — the training and XAI code paths
are identical. This is enforced by the `DataModule` abstraction in
`egh490/data/datamodule.py`.

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> EGH490_Research_Project
cd EGH490_Research_Project

# 2. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Generate synthetic data so the pipeline can be exercised end-to-end
python scripts/generate_synthetic.py --out data/synthetic/ --n 600

# 4. Smoke-test training (tiny run on CPU)
python scripts/train.py --config configs/smoke.yaml

# 5. Full training on one base model (task=validity)
python scripts/train.py --config configs/train_electra_validity.yaml

# 6. Build the ensemble from four trained checkpoints
python scripts/ensemble.py --config configs/ensemble_validity.yaml

# 7. Generate LIME/SHAP/attention explanations
python scripts/explain.py --config configs/xai_lime.yaml
python scripts/explain.py --config configs/xai_shap.yaml

# 8. Evaluate fidelity & stability (50 perturbation iters per instance)
python scripts/evaluate_xai.py --config configs/xai_eval.yaml

# 9. Run linguistic/conceptual pattern analysis
python scripts/analyse_patterns.py --config configs/analysis.yaml
```

## Reproducibility

- All random seeds fixed in `egh490/utils/seeding.py`.
- Library versions pinned in `pyproject.toml`.
- Hugging Face model revisions pinned per base model in `configs/models/`.
- Every run writes a manifest (`outputs/logs/<run_id>/manifest.json`)
  capturing config hash, git SHA, package versions, and hardware.
- 5-fold stratified splits deterministic given a seed; split indices saved
  to `data/splits/` so every model sees the same folds.

## Ethics

No real student data is touched in this repository until the HREC application
(Chief Investigator: Dr Sam Cunningham-Nelson, April 2026) is approved.
`data/raw/` is `.gitignore`d and guarded by a runtime check in
`egh490/data/real_corpus.py` that refuses to load without a
`ETHICS_APPROVED=1` environment flag plus an approval reference on disk.

## References

See `docs/references.md` for the full proposal reference list. Key anchors:

- Somers, Cunningham-Nelson & Boles (2021) — baseline ensemble
- Cunningham-Nelson et al. (2018) — CCU framework and pointer categories
- Ribeiro, Singh & Guestrin (2016) — LIME
- Lundberg & Lee (2017) — SHAP
- Gunasekara & Saarela (2025) — fidelity & stability metrics
- Wiegreffe & Pinter (2019) — attention as plausible, not faithful