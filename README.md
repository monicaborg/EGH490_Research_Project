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
from egh490.models import TransformerClassifier, Trainer, Ensemble
from egh490.data import DataModule
from egh490.utils import set_global_seed, load_config, get_logger
```

## Current status

**53 passing tests. Full pipeline running end-to-end on synthetic data.
Ethics approved (HREC #11077). Awaiting labelled dataset from supervisor.**

| Layer | What's built | Tests |
|-------|-------------|-------|
| Utils | Config, seeding, logging, I/O, device auto-detect | 11 |
| Data | DataModule with CSV loading, label encoding, 5-fold CV | 15 |
| Models | TransformerClassifier wrapper for any HuggingFace model | 7 |
| Trainer | Fine-tuning with early stopping, warmup, 5 metrics | 8 |
| Ensemble | Hard + soft voting with confidence tie-breaking | 12 |
| Scripts | `train.py` — end-to-end training with printed results | — |
| **Total** | | **53** |

All four base models verified on synthetic data:

| Model | Checkpoint | Params | Synthetic acc | Time (98 examples) |
|-------|-----------|--------|--------------|-------------------|
| ELECTRA-small | `google/electra-small-discriminator` | 14M | 50% | 33s |
| RoBERTa-base | `roberta-base` | 125M | 75% | 170s |
| XLNet-base | `xlnet-base-cased` | 110M | 75% | 208s |
| ALBERT-base-v2 | `albert-base-v2` | 12M | 90% | 197s |

Synthetic accuracy is not meaningful — listed to confirm each architecture
trains without error. Real data (~3500 responses) should produce 91–97%
per Somers et al. (2021).

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
├── configs/            YAML configs (data, model, training, xai)
├── egh490/             Python package (flat layout)
│   ├── data/           DataModule, schema, CSV loading, k-fold splits
│   │   ├── schema.py       Column names, label mappings, task config
│   │   └── datamodule.py   Load CSV → encode labels → stratified k-fold
│   ├── models/         Transformer wrapper, trainer, ensemble
│   │   ├── base.py         TransformerClassifier (predict / predict_proba)
│   │   ├── trainer.py      Fine-tuning via HuggingFace Trainer API
│   │   └── ensemble.py     Majority vote with confidence tie-breaking
│   ├── xai/            LIME, SHAP, attention (to be built)
│   ├── evaluation/     CV harness, fidelity, stability (to be built)
│   ├── analysis/       Pattern analysis, fairness audit (to be built)
│   └── utils/          Seeding, logging, I/O, config loader, device
│       ├── seeding.py      Deterministic seeds across Python/NumPy/PyTorch
│       ├── config.py       YAML loader with defaults inheritance
│       ├── io.py           YAML/JSON read/write helpers
│       ├── logging.py      Consistent logger factory
│       └── device.py       Auto-detect CPU / CUDA / MPS
├── scripts/
│   └── train.py        Train a single model on one fold (CLI)
├── tests/              53 passing tests (unit + integration)
├── data/
│   ├── raw/            Real corpus — requires ETHICS_APPROVED=1
│   └── synthetic/      100 synthetic responses for pipeline dev
└── outputs/            Checkpoints, predictions, explanations, figures
```

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/monicaborg/EGH490_Research_Project.git
cd EGH490_Research_Project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the test suite (53 tests, ~25 seconds)
pytest -v

# 3. Train ELECTRA on synthetic data (fold 1, validity task)
python scripts/train.py

# 4. Train with a different model
python scripts/train.py --checkpoint roberta-base
python scripts/train.py --checkpoint xlnet-base-cased
python scripts/train.py --checkpoint albert-base-v2

# 5. Train on a specific fold or task
python scripts/train.py --fold 3
python scripts/train.py --task confidence

# 6. Use real data (when labelled dataset arrives)
python scripts/train.py --csv data/raw/labelled_responses.csv

# 7. Reduce memory usage for larger models
python scripts/train.py --checkpoint xlnet-base-cased --batch-size 8
python scripts/train.py --checkpoint roberta-base --batch-size 4
```

## Training script options

```
python scripts/train.py [OPTIONS]

Data:
  --csv PATH              CSV file (default: data/synthetic/synthetic_responses.csv)
  --task {validity,confidence}  Classification task (default: validity)
  --fold N                CV fold to train, 1-indexed (default: 1)
  --n-folds N             Number of folds (default: 5)

Model:
  --checkpoint NAME       HuggingFace model (default: google/electra-small-discriminator)
  --max-length N          Max token length (default: 256)

Training:
  --epochs N              (default: 6)
  --batch-size N          (default: 16)
  --lr FLOAT              Learning rate (default: 2e-5)
  --patience N            Early stopping patience (default: 2)
  --seed N                (default: 20260413)

Output:
  --save-model            Save fine-tuned model to outputs/checkpoints/
  --device {cpu,cuda,mps} Force device (default: auto-detect)
```

## Data schema

The pipeline expects a CSV with these columns:

| Column | Description | Used by |
|--------|------------|---------|
| `uid` | Unique response ID | Metadata |
| `ccuname` | CCU identifier (ccu1–ccu6) | Analysis |
| `time` | Submission timestamp | Metadata |
| `q1mcr` | MCQ answer selected (a/b/c/d) | Analysis |
| `q1txr` | **Free-text response** | **Model input** |
| `q2mcr` | MCQ Q2 answer | Metadata |
| `q3mcr` | MCQ Q3 answer | Metadata |
| `validity` | correct / incorrect | **Label (task 1)** |
| `confidence` | high / low | **Label (task 2)** |

Column names are defined once in `egh490/data/schema.py`. If the real
dataset uses different headers, only that file needs to change.

## Two-phase development strategy

The pipeline is developed against **synthetic data** and switches to the
real corpus with a config change only:

| Phase | Data | Goal | Status |
|-------|------|------|--------|
| 1 | `data/synthetic/` | End-to-end pipeline functional | ✅ Complete |
| 2 | `data/raw/` (SSCI/CCU) | Replicate Somers et al. (2021) | ⏳ Awaiting labels |
| 3 | either | XAI layer + fidelity/stability eval | ⬜ Next |

## Memory and hardware

Tested on MacBook (Apple Silicon, 16 GB RAM):

| Model | Batch 16 | Batch 8 | Batch 4 |
|-------|----------|---------|---------|
| ELECTRA-small | ✅ comfortable | — | — |
| ALBERT-base-v2 | ✅ comfortable | — | — |
| RoBERTa-base | ⚠️ tight | ✅ recommended | — |
| XLNet-base | ⚠️ near limit | ✅ recommended | ✅ safe |

For RoBERTa and XLNet on larger datasets, reduce batch size. To maintain
the effective batch of 16, use gradient accumulation:
```bash
python scripts/train.py --checkpoint xlnet-base-cased --batch-size 4
```
The `TrainingConfig` supports `gradient_accumulation_steps` to simulate
larger effective batches within memory constraints. This does not affect
final model accuracy.

## Reproducibility

- All random seeds fixed in `egh490/utils/seeding.py`.
- Library versions pinned in `pyproject.toml`.
- Hugging Face model revisions pinned per base model in `configs/models/`.
- 5-fold stratified splits deterministic given a seed.
- Single-word responses removed per Somers et al. preprocessing.
- Label encoding: validity (incorrect=0, correct=1),
  confidence (low=0, high=1).

## Ethics

HREC approval #11077 (approved 24/04/2026, expires 
24/04/2031, CI: Dr Sam Cunningham-Nelson). `data/raw/` is `.gitignore`d
and guarded by a runtime check requiring `ETHICS_APPROVED=1` plus an
approval reference on disk.

## What's next

1. **Get labelled dataset from Sam** — critical path blocker
2. **`scripts/train_all_folds.py`** — automate 5-fold CV for one model
3. **`scripts/ensemble.py`** — combine 4 trained models, evaluate
4. **Replicate Somers et al.** — compare metrics to published numbers
5. **XAI layer** — LIME + SHAP on test fold responses
6. **Fidelity + stability** — evaluate explanation quality
7. **Pattern analysis + fairness audit**
8. **Final report + oral defence**

## References

See `docs/references.md` for the full proposal reference list. Key anchors:

- Somers, Cunningham-Nelson & Boles (2021) — baseline ensemble
- Cunningham-Nelson et al. (2018) — CCU framework and pointer categories
- Ribeiro, Singh & Guestrin (2016) — LIME
- Lundberg & Lee (2017) — SHAP
- Gunasekara & Saarela (2025) — fidelity & stability metrics
- Wiegreffe & Pinter (2019) — attention as plausible, not faithful