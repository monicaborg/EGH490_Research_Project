# EGH490_Research_Project

Codebase for the EGH490 project *Explainable Automated Scoring of Conceptual
Reasoning in Signals & Systems* (Monica Borg, n9802045, supervisors Dr Sam
Cunningham-Nelson and Dr Wageeh Boles, QUT Faculty of Engineering, 2026).

This project replicates the transformer ensemble from Somers, Cunningham-Nelson
& Boles (2021) on Signals & Systems MCQ free-text explanations and layers
explainable-AI techniques (LIME, SHAP, attention visualisation) on top, with
fidelity, stability and coverage metrics adapted from DeYoung et al. (2020)
and Gunasekara & Saarela (2025).

The Python package is named `egh490`, so imports stay short and clean:

```python
from egh490.models import TransformerClassifier, Trainer, Ensemble
from egh490.data import DataModule
from egh490.xai import LimeExplainer, ShapExplainer, AttentionExtractor
from egh490.utils import set_global_seed, load_config, get_logger
```

## Current status

**62 passing tests. Full pipeline running end-to-end on the real corpus.
All four models trained across all six CCUs. XAI layer complete, including
unigram and bigram attribution. Ethics approved (HREC #11077).**

| Layer | What's built | Tests |
|-------|-------------|-------|
| Utils | Config, seeding, logging, I/O, device auto-detect | 11 |
| Data | DataModule with CSV loading, label encoding, 5-fold CV, CCU filter | 15 |
| Models | TransformerClassifier wrapper for any HuggingFace model | 7 |
| Trainer | Fine-tuning with early stopping, warmup, optional class weighting | 8 |
| Ensemble | Hard + soft voting with confidence tie-breaking | 12 |
| XAI | LIME, SHAP, attention, fidelity/stability/coverage, visualisation | 9 |
| Evaluation | Cohen's kappa inter-rater agreement | — |
| **Total** | | **62** |

## Results — validity classification, full corpus

Trained on the full 3,386-response corpus (no deduplication, no class
weighting), 5-fold stratified CV, per-CCU. Accuracy is the cross-validation
mean.

| CCU | RoBERTa | ALBERT | XLNet | ELECTRA | Somers et al. (2021) |
|-----|---------|--------|-------|---------|----------------------|
| CCU1 | 95.8% | 95.1% | 87.6% | 80.7% | 97.97% |
| CCU2 | 96.6% | 96.6% | 93.6% | 80.6% | 97.34% |
| CCU3 | 91.6% | 88.5% | 79.5% | 61.6% | 98.66% |
| CCU4 | 97.5% | 94.7% | 91.2% | 84.0% | 96.86% |
| CCU5 | 92.3% | 90.4% | 88.8% | 78.1% | 93.54% |
| CCU6 | 93.0% | 88.5% | 77.4% | 60.2% | 95.95% |

RoBERTa clears 91% on every CCU and exceeds the published benchmark on CCU4.
ELECTRA-small failed to learn a usable decision boundary across all CCUs and
hyperparameter configurations tested (learning rates 2e-5 through 3e-4, batch
sizes 8 and 16), producing F1 scores around 0.45 with accuracy pinned to the
majority-class rate. It is reported for completeness but **excluded from the
final ensemble**, which uses RoBERTa + ALBERT + XLNet.

### Key training findings

- **Deduplication hurts.** Removing exact-duplicate responses cut the training
  set by ~21% and consistently lowered accuracy (e.g. ALBERT CCU2: 90.7%
  deduplicated vs 93.4% full). The full corpus is retained, matching Somers'
  approach and better reflecting real deployment noise.
- **Single-word filtering removed.** An inherited preprocessing step silently
  dropped 723 responses; these are retained, since they were manually annotated
  and form part of the realistic response distribution.
- **Annotation conventions affect class balance.** Responses naming a concept
  without applying it to justify the MCQ selection (e.g. bare `f = 1/T`) are
  classified as invalid — a deliberate design decision reflecting the research
  question's focus on conceptual application rather than formula recall.

## Results — explanation quality

Explanations generated on the three-model ensemble, sampled responses per CCU,
in both unigram (word-level) and bigram (phrase-level) attribution modes.

| Metric | LIME unigram | LIME bigram | SHAP unigram | SHAP bigram |
|--------|-------------|-------------|--------------|-------------|
| Comprehensiveness | 0.338 | 0.348 | 0.581 | 0.667 |
| Sufficiency | 0.721 | 0.638 | 0.970 | 0.976 |
| Stability | 0.992 | 0.987 | 1.000 | 1.000 |
| Coverage | 0.200 | 0.440 | 0.280 | 0.220 |

*(CCU2 ensemble, illustrative — per-CCU results in `outputs/explanations/`.)*

### Key XAI findings

- **LIME and SHAP agree strongly** (Pearson r around 0.69–0.79 across shared
  tokens per CCU), providing mutual validation that the attributions reflect
  genuine model behaviour rather than method-specific artefacts.
- **Bigram attribution roughly doubles LIME's coverage** (0.20 to 0.44) while
  maintaining or improving faithfulness, and produces more interpretable
  features — "higher frequency" as one concept rather than two isolated words.
- **Attention is a weaker explanation signal.** On substantive responses,
  attention weight frequently concentrates on punctuation and structural
  tokens rather than concept words, empirically reproducing the
  "attention is not explanation" finding (Jain & Wallace 2019) on this corpus.
- **Coverage tracks response length and model confidence.** Terse or
  phrase-level answers receive diffuse, sub-threshold attributions — a property
  of the responses rather than a defect in the method.

## Scope at a glance

Two classification tasks over student-written explanations:

1. **Validity** — is the conceptual reasoning in the explanation correct?
2. **Confidence** — does the student express high or low confidence?

Base models: `roberta-base`, `albert-base-v2`, `xlnet-base-cased`
(plus `electra-small-discriminator`, trained but excluded from the ensemble),
combined by soft voting.

Explanation pipelines:

- **LIME** — text perturbation + local linear surrogate; unigram or bigram.
- **SHAP** — Partition Explainer over token groups; unigram or bigram.
- **Attention** — supplementary qualitative visualisation only
  (Wiegreffe & Pinter 2019: plausible, not faithful).

## Repository layout

```
EGH490_Research_Project/
├── configs/            YAML configs (data, model, training, xai)
├── egh490/             Python package (flat layout)
│   ├── data/           DataModule, schema, CSV loading, k-fold splits
│   │   ├── schema.py           Column names, label mappings, task config
│   │   ├── datamodule.py       Load CSV, encode labels, stratified k-fold
│   │   ├── convert_raw_data.py Map marked export to pipeline schema
│   │   └── prepare_datasets.py Build full / junk-filtered dataset variants
│   ├── models/         Transformer wrapper, trainer, ensemble
│   │   ├── base.py             TransformerClassifier (predict/predict_proba)
│   │   ├── trainer.py          Fine-tuning, optional class-weighted loss
│   │   └── ensemble.py         Soft/hard voting with confidence tie-breaking
│   ├── xai/            Explanation generation and evaluation
│   │   ├── lime_explainer.py   LIME, unigram + bigram modes
│   │   ├── shap_explainer.py   SHAP Partition Explainer, unigram + bigram
│   │   ├── attention.py        Attention weight extraction
│   │   ├── evaluation.py       Fidelity, stability, coverage
│   │   └── visualise.py        Token heatmaps, feature importance plots
│   ├── evaluation/     Inter-rater agreement (Cohen's kappa)
│   └── utils/          Seeding, logging, I/O, config loader, device
├── scripts/
│   ├── train_all_models.py        Train all models x folds for one CCU
│   ├── explain.py                 Generate LIME/SHAP/attention + metrics
│   ├── plot_results.py            Training result figures
│   ├── plot_xai.py                XAI figures from saved JSON
│   ├── export_educator_report.py  Flat per-response CSV for educators
│   └── compute_agreement.py       Cohen's kappa between two markers
├── tests/              62 passing tests (unit + integration)
├── data/
│   ├── raw/            Real corpus — gitignored, ethics-restricted
│   └── archive/        Superseded dataset variants
└── outputs/
    ├── metrics/            Per-CCU training results (JSON)
    ├── explanations/       Per-CCU XAI outputs (JSON)
    ├── figures/            Generated figures
    ├── educator_reports/   Flattened per-response CSVs
    └── archive/            Superseded runs
```

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/monicaborg/EGH490_Research_Project.git
cd EGH490_Research_Project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the test suite (62 tests)
pytest -v

# 3. Train all four models on one CCU
python scripts/train_all_models.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --models roberta albert electra xlnet --save-models

# 4. Generate explanations for that CCU (ensemble, excludes ELECTRA by default)
python scripts/explain.py --ensemble \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1 \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --n-samples 150

# 5. Same, with phrase-level (bigram) attribution
python scripts/explain.py --ensemble \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1 \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --n-samples 150 --ngram 2

# 6. Build figures and the educator-facing report
python scripts/plot_xai.py \
  --explanations-dir outputs/explanations/signals_systems_validity_corpus_ccu1_ensemble \
  --out-dir outputs/figures/xai_ccu1

python scripts/export_educator_report.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --explanations-dir outputs/explanations/signals_systems_validity_corpus_ccu1_ensemble \
  --ccu ccu1 --output outputs/educator_reports/ccu1_report.csv
```

## Key script options

```
scripts/train_all_models.py
  --csv PATH                CSV file
  --ccu NAME                Train on one CCU only (ccu1-ccu6)
  --models [NAMES]          Subset of electra/roberta/xlnet/albert
  --skip-models [NAMES]     Exclude specific models
  --epochs N                Max epochs (default: 6)
  --lr FLOAT                Learning rate (default: 2e-5)
  --patience N              Early stopping patience (default: 2, 0 disables)
  --class-weighted          Inverse-frequency class weights in the loss
  --save-models             Persist checkpoints for later XAI use

scripts/explain.py
  --ensemble                Explain the ensemble rather than one model
  --checkpoint PATH         Single-model checkpoint (without --ensemble)
  --checkpoint-dir PATH     Where per-fold checkpoints live
  --dataset-tag TAG         Which CCU's checkpoints to load
  --ccu NAME                Restrict explained responses to one CCU
  --exclude-models [NAMES]  Models to leave out (default: electra)
  --n-samples N             Responses to explain
  --ngram {1,2}             Word-level (1) or phrase-level (2) attribution
  --no-lime / --no-shap / --no-attention / --no-eval
```

Batch sizes are set per model in `MODEL_CONFIGS` (ELECTRA 16, RoBERTa 8,
XLNet 4, ALBERT 16), chosen from memory benchmarking on Apple Silicon.

## Data schema

The pipeline expects a CSV with these columns:

| Column | Description | Used by |
|--------|------------|---------|
| `uid` | Unique response ID | Metadata |
| `ccuname` | CCU identifier (ccu1-ccu6) | Filtering, analysis |
| `time` | Submission timestamp | Metadata |
| `q1mcr` | MCQ answer selected (a/b/c/d) | Analysis |
| `q1txr` | **Free-text response** | **Model input** |
| `q2mcr` | MCQ Q2 answer | Metadata |
| `q3mcr` | MCQ Q3 answer | Metadata |
| `validity` | correct / incorrect | **Label (task 1)** |
| `confidence` | high / low | **Label (task 2)** |

Column names are defined once in `egh490/data/schema.py`.
`scripts/prepare_datasets.py` converts the marked spreadsheet export into
this schema.

## Memory and hardware

Tested on MacBook (Apple Silicon, 16 GB RAM):

| Model | Batch 16 | Batch 8 | Batch 4 |
|-------|----------|---------|---------|
| ELECTRA-small | comfortable | — | — |
| ALBERT-base-v2 | comfortable | — | — |
| RoBERTa-base | tight | recommended | — |
| XLNet-base | near limit | recommended | safe |

Training the full grid (4 models x 6 CCUs x 5 folds) takes roughly 35-40
hours on this hardware; XLNet at batch 4 dominates that total. Explanation
generation is ~45-60 minutes per CCU per attribution mode, driven by LIME's
~1,000 model evaluations per explained response.

Checkpoints are large — `outputs/checkpoints/` is gitignored, and superseded
checkpoint sets should be deleted rather than archived (a full set across all
dataset variants reached 455 GB during development).

## Reproducibility

- All random seeds fixed in `egh490/utils/seeding.py` (default 20260413).
- Library versions pinned in `pyproject.toml`.
- 5-fold stratified splits deterministic given a seed.
- Label encoding: validity (incorrect=0, correct=1),
  confidence (low=0, high=1).
- XAI outputs are serialised to JSON, so figures can be regenerated without
  re-running the (expensive) explanation step.

## Ethics

HREC approval #11077 (LR 2026-11077-29139, approved 24/04/2026, expires
24/04/2031, CI: Dr Sam Cunningham-Nelson). `data/raw/` is gitignored and
guarded by a runtime check requiring `ETHICS_APPROVED=1` plus an approval
reference on disk. The labelled corpus is shared with supervisors directly
rather than through this repository.

## What's next

1. **Inter-rater review** — supervisor verification of the annotated corpus;
   Cohen's kappa reported via `scripts/compute_agreement.py`
2. **Per-CCU XAI analysis** — cross-CCU comparison of attribution patterns
3. **Educator reports** — flattened per-response outputs for all six CCUs
4. **Final report + oral defence**

## References

See `docs/references.md` for the full reference list. Key anchors:

- Somers, Cunningham-Nelson & Boles (2021) — baseline ensemble
- Cunningham-Nelson et al. (2018) — CCU framework and pointer categories
- Ribeiro, Singh & Guestrin (2016) — LIME
- Lundberg & Lee (2017) — SHAP
- DeYoung et al. (2020) — ERASER: comprehensiveness & sufficiency
- Gunasekara & Saarela (2025) — XAI in education, qualitative assessment
- Jain & Wallace (2019); Wiegreffe & Pinter (2019) — attention as
  plausible, not faithful