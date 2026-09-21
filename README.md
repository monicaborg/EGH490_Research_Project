# EGH490_Research_Project

Codebase for the EGH490 capstone project, submitted as *Right Answer, Wrong
Reason: Explainable AI for Conceptual Understanding Assessment in Engineering
Education* (working title for eventual journal submission; project title
throughout the QUT capstone is *Explainable Automated Scoring of Conceptual
Reasoning in Signals & Systems*). Monica Borg, n9802045, supervisors Dr Sam
Cunningham-Nelson and Dr Wageeh Boles, QUT Faculty of Engineering, 2026.

This project replicates the transformer ensemble from Somers, Cunningham-Nelson
& Boles (2021) on Signals & Systems MCQ free-text explanations and layers
explainable-AI techniques (LIME, SHAP, attention visualisation) on top, with
fidelity, stability and coverage metrics adapted from DeYoung et al. (2020)
and Gunasekara & Saarela (2025). A secondary classification task (expressed
confidence) is used as a comparative case to test whether explanation quality
is a property of the classifier or of the underlying task.

The Python package is named `egh490`, so imports stay short and clean:

```python
from egh490.models import TransformerClassifier, Trainer, Ensemble
from egh490.data import DataModule
from egh490.xai import LimeExplainer, ShapExplainer, AttentionExtractor
from egh490.utils import set_global_seed, load_config, get_logger
```

## Current status

**62 passing tests. Both classification tasks (validity, confidence) trained
and explained end-to-end across all six CCUs, both attribution granularities.
Ensemble evaluated directly (not inferred from individual members). MCQ-plus-
text combined-input pilot complete. Ethics approved (HREC #11077). Inter-rater
review in progress (3 of 6 CCUs returned).**

| Layer | What's built | Tests |
|-------|-------------|-------|
| Utils | Config, seeding, logging, I/O, device auto-detect | 11 |
| Data | DataModule with CSV loading, label encoding, 5-fold CV, CCU filter, optional MCQ-augmented input | 15 |
| Models | TransformerClassifier wrapper for any HuggingFace model | 7 |
| Trainer | Fine-tuning with early stopping, warmup, optional class weighting | 8 |
| Ensemble | Hard + soft voting with confidence tie-breaking | 12 |
| XAI | LIME, SHAP, attention, fidelity/stability/coverage, visualisation | 9 |
| Evaluation | Cohen's kappa inter-rater agreement; inference-only ensemble scoring | — |
| **Total** | | **62** |

## Results — classifier performance, full corpus

Trained on the full 3,386-response corpus (no deduplication, no class
weighting — both tested and found unnecessary/detrimental once the full
corpus was adopted), 5-fold stratified CV, per-CCU.

### Validity — ensemble vs published baseline

| CCU | Ensemble accuracy | Ensemble AUC | Somers accuracy | Somers AUC |
|-----|-------------------:|--------------:|-----------------:|------------:|
| CCU1 | 95.9% | 0.966 | 97.97% | 0.8641 |
| CCU2 | 97.0% | 0.989 | 97.34% | 0.9747 |
| CCU3 | 89.2% | 0.950 | 98.66% | 0.9814 |
| CCU4 | 96.4% | 0.983 | 96.86% | 0.9661 |
| CCU5 | 92.5% | 0.960 | 93.54% | 0.9376 |
| CCU6 | 89.8% | 0.964 | 95.95% | 0.9577 |
| **Mean** | **93.47%** | **0.9687** | **96.72%** | **0.9469** |

The ensemble trails the published baseline on accuracy but **exceeds it on
AUC in 5 of 6 CCUs**, most sharply on CCU1 (0.966 vs 0.8641). AUC is
threshold-independent and less sensitive to class imbalance than accuracy;
Somers' highest-accuracy/lowest-AUC combination on CCU1 is consistent with an
imbalanced dataset rewarding majority-class prediction. Ensembling is not
uniformly beneficial here — it beats its strongest individual member on only
2 of 6 CCUs, tracking XLNet's per-CCU weakness (see per-model table below).

### Validity — per-model accuracy

| CCU | RoBERTa | ALBERT | XLNet | ELECTRA |
|-----|--------:|-------:|------:|--------:|
| CCU1 | 95.8% | 96.0% | 87.6% | 80.7% |
| CCU2 | 96.6% | 96.6% | 93.6% | 80.6% |
| CCU3 | 91.6% | 88.9% | 79.5% | 61.6% |
| CCU4 | 97.4% | 94.7% | 91.2% | 84.0% |
| CCU5 | 92.3% | 90.4% | 88.8% | 78.1% |
| CCU6 | 93.0% | 88.4% | 77.4% | 60.2% |

ELECTRA-small failed to learn a usable decision boundary across every CCU and
hyperparameter configuration tested (learning rates 2e-5–3e-4, batch sizes
8 and 16), producing F1 around 0.45 with accuracy pinned to the majority-class
rate. Reported for completeness but **excluded from the final ensemble**,
which uses RoBERTa + ALBERT + XLNet.

### Confidence (secondary task) — ensemble

| CCU | Accuracy | AUC | Macro F1 |
|-----|---------:|----:|---------:|
| CCU1 | 98.9% | 0.996 | 0.967 |
| CCU2 | 97.8% | 0.996 | 0.977 |
| CCU3 | 97.3% | 0.992 | 0.960 |
| CCU4 | 98.1% | 0.993 | 0.979 |
| CCU5 | 97.3% | 0.985 | 0.971 |
| CCU6 | 98.1% | 0.996 | 0.972 |
| **Mean** | **97.94%** | **0.9931** | **0.971** |

Confidence is both more accurate and far more *consistent* across CCUs than
validity (1.6-point accuracy spread vs 7.8 points) — consistent with
confidence being carried by surface linguistic features (assertiveness,
brevity, hedging) that behave uniformly regardless of topic, while validity
depends on the difficulty of the underlying concept.

### Key training findings

- **Deduplication hurts.** Removing exact-duplicate responses cut the
  training set by ~19% and consistently lowered accuracy (e.g. ALBERT CCU2:
  90.7% deduplicated vs 93.4% full). The full corpus is retained.
- **Single-word filtering removed.** An inherited preprocessing step silently
  dropped 723 responses; these are retained as part of the realistic response
  distribution.
- **Annotation conventions affect class balance and training stability, but
  not final accuracy.** A matched case study (CCU6, one annotation variable
  held constant) found the stricter conceptual-application convention carries
  no measurable accuracy cost relative to a more lenient alternative once
  class weighting is applied consistently.
- **MCQ-plus-text combined input — tested, inconclusive.** A two-CCU pilot
  (RoBERTa, CCU3 and CCU6) found no meaningful effect on CCU3 and a clear
  degradation on CCU6, most plausibly a fold-dependent shortcut on the
  corpus's smallest, most volatile CCU rather than a genuine signal. Full-
  scale, multi-model trial remains open.

## Results — explanation quality

Explanations generated on the three-model ensemble, 150 sampled responses per
CCU (fixed seed — identical sample across task/granularity, enabling direct
same-response comparison), both unigram and bigram attribution, both tasks.

| Metric | LIME (validity, uni/bi) | SHAP (validity, uni/bi) | LIME (confidence, uni/bi) | SHAP (confidence, uni/bi) |
|--------|--------------------------|--------------------------|-----------------------------|-----------------------------|
| Comprehensiveness | 0.435 / 0.400 | 0.607 / 0.700 | 0.365 / 0.353 | 0.529 / 0.639 |
| Sufficiency | 0.750 / 0.656 | 0.926 / 0.960 | 0.804 / 0.707 | 0.969 / 0.979 |
| Stability | 0.905 / 0.809 | 1.000 / 1.000 | 0.913 / 0.761 | 1.000 / 1.000 |
| Coverage | 0.531 / 0.520 | 0.590 / 0.559 | 0.634 / 0.632 | 0.756 / 0.658 |

### Key XAI findings

- **LIME and SHAP agree strongly** (Pearson r = 0.72–0.83 per CCU, mean 0.770,
  validity unigram) — mutual validation from two methods with entirely
  different theoretical foundations. Agreement is lower on confidence
  (mean 0.639) despite confidence being the higher-coverage task, and the gap
  replicates under bigram attribution — consistent with confidence resting on
  several redundant, individually-sufficient cues rather than one or two
  dominant tokens.
- **SHAP is more faithful than LIME on every metric, every CCU, both tasks,
  both granularities**, and achieves perfect stability (1.000) throughout —
  the expected result of deterministic Shapley computation vs LIME's
  stochastic sampling.
- **Bigram attribution helps SHAP, hurts LIME.** Comprehensiveness rises for
  SHAP (+0.093 validity, +0.110 confidence) and falls for LIME; LIME's
  stability also degrades under bigram (validity 0.905→0.809; confidence
  0.913→0.761). Coverage effects are CCU-dependent: bigrams recover coverage
  where unigram was weakest (e.g. validity CCU2/CCU4) and can cost coverage
  where unigram was already strong (validity CCU1).
- **Confidence vs validity is a stable, replicated task contrast.** Confidence
  explanations show higher coverage and sufficiency but lower comprehensive-
  ness than validity at *both* granularities — evidence that explanation
  quality tracks the task's underlying linguistic signal, not classifier
  accuracy. A more accurate classifier (confidence) does not automatically
  produce a more comprehensively-explainable one.
- **Attention is a weaker, architecture-dependent explanation signal.**
  ALBERT's top attention weight falls on the sequence-final token in 86.8% of
  validity responses (up to 100% on some CCUs) regardless of content —
  reproducing "attention is not explanation" (Jain & Wallace 2019) on real
  student writing. Severity varies sharply by architecture (RoBERTa 40.2%,
  XLNet 17.2%), a dimension the original debate did not consider. Attention
  has no meaningful bigram condition (it reads each transformer's native
  subword tokenizer, independent of the LIME/SHAP granularity setting), so a
  single set of figures is reported per task.

## Scope at a glance

Two classification tasks over student-written explanations:

1. **Validity** (primary) — is the conceptual reasoning in the explanation
   correct?
2. **Confidence** (secondary, comparative case) — does the student express
   high or low confidence?

Base models: `roberta-base`, `albert-base-v2`, `xlnet-base-cased`
(plus `electra-small-discriminator`, trained but excluded from the ensemble),
combined by soft voting.

Explanation pipelines:

- **LIME** — text perturbation + local linear surrogate; unigram or bigram.
- **SHAP** — Partition Explainer over token groups; unigram or bigram.
- **Attention** — supplementary qualitative visualisation only
  (Wiegreffe & Pinter 2019: plausible, not faithful).

Analytical framework (Section 3.8 in the report): a four-category cross-
tabulation of reasoning validity against MCQ correctness — genuine
understanding, slip, guess, consistent misconception — used to audit whether
classifier errors concentrate on the educationally significant "guess" case
(invalid reasoning, correct MCQ; 30.5% of the corpus).

## Repository layout

```
EGH490_Research_Project/
├── configs/            YAML configs (data, model, training, xai)
├── egh490/             Python package (flat layout)
│   ├── data/           DataModule, schema, CSV loading, k-fold splits
│   │   ├── schema.py           Column names, label mappings, task config
│   │   ├── datamodule.py       Load CSV → encode labels → stratified k-fold;
│   │   │                       optional MCQ-augmented input (include_mcq)
│   │   ├── convert_raw_data.py Map marked export → pipeline schema
│   │   └── prepare_datasets.py Build full / junk-filtered dataset variants
│   ├── models/         Transformer wrapper, trainer, ensemble
│   │   ├── base.py             TransformerClassifier (predict / predict_proba)
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
│   ├── train_all_models.py        Train all models x folds x task for one CCU
│   ├── explain.py                 Generate LIME/SHAP/attention + metrics
│   ├── evaluate_ensemble.py       Inference-only ensemble scoring (no retrain)
│   ├── mcq_mismatch_analysis.py   Validity x MCQ cross-tabulation and
│   │                               classifier error breakdown by category
│   ├── plot_results.py            Training result figures
│   ├── plot_xai.py                XAI figures from saved JSON
│   ├── export_educator_report.py  Flat per-response CSV for educators
│   └── compute_agreement.py       Cohen's kappa between two markers
├── tests/              62 passing tests (unit + integration)
├── data/
│   ├── raw/            Real corpus — gitignored, ethics-restricted
│   └── archive/        Superseded dataset variants
└── outputs/
    ├── metrics/            Per-CCU, per-task training results (JSON)
    ├── explanations/       Per-CCU, per-task XAI outputs (JSON)
    ├── figures/            Generated figures
    ├── educator_reports/   Flattened per-response CSVs
    ├── agreement/          Inter-rater agreement outputs
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

# 3. Train all four models on one CCU (validity, default task)
python scripts/train_all_models.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --models roberta albert electra xlnet --save-models

# 3a. Same, for the confidence task
python scripts/train_all_models.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --task confidence --models roberta albert xlnet --save-models

# 4. Score the ensemble directly (inference only, no retraining)
python scripts/evaluate_ensemble.py \
  --csv data/raw/signals_systems_validity_corpus.csv --ccu ccu1 \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1

# 5. Generate explanations for that CCU (ensemble, excludes ELECTRA by default)
python scripts/explain.py --ensemble \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1 \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --n-samples 150

# 6. Same, with phrase-level (bigram) attribution
python scripts/explain.py --ensemble \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1 \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --n-samples 150 --ngram 2

# 7. Build figures and the educator-facing report
python scripts/plot_xai.py \
  --explanations-dir outputs/explanations/signals_systems_validity_corpus_ccu1_ensemble \
  --out-dir outputs/figures/xai_ccu1

python scripts/export_educator_report.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --explanations-dir outputs/explanations/signals_systems_validity_corpus_ccu1_ensemble \
  --ccu ccu1 --output outputs/educator_reports/ccu1_report.csv

# 8. MCQ-vs-reasoning mismatch analysis (Section 3.8 framework)
python scripts/mcq_mismatch_analysis.py \
  --csv data/raw/signals_systems_validity_corpus.csv --ccu ccu1 \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1
```

## Key script options

```
scripts/train_all_models.py
  --csv PATH                CSV file
  --ccu NAME                Train on one CCU only (ccu1–ccu6)
  --task {validity,confidence}  Which annotated task (default: validity)
  --models [NAMES]          Subset of electra/roberta/xlnet/albert
  --skip-models [NAMES]     Exclude specific models
  --epochs N                Max epochs (default: 6)
  --lr FLOAT                Learning rate (default: 2e-5)
  --patience N              Early stopping patience (default: 2, 0 disables)
  --class-weighted          Inverse-frequency class weights in the loss
  --include-mcq             Pilot: prepend student's MCQ selection to the
                             text seen by the model (Section 3.8 extension)
  --save-models              Persist checkpoints for later XAI use

scripts/evaluate_ensemble.py
  --csv PATH / --ccu NAME / --task {validity,confidence}
  --checkpoint-dir PATH      Where per-fold checkpoints live
  --dataset-tag TAG          Which CCU's checkpoints to load
  --models [NAMES]           Ensemble members (default excludes electra)
  (inference-only — reconstructs training folds, soft-votes saved
  checkpoints on each fold's held-out set; no retraining)

scripts/explain.py
  --ensemble                Explain the ensemble rather than one model
  --checkpoint PATH         Single-model checkpoint (without --ensemble)
  --checkpoint-dir PATH     Where per-fold checkpoints live
  --dataset-tag TAG         Which CCU's checkpoints to load
  --ccu NAME                Restrict explained responses to one CCU
  --task {validity,confidence}  Which task's explanations to generate
  --exclude-models [NAMES]  Models to leave out (default: electra)
  --n-samples N             Responses to explain
  --ngram {1,2}             Word-level (1) or phrase-level (2) attribution
  --no-lime / --no-shap / --no-attention / --no-eval

scripts/mcq_mismatch_analysis.py
  --csv PATH / --ccu NAME
  --checkpoint-dir PATH / --dataset-tag TAG
  --models [NAMES]           Ensemble members (default excludes electra)
  (produces ground-truth and classifier-error cross-tabulations against
  the four-category ground-truth framework — Section 3.8)
```

Batch sizes are set per model in `MODEL_CONFIGS` (ELECTRA 16, RoBERTa 8,
XLNet 4, ALBERT 16), chosen from memory benchmarking on Apple Silicon.
Output paths and checkpoint/explanation directory names are task-namespaced
(e.g. `..._ensemble` for validity, `..._ensemble_confidence` for confidence)
so a confidence run never overwrites validity results, or vice versa.

## Data schema

The pipeline expects a CSV with these columns:

| Column | Description | Used by |
|--------|------------|---------|
| `uid` | Unique response ID | Metadata |
| `ccuname` | CCU identifier (ccu1–ccu6) | Filtering, analysis |
| `time` | Submission timestamp | Metadata |
| `q1mcr` | MCQ answer selected (a/b/c/d) | Analysis, `--include-mcq` |
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

Training the full grid (4 models × 6 CCUs × 5 folds × 2 tasks) takes roughly
70–80 hours on this hardware; XLNet at batch 4 dominates that total.
Explanation generation is ~2–2.5 hours per CCU per task per attribution mode
(LIME dominates at ~1,000 model evaluations per explained response).

Checkpoints are large — `outputs/checkpoints/` is gitignored, and superseded
checkpoint sets should be deleted rather than archived (a full set across all
dataset variants reached 455 GB during development).

## Reproducibility

- All random seeds fixed in `egh490/utils/seeding.py` (default 20260413).
- Fixed seed means XAI response sampling is *identical* across task and
  granularity for a given CCU — the same 150 responses are explained whether
  running validity or confidence, unigram or bigram, enabling direct
  same-response comparison (used throughout Section 4.6's figures).
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

1. **Inter-rater review completion** — 3 of 6 CCUs returned (pooled κ = 0.722,
   "substantial" agreement, disagreement traced to a single identified
   annotation convention); CCU4–6 outstanding.
2. **Full-scale MCQ-plus-text trial** — the two-CCU pilot (Appendix A17) was
   inconclusive; a six-CCU, multi-model trial with a non-concatenative
   feature encoding is the natural follow-up.
3. **Threshold and ensemble-composition optimisation** — given the ensemble's
   AUC advantage over the published baseline, tuning the decision threshold
   or testing a two-model (RoBERTa + ALBERT) ensemble are cheap, no-retrain
   experiments likely to recover much of the residual accuracy gap.
4. **Deployment study** — whether exposing these explanations to educators or
   students measurably changes teaching decisions or understanding.
5. **Final report + oral defence.**

## References

See `docs/references.md` for the full reference list. Key anchors:

- Somers, Cunningham-Nelson & Boles (2021) — baseline ensemble
- Cunningham-Nelson et al. (2018); Cunningham-Nelson (2019, doctoral thesis)
  — CCU framework, pointer categories, Concept Understanding Matrix
- Ribeiro, Singh & Guestrin (2016) — LIME
- Lundberg & Lee (2017) — SHAP
- DeYoung et al. (2020) — ERASER: comprehensiveness & sufficiency
- Gunasekara & Saarela (2025) — XAI in education, qualitative assessment
- Jain & Wallace (2019); Wiegreffe & Pinter (2019) — attention as
  plausible, not faithful