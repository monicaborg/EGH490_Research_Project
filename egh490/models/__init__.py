"""Transformer classifiers, training loop, and ensemble.

Re-exports the public API so callers can write::

    from egh490.models import TransformerClassifier, Trainer, TrainingConfig, Ensemble

instead of importing from individual modules.
"""

from egh490.models.base import TransformerClassifier
from egh490.models.ensemble import Ensemble
from egh490.models.trainer import Trainer, TrainingConfig, compute_metrics

__all__ = [
    "Ensemble",
    "TransformerClassifier",
    "Trainer",
    "TrainingConfig",
    "compute_metrics",
]