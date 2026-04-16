"""Transformer classifiers, training loop, and ensemble.

Re-exports the public API so callers can write::

    from egh490.models import TransformerClassifier

instead of importing from the individual module.
"""

from egh490.models.base import TransformerClassifier

__all__ = ["TransformerClassifier"]