"""Data loading, schema, and cross-validation splits.

Re-exports the public API so callers can write::

    from egh490.data import DataModule, get_task_config

instead of importing from individual modules.
"""

from egh490.data.datamodule import DataModule
from egh490.data.schema import (
    COL_CCU,
    COL_CONFIDENCE,
    COL_MCQ,
    COL_TEXT,
    COL_UID,
    COL_VALIDITY,
    CONFIDENCE_LABELS,
    VALIDITY_LABELS,
    get_task_config,
)

__all__ = [
    "COL_CCU",
    "COL_CONFIDENCE",
    "COL_MCQ",
    "COL_TEXT",
    "COL_UID",
    "COL_VALIDITY",
    "CONFIDENCE_LABELS",
    "DataModule",
    "VALIDITY_LABELS",
    "get_task_config",
]