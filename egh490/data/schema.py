"""Schema for a single student response record.

Defines the column names and label mappings used across the data layer.
These constants are the single source of truth for column names — if the
real dataset from Sam uses slightly different names, only this file needs
to change.

The schema deliberately mirrors the real CSV structure:
    uid, ccuname, time, q1mcr, q1txr, q2mcr, q3mcr, validity, confidence

The classifier only uses ``q1txr`` (the free-text response) as input and
one of the two label columns as the target. The remaining columns are
metadata preserved for analysis but not fed to the model.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
# Column name constants
# ------------------------------------------------------------------ #

COL_UID = "uid"
COL_CCU = "ccuname"
COL_TIME = "time"
COL_MCQ = "q1mcr"
COL_TEXT = "q1txr"          # the free-text response — model input
COL_MCQ2 = "q2mcr"
COL_MCQ3 = "q3mcr"
COL_VALIDITY = "validity"
COL_CONFIDENCE = "confidence"

# ------------------------------------------------------------------ #
# Label mappings (string → int)
# ------------------------------------------------------------------ #

VALIDITY_LABELS = {"incorrect": 0, "correct": 1}
CONFIDENCE_LABELS = {"low": 0, "high": 1}

# Reverse mappings for display
VALIDITY_NAMES = {v: k for k, v in VALIDITY_LABELS.items()}
CONFIDENCE_NAMES = {v: k for k, v in CONFIDENCE_LABELS.items()}

# ------------------------------------------------------------------ #
# Task configuration
# ------------------------------------------------------------------ #

TASK_CONFIG = {
    "validity": {
        "label_column": COL_VALIDITY,
        "label_map": VALIDITY_LABELS,
        "label_names": VALIDITY_NAMES,
        "num_labels": 2,
    },
    "confidence": {
        "label_column": COL_CONFIDENCE,
        "label_map": CONFIDENCE_LABELS,
        "label_names": CONFIDENCE_NAMES,
        "num_labels": 2,
    },
}


def get_task_config(task: str) -> dict:
    """Return the label column, mapping, and num_labels for a task.

    Parameters
    ----------
    task
        Either ``"validity"`` or ``"confidence"``.

    Raises
    ------
    ValueError
        If the task name is not recognised.
    """
    if task not in TASK_CONFIG:
        raise ValueError(
            f"Unknown task {task!r}. Must be one of {list(TASK_CONFIG.keys())}"
        )
    return TASK_CONFIG[task]