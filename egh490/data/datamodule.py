"""DataModule: the single entry point for loading data into the pipeline.

Loads either synthetic or real CSV data, maps string labels to integers,
and produces stratified k-fold train/test splits. Every downstream
consumer (Trainer, Ensemble, XAI) receives data through this interface.

Switching from synthetic to real data is a one-line config change::

    data:
      source: "synthetic"   # → change to "real"

The DataModule doesn't care which file it reads — it just needs the
columns defined in ``egh490.data.schema``.

Example
-------
>>> from egh490.data.datamodule import DataModule
>>> dm = DataModule(csv_path="data/synthetic/synthetic_responses.csv", task="validity")
>>> print(dm.texts[:3])
>>> print(dm.labels[:3])
>>> for fold, (train_idx, test_idx) in enumerate(dm.kfold_splits()):
...     train_texts = [dm.texts[i] for i in train_idx]
...     test_texts = [dm.texts[i] for i in test_idx]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from egh490.data.schema import COL_TEXT, get_task_config
from egh490.utils.logging import get_logger

logger = get_logger(__name__)


class DataModule:
    """Load a labelled CSV and prepare it for training and evaluation.

    Parameters
    ----------
    csv_path
        Path to the CSV file (synthetic or real).
    task
        ``"validity"`` or ``"confidence"`` — determines which label column
        is used and how string labels map to integers.
    n_folds
        Number of cross-validation folds. Default 5 per Somers et al.
    seed
        Random seed for reproducible fold splits.
    text_column
        Override the text column name if the real data uses a different
        header. Defaults to the value in ``schema.py``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        task: str = "validity",
        n_folds: int = 5,
        seed: int = 20260413,
        text_column: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.task = task
        self.n_folds = n_folds
        self.seed = seed

        task_cfg = get_task_config(task)
        self.label_column = task_cfg["label_column"]
        self.label_map = task_cfg["label_map"]
        self.label_names = task_cfg["label_names"]
        self.num_labels = task_cfg["num_labels"]
        self.text_col = text_column or COL_TEXT

        self._df = self._load_and_validate()
        self.texts: list[str] = self._df[self.text_col].tolist()
        self.labels: list[int] = self._df["_label"].tolist()

        logger.info(
            "DataModule loaded: %d responses, task=%s, %d folds, labels=%s",
            len(self.texts),
            self.task,
            self.n_folds,
            dict(pd.Series(self.labels).value_counts().sort_index()),
        )

    # ------------------------------------------------------------------ #
    # Loading and validation
    # ------------------------------------------------------------------ #

    def _load_and_validate(self) -> pd.DataFrame:
        """Read CSV, validate required columns, encode labels, drop bad rows."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Data file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Check required columns exist
        for col in [self.text_col, self.label_column]:
            if col not in df.columns:
                raise ValueError(
                    f"Column {col!r} not found in {self.csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Drop rows with missing text or labels
        before = len(df)
        df = df.dropna(subset=[self.text_col, self.label_column]).copy()
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("Dropped %d rows with missing text or labels", dropped)

        # Convert text to string (handles any numeric entries)
        df[self.text_col] = df[self.text_col].astype(str)

        # Single-word responses retained — consistent with training on the
        # full realistic distribution; all responses were manually annotated.

        # Encode labels
        unknown = set(df[self.label_column].unique()) - set(self.label_map.keys())
        if unknown:
            raise ValueError(
                f"Unknown label values in column {self.label_column!r}: {unknown}. "
                f"Expected: {list(self.label_map.keys())}"
            )
        df["_label"] = df[self.label_column].map(self.label_map)

        # Reset index so integer indexing works cleanly
        df = df.reset_index(drop=True)

        return df

    # ------------------------------------------------------------------ #
    # Cross-validation splits
    # ------------------------------------------------------------------ #

    def kfold_splits(self) -> Iterator[tuple[list[int], list[int]]]:
        """Yield ``(train_indices, test_indices)`` for each CV fold.

        Uses stratified k-fold so class balance is preserved in every
        split. Deterministic given the seed.

        Yields
        ------
        tuple[list[int], list[int]]
            Train and test index lists for each fold.
        """
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        labels_array = np.array(self.labels)

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(self.labels)), labels_array)
        ):
            logger.info(
                "Fold %d/%d: %d train, %d test",
                fold_idx + 1,
                self.n_folds,
                len(train_idx),
                len(test_idx),
            )
            yield train_idx.tolist(), test_idx.tolist()

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def get_texts_and_labels(
        self, indices: list[int]
    ) -> tuple[list[str], list[int]]:
        """Return texts and labels for a list of indices."""
        texts = [self.texts[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        return texts, labels

    def get_dataframe(self) -> pd.DataFrame:
        """Return the full cleaned dataframe (includes metadata columns)."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self.texts)