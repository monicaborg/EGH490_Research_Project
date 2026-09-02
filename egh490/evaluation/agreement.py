"""Inter-rater agreement between two independent markers.

Computes Cohen's kappa and raw percent agreement between your own markings
(the "base" ratings, e.g. Monica's) and a second marker's ratings (e.g.
Sam's), for the three marked columns: concept mentioned, response
certainty, and free-text validity.

Designed for the common real-world case where the second marker only
reviews a sample rather than the full dataset: rows are matched on ID, and
agreement is computed only over the overlapping subset. Coverage (how many
rows overlapped) is always reported alongside the kappa, since a kappa
computed over 20 rows means something different from one computed over 2000.

Kappa is interpreted using the Landis & Koch (1977) benchmarks, which is the
standard reference cited in annotation-agreement literature:
    < 0.00        no agreement (worse than chance)
    0.00 - 0.20   slight
    0.21 - 0.40   fair
    0.41 - 0.60   moderate
    0.61 - 0.80   substantial
    0.81 - 1.00   almost perfect

Example
-------
>>> from egh490.evaluation.agreement import compare_markings
>>> report = compare_markings("data/raw/Monica_Data_CCUS.csv", "data/raw/Sam_marked.csv")
>>> report.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ------------------------------------------------------------------ #
# Column configuration
# ------------------------------------------------------------------ #

# The three columns markers assign. Keys are the canonical short names used
# throughout this module; values are the possible header spellings seen
# across the different exports so far (the header text has varied slightly
# between files over the course of the project).
MARK_COLUMNS: dict[str, list[str]] = {
    "concept": ["Concept Mentioned", "concept mentioned"],
    "certainty": ["Response Certainty", "Response Certainty ", "response certainty",
                  "response uncertainty"],
    "validity": ["Free-Text Validity", " free-text validity", "free-text validity"],
}

ID_COLUMNS = ["ID", "uid", "id"]
CCU_COLUMNS = ["CCU", "ccuname", "ccu"]

KAPPA_BENCHMARKS = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.00, "slight"),
    (float("-inf"), "no agreement (at or below chance)"),
]


def interpret_kappa(k: float) -> str:
    """Return the Landis & Koch (1977) qualitative label for a kappa value."""
    if k != k:  # NaN check without importing math
        return "undefined (insufficient variation)"
    for threshold, label in KAPPA_BENCHMARKS:
        if k >= threshold:
            return label
    return "no agreement"


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case/whitespace tolerant)."""
    normalised = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalised:
            return normalised[key]
    return None


def _load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


# ------------------------------------------------------------------ #
# Result containers
# ------------------------------------------------------------------ #

@dataclass
class ColumnAgreement:
    """Agreement statistics for a single marked column (e.g. validity)."""

    column: str
    n_overlap: int
    kappa: float
    percent_agreement: float
    disagreement_ids: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "column": self.column,
            "n_overlap": self.n_overlap,
            "kappa": None if self.kappa != self.kappa else round(self.kappa, 4),
            "kappa_interpretation": interpret_kappa(self.kappa),
            "percent_agreement": round(self.percent_agreement, 4),
            "n_disagreements": len(self.disagreement_ids),
        }


@dataclass
class CCUAgreementReport:
    """Agreement statistics for one CCU, across all three marked columns."""

    ccu: str
    n_overlap: int
    columns: dict[str, ColumnAgreement]


@dataclass
class AgreementReport:
    """Full agreement report: overall and per-CCU, for all three columns."""

    overall: dict[str, ColumnAgreement]
    per_ccu: dict[str, CCUAgreementReport]
    n_base_rows: int
    n_second_rows: int
    n_matched_rows: int
    unmatched_base_ids: list = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a readable summary table to the console."""
        print("=" * 70)
        print("INTER-RATER AGREEMENT SUMMARY")
        print("=" * 70)
        print(f"Base marker rows:    {self.n_base_rows}")
        print(f"Second marker rows:  {self.n_second_rows}")
        print(f"Matched on ID:       {self.n_matched_rows}")
        if self.unmatched_base_ids:
            print(f"Base rows with no match in second marker: {len(self.unmatched_base_ids)}")
        print()

        print("-" * 70)
        print("OVERALL (all CCUs combined)")
        print("-" * 70)
        print(f"{'Column':<12} {'n':>6} {'% agree':>10} {'kappa':>8}  Interpretation")
        for name, ca in self.overall.items():
            k = "n/a" if ca.kappa != ca.kappa else f"{ca.kappa:.3f}"
            print(f"{name:<12} {ca.n_overlap:>6} {ca.percent_agreement*100:>9.1f}% "
                  f"{k:>8}  {interpret_kappa(ca.kappa)}")
        print()

        print("-" * 70)
        print("PER CCU")
        print("-" * 70)
        for ccu, rep in sorted(self.per_ccu.items()):
            print(f"\nCCU {ccu}  (n={rep.n_overlap} matched)")
            for name, ca in rep.columns.items():
                k = "n/a" if ca.kappa != ca.kappa else f"{ca.kappa:.3f}"
                print(f"  {name:<12} {ca.n_overlap:>5} rows  "
                      f"{ca.percent_agreement*100:>6.1f}% agree  kappa={k}  "
                      f"({interpret_kappa(ca.kappa)})")
        print("=" * 70)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the report into a tidy DataFrame for export to Excel/CSV."""
        rows = []
        for name, ca in self.overall.items():
            d = ca.as_dict()
            d["ccu"] = "ALL"
            rows.append(d)
        for ccu, rep in self.per_ccu.items():
            for name, ca in rep.columns.items():
                d = ca.as_dict()
                d["ccu"] = ccu
                rows.append(d)
        df = pd.DataFrame(rows)
        return df[["ccu", "column", "n_overlap", "percent_agreement",
                    "kappa", "kappa_interpretation", "n_disagreements"]]

    def disagreement_rows(self, base_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Return the base-marker rows where any column disagreed, for review."""
        all_ids = set()
        for ca in self.overall.values():
            all_ids.update(ca.disagreement_ids)
        return base_df[base_df[id_col].isin(all_ids)]


# ------------------------------------------------------------------ #
# Core computation
# ------------------------------------------------------------------ #

def _column_agreement(merged: pd.DataFrame, base_col: str, second_col: str,
                       id_col: str, name: str) -> ColumnAgreement:
    sub = merged[[id_col, base_col, second_col]].dropna()
    n = len(sub)
    if n == 0:
        return ColumnAgreement(name, 0, float("nan"), float("nan"), [])

    a = sub[base_col].astype(int)
    b = sub[second_col].astype(int)

    agree_mask = a == b
    pct = agree_mask.mean()

    # cohen_kappa_score needs both raters to show some variation; if either
    # rater used only one label across the overlap, kappa is undefined.
    if a.nunique() < 2 and b.nunique() < 2:
        kappa = float("nan")
    else:
        kappa = cohen_kappa_score(a, b)

    disagreement_ids = sub.loc[~agree_mask, id_col].tolist()
    return ColumnAgreement(name, n, kappa, pct, disagreement_ids)


def compare_markings(
    base_path: str | Path,
    second_path: str | Path,
    *,
    base_label: str = "base",
    second_label: str = "second",
) -> AgreementReport:
    """Compute Cohen's kappa between two markers' files.

    Parameters
    ----------
    base_path
        Path to your own marked CSV/XLSX (e.g. Monica's markings). Must have
        an ID column and the three mark columns (concept/certainty/validity)
        under any of the recognised header spellings.
    second_path
        Path to the second marker's file (e.g. Sam's). Can cover the full
        dataset or just a sample — agreement is computed only over rows
        present in both files, matched by ID.
    base_label, second_label
        Optional display names for the two markers (currently used only for
        clarity if you print `merged` columns yourself; the report itself is
        symmetric and doesn't depend on which side is "base").

    Returns
    -------
    AgreementReport
        Overall and per-CCU kappa / percent agreement for each of the three
        marked columns, plus lists of disagreeing IDs for qualitative review.
    """
    base_df = _load(base_path)
    second_df = _load(second_path)

    base_id = _find_column(base_df, ID_COLUMNS)
    second_id = _find_column(second_df, ID_COLUMNS)
    if base_id is None or second_id is None:
        raise ValueError(
            "Could not find an ID column in one of the files. Expected one "
            f"of {ID_COLUMNS}. Base columns: {list(base_df.columns)}. "
            f"Second columns: {list(second_df.columns)}."
        )

    base_ccu = _find_column(base_df, CCU_COLUMNS)
    second_ccu = _find_column(second_df, CCU_COLUMNS)

    # Resolve mark columns on each side independently, since header spelling
    # has varied between exports over the course of the project.
    base_marks = {}
    second_marks = {}
    for short_name, candidates in MARK_COLUMNS.items():
        bc = _find_column(base_df, candidates)
        sc = _find_column(second_df, candidates)
        if bc is None:
            raise ValueError(f"Could not find '{short_name}' column in base file. "
                              f"Tried {candidates}. Have: {list(base_df.columns)}")
        if sc is None:
            raise ValueError(f"Could not find '{short_name}' column in second file. "
                              f"Tried {candidates}. Have: {list(second_df.columns)}")
        base_marks[short_name] = bc
        second_marks[short_name] = sc

    # Rename to disambiguated columns before merging, so overlapping header
    # names (e.g. both files using "Free-Text Validity") don't collide.
    base_renamed = base_df.rename(columns={
        base_id: "_id", **{base_marks[k]: f"base__{k}" for k in MARK_COLUMNS}
    })
    second_renamed = second_df.rename(columns={
        second_id: "_id", **{second_marks[k]: f"second__{k}" for k in MARK_COLUMNS}
    })
    if base_ccu:
        base_renamed = base_renamed.rename(columns={base_ccu: "_ccu"})
    if second_ccu:
        second_renamed = second_renamed.rename(columns={second_ccu: "_ccu"})

    base_keep = ["_id"] + (["_ccu"] if base_ccu else []) + [f"base__{k}" for k in MARK_COLUMNS]
    second_keep = ["_id"] + [f"second__{k}" for k in MARK_COLUMNS]

    merged = base_renamed[base_keep].merge(
        second_renamed[second_keep], on="_id", how="inner"
    )

    unmatched = set(base_renamed["_id"]) - set(merged["_id"])

    overall = {}
    for name in MARK_COLUMNS:
        overall[name] = _column_agreement(
            merged, f"base__{name}", f"second__{name}", "_id", name
        )

    per_ccu: dict[str, CCUAgreementReport] = {}
    if "_ccu" in merged.columns:
        for ccu_value, group in merged.groupby("_ccu"):
            cols = {}
            for name in MARK_COLUMNS:
                cols[name] = _column_agreement(
                    group, f"base__{name}", f"second__{name}", "_id", name
                )
            per_ccu[str(ccu_value)] = CCUAgreementReport(
                ccu=str(ccu_value), n_overlap=len(group), columns=cols
            )

    return AgreementReport(
        overall=overall,
        per_ccu=per_ccu,
        n_base_rows=len(base_df),
        n_second_rows=len(second_df),
        n_matched_rows=len(merged),
        unmatched_base_ids=sorted(unmatched),
    )
