"""Compute Cohen's kappa agreement between your markings and a second marker's.

Once Sam returns his markings (full dataset or a reviewed sample), run this
to get kappa and percent agreement for concept/certainty/validity, both
overall and per CCU, plus a CSV of the disagreeing rows for follow-up.

Usage
-----
    python scripts/compute_agreement.py \\
        --base data/raw/Monica_Data_CCUS.csv \\
        --second data/raw/Sam_marked.csv \\
        --output-dir outputs/agreement

The two files just need an ID column and the three mark columns
(concept mentioned / response certainty / free-text validity, under any of
the header spellings used across this project) — they don't need to have
identical structure otherwise, and the second file can cover only a sample
of rows (agreement is computed over whatever IDs are present in both).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute inter-rater agreement (Cohen's kappa).")
    p.add_argument("--base", required=True, help="Your own marked CSV/XLSX (e.g. Monica's)")
    p.add_argument("--second", required=True, help="The second marker's CSV/XLSX (e.g. Sam's)")
    p.add_argument("--output-dir", default="outputs/agreement",
                    help="Where to save the summary table and disagreement rows")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from egh490.evaluation.agreement import compare_markings

    report = compare_markings(args.base, args.second)
    report.print_summary()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = report.to_dataframe()
    summary_path = out_dir / "agreement_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table -> {summary_path}")

    # Disagreement rows for qualitative follow-up (pull from the base file
    # so the original text/context is included, not just IDs).
    import pandas as pd
    base_path = Path(args.base)
    base_df = pd.read_csv(base_path) if base_path.suffix.lower() == ".csv" else pd.read_excel(base_path)
    id_col = next((c for c in base_df.columns if c.strip().lower() in ("id", "uid")), None)

    if id_col:
        dis_df = report.disagreement_rows(base_df, id_col)
        dis_path = out_dir / "disagreement_rows.csv"
        dis_df.to_csv(dis_path, index=False)
        print(f"Saved {len(dis_df)} disagreement rows -> {dis_path}")

    if report.unmatched_base_ids:
        print(f"\nNote: {len(report.unmatched_base_ids)} of your rows had no "
              f"matching ID in the second marker's file (expected if Sam "
              f"reviewed a sample rather than the full dataset).")


if __name__ == "__main__":
    main()
