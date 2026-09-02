"""Prepare two dataset variants for the full-vs-clean ablation comparison.

Produces:
  1. Monica_Data_CCUS_converted.csv        - all deduplicated rows (unchanged)
  2. Monica_Data_CCUS_converted_clean.csv  - the same, minus rows marked as
     junk during manual annotation (concept mentioned = 0, response
     certainty = 0, free-text validity = 0 — the explicit "gibberish /
     single-char / blank / idk / guess" convention used throughout marking).

Using the human marking to define "junk" (rather than a new text-length or
regex heuristic) keeps the two dataset variants grounded in the same
annotation methodology already documented for the rest of the project.

Usage
-----
    python scripts/prepare_datasets.py \\
        --input data/raw/Monica_Data_CCUS.csv \\
        --output-dir data/raw
"""
import argparse
import pandas as pd


def find_col(df, candidates):
    normalised = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalised:
            return normalised[key]
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)

    col_map = {
        find_col(df, ["ID", "uid"]): "uid",
        find_col(df, ["CCU", "ccuname"]): "_ccu_raw",
        find_col(df, ["Time", "time"]): "time",
        find_col(df, ["Q1 MCR", "q1mcr"]): "q1mcr",
        find_col(df, ["Q1 Text Response", "q1txr"]): "q1txr",
        find_col(df, ["q2mcr"]): "q2mcr",
        find_col(df, ["q3mcr"]): "q3mcr",
        find_col(df, ["Concept Mentioned", "concept mentioned"]): "_concept_raw",
        find_col(df, ["Response Certainty", "response certainty"]): "_certainty_raw",
        find_col(df, ["Free-Text Validity", "free-text validity"]): "_validity_raw",
    }
    df = df.rename(columns=col_map)

    df["ccuname"] = "ccu" + df["_ccu_raw"].astype(str)
    df["validity"] = df["_validity_raw"].map({1: "correct", 0: "incorrect"})
    df["confidence"] = df["_certainty_raw"].map({1: "high", 0: "low"})
    df["is_junk"] = (
        (df["_concept_raw"] == 0) & (df["_certainty_raw"] == 0) & (df["_validity_raw"] == 0)
    )

    out_cols = ["uid", "ccuname", "time", "q1mcr", "q1txr", "q2mcr", "q3mcr",
                "validity", "confidence"]

    full_path = f"{args.output_dir}/Monica_Data_CCUS_converted.csv"
    clean_path = f"{args.output_dir}/Monica_Data_CCUS_converted_clean.csv"

    df[out_cols].to_csv(full_path, index=False)
    df.loc[~df["is_junk"], out_cols].to_csv(clean_path, index=False)

    print(f"Full dataset:  {len(df)} rows -> {full_path}")
    print(f"Clean dataset: {(~df['is_junk']).sum()} rows -> {clean_path} "
          f"({df['is_junk'].sum()} junk rows removed)")
    print()
    print("Junk rows removed per CCU:")
    print(df.groupby("ccuname")["is_junk"].sum())
    print()
    print("Remaining rows per CCU (clean dataset):")
    print(df.loc[~df["is_junk"]].groupby("ccuname").size())


if __name__ == "__main__":
    main()
