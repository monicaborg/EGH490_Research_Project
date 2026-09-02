"""Convert combined CCU export into the schema DataModule expects.

Maps column names and label encodings from the raw marked export
(CCU, ID, Time, Q1 MCR, Q1 Text Response, q2mcr, q3mcr, MCQ Correctness,
Concept Mentioned, Response Certainty, Free-Text Validity) to the schema
columns (uid, ccuname, time, q1mcr, q1txr, q2mcr, q3mcr, validity, confidence)
with string label values, so no changes are needed to DataModule or schema.py.

Usage:
    python scripts/convert_raw_data.py \
        --input data/raw/Monica_Data_CCUS.csv \
        --output data/raw/Monica_Data_CCUS_converted.csv
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
    p.add_argument("--output", required=True)
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
        find_col(df, ["Response Certainty", "response certainty"]): "_certainty_raw",
        find_col(df, ["Free-Text Validity", "free-text validity"]): "_validity_raw",
    }
    df = df.rename(columns=col_map)

    df["ccuname"] = "ccu" + df["_ccu_raw"].astype(str)
    df["validity"] = df["_validity_raw"].map({1: "correct", 0: "incorrect"})
    df["confidence"] = df["_certainty_raw"].map({1: "high", 0: "low"})

    out_cols = ["uid", "ccuname", "time", "q1mcr", "q1txr", "q2mcr", "q3mcr",
                "validity", "confidence"]
    df[out_cols].to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}")
    print(df["ccuname"].value_counts())


if __name__ == "__main__":
    main()