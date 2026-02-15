#!/usr/bin/env python3
"""
Create train/valid/test split from record list.

Reads split/data/record_list.csv (columns: study_id, path).
Writes split/data/meta_split.csv (columns: save_file, split).
Save_file format: mimic_iv_ecg_<study_id>.mat
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SOURCE_PREFIX = "mimic_iv_ecg"
DEFAULT_VALID_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1


def get_base_dir(args: argparse.Namespace) -> Path:
    return Path(args.base_dir or os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/valid/test split from record_list.")
    parser.add_argument("--base-dir", type=str, default=None, help="Project root (default: cwd or ECG_FINETUNE_BASE)")
    parser.add_argument("--record-list", type=str, default=None, help="Override path to record_list.csv")
    parser.add_argument("--out", type=str, default=None, help="Override path for meta_split.csv")
    parser.add_argument("--valid-frac", type=float, default=DEFAULT_VALID_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = get_base_dir(args)
    split_data = base / "split" / "data"
    record_path = Path(args.record_list) if args.record_list else split_data / "record_list.csv"
    out_path = Path(args.out) if args.out else split_data / "meta_split.csv"

    if not record_path.exists():
        print(f"ERROR: record_list not found: {record_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(record_path, low_memory=False)
    if "study_id" not in df.columns:
        print("ERROR: record_list.csv must have column 'study_id'", file=sys.stderr)
        return 1

    df = df.drop_duplicates(subset=["study_id"]).copy()
    df["save_file"] = SOURCE_PREFIX + "_" + df["study_id"].astype(str) + ".mat"

    rng = np.random.default_rng(args.seed)
    n = len(df)
    n_valid = max(0, int(n * args.valid_frac))
    n_test = max(0, int(n * args.test_frac))
    n_train = n - n_valid - n_test
    if n_train <= 0:
        print("ERROR: valid_frac + test_frac too large; no train samples", file=sys.stderr)
        return 1

    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = np.array(["train"] * len(df))
    split[indices[n_train : n_train + n_valid]] = "valid"
    split[indices[n_train + n_valid :]] = "test"
    df["split"] = split

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["save_file", "split"]].to_csv(out_path, index=False)
    print(f"Wrote {out_path} (train={n_train}, valid={n_valid}, test={n_test})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
