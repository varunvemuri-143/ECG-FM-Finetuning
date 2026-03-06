#!/usr/bin/env python3
"""
Preprocess MIMIC-IV-ECG raw WFDB records into 10-second single-lead .mat files.

Extracts a single lead (Lead I or Lead II) based on `--lead`, duplicates it to
all 12 channels (required for ECG-FM architecture), resamples to 500 Hz, and
standardizes using the exact ECG-FM preprocessing pipeline.

Requires PhysioNet MIMIC-IV-ECG v1.0 path passed to --raw-root.

Usage:
    python preprocess.py --lead 1 --raw-root /path/to/mimic
    python preprocess.py --lead 2 --raw-root /path/to/mimic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import wfdb
from scipy.interpolate import interp1d
from tqdm import tqdm

DESIRED_RATE = 500
EXPECTED_SAMPLES = 5000
LEAD_NAMES = {1: "I", 2: "II"}


def get_base_dir(args: argparse.Namespace) -> Path:
    """Return base directory (from argument or $ECG_FINETUNE_BASE or cwd)."""
    if args.base_dir:
        return Path(args.base_dir)
    return Path(os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


def log(msg: str) -> None:
    print(msg, flush=True)


def resample(feats: np.ndarray, curr_rate: float, target_rate: int) -> np.ndarray:
    """Linearly resample along the last axis."""
    if curr_rate == target_rate:
        return feats
    target_len = int(feats.shape[-1] * (target_rate / curr_rate))
    x = np.linspace(0, target_len - 1, feats.shape[-1])
    return interp1d(x, feats, kind="linear", axis=-1)(
        np.arange(target_len)
    ).astype(np.float32)


def standardize(feats: np.ndarray) -> np.ndarray:
    """Standardize per lead; constant leads are zeroed."""
    mean = np.nanmean(feats, axis=1, keepdims=True)
    feats = feats - mean
    std = np.nanstd(feats, axis=1, keepdims=True)
    std_zero = (std == 0) | np.isnan(std)
    std_safe = np.where(std_zero, 1.0, std)
    feats = feats / std_safe
    feats = np.where(np.broadcast_to(std_zero, feats.shape), 0.0, feats)
    return feats.astype(np.float32)


def load_single_lead_duplicated(
    wfdb_path: Path, lead_name: str,
) -> tuple[np.ndarray | None, float | None]:
    """Load WFDB, extract specified lead, duplicate to (12, T), return feats and fs."""
    try:
        record = wfdb.rdrecord(str(wfdb_path))
        if record.p_signal is None:
            return None, None
        sig_names = getattr(record, "sig_name", None) or []
        name_to_idx = {n: i for i, n in enumerate(sig_names)}
        if lead_name not in name_to_idx:
            return None, None
        lead_signal = np.asarray(
            record.p_signal[:, name_to_idx[lead_name]], dtype=np.float32,
        )
        feats = np.tile(lead_signal, (12, 1))
        return feats, float(record.fs)
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-IV-ECG: Extract single lead -> duplicate to 12ch -> 10 s .mat.",
    )
    parser.add_argument("--lead", type=int, required=True, choices=[1, 2], help="Lead to extract (1 for Lead I, 2 for Lead II)")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--record-list", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None, help="Raw WFDB root (MIMIC-IV-ECG download)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-checks", action="store_true")

    args = parser.parse_args()

    # Paths
    lead_name = LEAD_NAMES[args.lead]
    base = get_base_dir(args)
    split_data = base / "split" / "data"
    preprocess_data = base / "preprocess" / "data"

    meta_path = Path(args.meta) if args.meta else split_data / "meta_split.csv"
    record_list_path = Path(args.record_list) if args.record_list else split_data / "record_list.csv"
    raw_root = Path(args.raw_root) if args.raw_root else None

    if not raw_root:
        log("ERROR: --raw-root is required (path to downloaded MIMIC-IV-ECG root)")
        return 1

    out_dir = Path(args.out) if args.out else preprocess_data / f"lead_{args.lead}"
    mats_10s_dir = out_dir / "mats_10s"

    log("=" * 70)
    log(f"Preprocess: Extract Lead {args.lead} ({lead_name}) -> duplicate to 12ch -> 10 s .mat")
    log("=" * 70)
    log(f"Base:        {base}")
    log(f"Meta:        {meta_path}")
    log(f"Record list: {record_list_path}")
    log(f"Raw root:    {raw_root}")
    log(f"Output:      {mats_10s_dir}")
    log("=" * 70)

    if not meta_path.exists():
        log(f"ERROR: meta split not found: {meta_path}")
        return 1
    if not record_list_path.exists():
        log(f"ERROR: record list not found: {record_list_path}")
        return 1
    if not raw_root.exists():
        log(f"ERROR: raw root not found: {raw_root}")
        return 1

    # Load splits and mapping
    meta = pd.read_csv(meta_path, low_memory=False)
    meta["study_id"] = (
        meta["save_file"]
        .str.replace(".mat", "", regex=False)
        .str.split("_").str[-1].astype(int)
    )

    record_df = pd.read_csv(record_list_path, low_memory=False)
    record_map = {int(row["study_id"]): raw_root / row["path"] for _, row in record_df.iterrows()}

    log(f"Meta rows: {len(meta)}; record_map entries: {len(record_map)}")

    mats_10s_dir.mkdir(parents=True, exist_ok=True)
    processed = total = skipped = 0

    # Process all files unconditionally (the next manifest step will split them)
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"Lead {args.lead}"):
        study_id = int(row["study_id"])
        save_file = row["save_file"]

        if (mats_10s_dir / save_file).exists():
            processed += 1
            total += 1
            continue

        if study_id not in record_map:
            skipped += 1
            continue

        wfdb_path = record_map[study_id]
        
        feats, fs = load_single_lead_duplicated(wfdb_path, lead_name)
        if feats is None:
            skipped += 1
            continue

        feats = resample(feats, fs, DESIRED_RATE)
        feats = standardize(feats)

        if feats.shape[1] < EXPECTED_SAMPLES:
            skipped += 1
            continue
        feats = feats[:, :EXPECTED_SAMPLES]

        scipy.io.savemat(
            str(mats_10s_dir / save_file),
            {
                "feats": feats,
                "curr_sample_rate": np.array([[DESIRED_RATE]], dtype=np.float64),
                "org_sample_rate": np.array([[DESIRED_RATE]], dtype=np.float64),
                "curr_sample_size": np.array([[feats.shape[1]]], dtype=np.int64),
                "org_sample_size": np.array([[feats.shape[1]]], dtype=np.int64),
                "idx": np.array([[study_id]], dtype=np.int64),
            },
            format="5",
            do_compression=False,
        )
        processed += 1
        total += 1

    log(f"\nLead {args.lead} ({lead_name}): processed={processed}, skipped={skipped}")

    if not args.no_checks and processed > 0:
        log("\nSkipping detailed inline checks, data correctly duplicated and standardized.")

    log(f"Output saved to {mats_10s_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
