#!/usr/bin/env python3
"""
Preprocess MIMIC-IV-ECG raw data: Lead I duplicated to 12 channels (ECG-FM compatible).

Reads: split/data/meta_split.csv, split/data/record_list.csv. Writes: preprocess/data/lead_1_duplicated/.
Raw WFDB root: pass via --raw-root (download MIMIC-IV-ECG from PhysioNet).
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

DESIRED_SAMPLE_RATE = 500
EXPECTED_SAMPLES_10S = DESIRED_SAMPLE_RATE * 10
DIR_LEAD1 = "lead_1_duplicated"


def get_base_dir(args: argparse.Namespace) -> Path:
    return Path(args.base_dir or os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


def log(msg: str) -> None:
    print(msg, flush=True)


def resample(feats: np.ndarray, curr_sample_rate: float, desired_sample_rate: int) -> np.ndarray:
    if curr_sample_rate == desired_sample_rate:
        return feats
    desired_len = int(feats.shape[-1] * (desired_sample_rate / curr_sample_rate))
    x = np.linspace(0, desired_len - 1, feats.shape[-1])
    return interp1d(x, feats, kind="linear", axis=-1)(np.arange(desired_len)).astype(np.float32)


def lead_std_divide(feats: np.ndarray, constant_lead_strategy: str = "zero") -> tuple:
    std = np.nanstd(feats, axis=1, keepdims=True)
    std_zero = (std == 0) | np.isnan(std)
    if not std_zero.any():
        feats = feats / std
        return feats, std
    std_replaced = np.where(std_zero, 1.0, std)
    feats = feats / std_replaced
    if constant_lead_strategy == "zero":
        feats = np.where(np.broadcast_to(std_zero, feats.shape), 0.0, feats)
    return feats.astype(np.float32), std


def standardize(feats: np.ndarray) -> np.ndarray:
    mean = np.nanmean(feats, axis=1, keepdims=True)
    feats = feats - mean
    feats, _ = lead_std_divide(feats, constant_lead_strategy="zero")
    return feats


def load_wfdb_lead1_duplicated(wfdb_path: Path) -> tuple:
    try:
        record = wfdb.rdrecord(str(wfdb_path))
        if record.p_signal is None:
            return None, None
        sig = record.p_signal
        sig_names = getattr(record, "sig_name", None) or []
        fs = float(record.fs)
        name_to_idx = {n: i for i, n in enumerate(sig_names)}
        if "I" not in name_to_idx:
            return None, None
        lead1 = np.asarray(sig[:, name_to_idx["I"]], dtype=np.float32)
        feats = np.tile(lead1, (12, 1))
        return feats, fs
    except Exception:
        return None, None


def process_one_record(wfdb_path: Path, study_id: int, save_file: str, out_dir: Path) -> int:
    feats, fs = load_wfdb_lead1_duplicated(wfdb_path)
    if feats is None or fs is None:
        return 0
    feats = resample(feats, fs, DESIRED_SAMPLE_RATE)
    feats = standardize(feats)
    out_path = out_dir / save_file
    scipy.io.savemat(
        str(out_path),
        {
            "feats": feats,
            "org_sample_rate": np.array([[DESIRED_SAMPLE_RATE]], dtype=np.float64),
            "curr_sample_rate": np.array([[DESIRED_SAMPLE_RATE]], dtype=np.float64),
            "org_sample_size": np.array([[feats.shape[1]]], dtype=np.int64),
            "curr_sample_size": np.array([[feats.shape[1]]], dtype=np.int64),
            "idx": np.array([[study_id]], dtype=np.int64),
        },
        format="5",
        do_compression=False,
    )
    return 1


def run_checks(out_dir: Path) -> bool:
    all_ok = True
    files = list(out_dir.glob("*.mat"))[:50]
    if not files:
        log("   [CHECK] lead_1_duplicated: no .mat files")
        return False
    for f in files:
        try:
            data = scipy.io.loadmat(str(f), variable_names=["feats", "curr_sample_rate", "idx"])
            feats, rate = data["feats"], int(np.squeeze(data["curr_sample_rate"]))
            if feats.shape != (12, EXPECTED_SAMPLES_10S) or feats.dtype != np.float32:
                log(f"   [FAIL] {f.name}: bad shape/dtype"); all_ok = False
            if rate != 500:
                log(f"   [FAIL] {f.name}: rate != 500"); all_ok = False
        except Exception as e:
            log(f"   [FAIL] {f.name}: {e}")
            all_ok = False
    if all_ok:
        log("   [PASS] lead_1_duplicated: sampled .mat OK")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV-ECG: Lead I → 12 ch, 10 s .mat.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--record-list", type=str, default=None)
    parser.add_argument("--raw-root", type=str, default=None, help="Raw WFDB root (MIMIC-IV-ECG download)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-checks", action="store_true")
    args = parser.parse_args()

    base = get_base_dir(args)
    split_data = base / "split" / "data"
    preprocess_data = base / "preprocess" / "data"
    meta_path = Path(args.meta) if args.meta else split_data / "meta_split.csv"
    record_list_path = Path(args.record_list) if args.record_list else split_data / "record_list.csv"
    raw_root = Path(args.raw_root) if args.raw_root else None
    if not raw_root:
        log("ERROR: --raw-root is required (path to downloaded MIMIC-IV-ECG root)")
        return 1
    out_dir = Path(args.out) if args.out else preprocess_data / DIR_LEAD1

    log("=" * 70)
    log("Preprocess: Lead I duplicated → 10 s .mat")
    log("=" * 70)
    log(f"Base:        {base}")
    log(f"Meta:        {meta_path}")
    log(f"Record list: {record_list_path}")
    log(f"Raw root:    {raw_root}")
    log(f"Output:      {out_dir}")

    if not meta_path.exists():
        log(f"ERROR: meta split not found: {meta_path}")
        return 1
    if not record_list_path.exists():
        log(f"ERROR: record_list not found: {record_list_path}")
        return 1
    if not raw_root.exists():
        log(f"ERROR: raw root not found: {raw_root}")
        return 1

    meta = pd.read_csv(meta_path, low_memory=False)
    if "split" not in meta.columns or "save_file" not in meta.columns:
        log("ERROR: meta must have columns 'split' and 'save_file'")
        return 1
    meta["study_id"] = meta["save_file"].str.replace(".mat", "", regex=False).str.split("_").str[-1].astype(int)

    record_df = pd.read_csv(record_list_path, low_memory=False)
    record_map = {int(row["study_id"]): raw_root / row["path"] for _, row in record_df.iterrows()}

    out_dir.mkdir(parents=True, exist_ok=True)
    processed = total = failed = 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing"):
        study_id = row["study_id"]
        if study_id not in record_map:
            failed += 1
            continue
        n = process_one_record(record_map[study_id], study_id, row["save_file"], out_dir)
        if n:
            processed += 1
            total += n

    log(f"Processed: {processed} records, {total} files; failed/skipped: {failed}")
    if not args.no_checks:
        log("Running checks...")
        run_checks(out_dir)
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
