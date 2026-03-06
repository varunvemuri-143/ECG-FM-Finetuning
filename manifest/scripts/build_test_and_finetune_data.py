#!/usr/bin/env python3
"""
Build ECG-FM manifests and 5s segments for single-lead data.

Reads 10s preprocessed `.mat` files from `preprocess/data/lead_{lead}/mats_10s/`,
segments them into 5s non-overlapping chunks, and builds the Fairseq manifestations
(train.tsv, valid.tsv, test.tsv, y.npy, label_def.csv, pos_weight.txt) in
`manifest/data/lead_{lead}/manifests/`.

Usage:
    python build_test_and_finetune_data.py --lead 1
    python build_test_and_finetune_data.py --lead 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError
from tqdm import tqdm

SEGMENT_SAMPLES = 2500
LEAD_NAMES = {1: "I", 2: "II"}


def get_base_dir(args: argparse.Namespace) -> Path:
    """Return base directory (from argument or $ECG_FINETUNE_BASE or cwd)."""
    if args.base_dir:
        return Path(args.base_dir)
    return Path(os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


def log(msg: str) -> None:
    print(msg, flush=True)


def build_label_accessor(labels_final: Path) -> tuple[Callable[[int], np.ndarray], set[int], list[str], Path]:
    """
    Returns a fast accessor function for study_id -> label row,
    along with the set of valid study IDs, column names, and def path.
    """
    labels_csv = labels_final / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"{labels_csv} not found")

    label_def_path = labels_final / "label_def_recomputed.csv"
    if not label_def_path.exists():
        label_def_path = labels_final / "label_def.csv"
    if not label_def_path.exists():
        raise FileNotFoundError(f"{label_def_path} not found")

    label_cols = pd.read_csv(label_def_path)["name"].tolist()
    labels_df = pd.read_csv(labels_csv)
    if not all(c in labels_df.columns for c in label_cols):
        raise ValueError(f"{labels_csv} missing columns from label_def")

    labels_by_idx = labels_df.set_index("idx")[label_cols]
    mapping_path = labels_final / "study_id_mapping.csv"

    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        study_id_to_idx = mapping.set_index("study_id")["idx"].to_dict()
        valid_ids = set(int(s) for s in study_id_to_idx.keys())

        def get_row(study_id: int) -> np.ndarray:
            idx = study_id_to_idx[study_id]
            return labels_by_idx.loc[idx].values.astype(np.float32)
    else:
        valid_ids = set(labels_df["idx"].astype(int))

        def get_row(study_id: int) -> np.ndarray:
            return labels_by_idx.loc[study_id].values.astype(np.float32)

    return get_row, valid_ids, label_cols, label_def_path


def expected_5s_paths(out_dir: Path, base_name: str) -> list[Path]:
    """Expected 5s segment paths for one 10s file (always 2 segments)."""
    return [out_dir / f"{base_name}_0.mat", out_dir / f"{base_name}_1.mat"]


def segment_10s_to_5s(
    src_mat: Path, out_dir: Path, base_name: str, study_id: int,
) -> list[Path]:
    data = loadmat(str(src_mat), variable_names=["feats", "curr_sample_rate"])
    feats = data["feats"]
    rate = int(np.squeeze(data["curr_sample_rate"]))
    if rate != 500:
        raise ValueError(f"Expected 500 Hz, got {rate}")
    n_seg = feats.shape[-1] // SEGMENT_SAMPLES
    out_paths = []
    for seg_i in range(n_seg):
        seg_feats = feats[
            :, seg_i * SEGMENT_SAMPLES : (seg_i + 1) * SEGMENT_SAMPLES
        ].copy()
        seg_data = {
            "feats": seg_feats,
            "curr_sample_rate": np.array([[500]], dtype=np.float64),
            "idx": np.array([[0]], dtype=np.int64),
            "original_idx": np.array([[study_id]], dtype=np.int64),
        }
        out_path = out_dir / f"{base_name}_{seg_i}.mat"
        savemat(str(out_path), seg_data, format="5", do_compression=False)
        out_paths.append(out_path)
    return out_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ECG-FM manifests and 5s segments for single-lead data.",
    )
    parser.add_argument("--lead", type=int, required=True, choices=[1, 2], help="Lead processed (1 for Lead I, 2 for Lead II)")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    args = parser.parse_args()

    # Paths
    lead_name = LEAD_NAMES[args.lead]
    base = get_base_dir(args)
    split_data = base / "split" / "data"
    labels_final = base / "labels" / "computed_labels"
    mats_10s = base / "preprocess" / "data" / f"lead_{args.lead}" / "mats_10s"

    lead_dir = base / "manifest" / "data" / f"lead_{args.lead}"
    seg_dir = lead_dir / "segmented_5s"
    man_dir = lead_dir / "manifests"

    meta_path = Path(args.meta) if args.meta else split_data / "meta_split.csv"

    seg_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"Manifest: Build splits and manifests (Lead {args.lead} = {lead_name})")
    log("=" * 70)
    log(f"Base:          {base}")
    log(f"Meta split:    {meta_path}")
    log(f"Labels final:  {labels_final}")
    log(f"mats_10s:      {mats_10s}")
    log(f"Output:        {lead_dir}")
    log("=" * 70)

    if not meta_path.exists():
        log(f"ERROR: meta split not found: {meta_path}")
        return 1
    if not labels_final.exists():
        log(f"ERROR: labels_final not found: {labels_final}")
        return 1
    if not mats_10s.exists():
        log(f"ERROR: mats_10s not found: {mats_10s}")
        log("Run preprocess script first.")
        return 1

    meta = pd.read_csv(meta_path, low_memory=False)
    meta["base_name"] = meta["save_file"].str.replace(".mat", "", regex=False)
    meta["study_id"] = meta["base_name"].str.split("_").str[-1].astype(int)

    get_label_row, valid_ids, label_cols, label_def_path = build_label_accessor(labels_final)
    n_labels = len(label_cols)
    log(f"Labels: {n_labels} columns, {len(valid_ids)} study_ids with labels")

    available_mats = set(f.name for f in mats_10s.glob("*.mat"))
    meta_available = meta[meta["save_file"].isin(available_mats)].copy()
    log(f"Meta rows: {len(meta)}, available .mat files: {len(available_mats)}\n")

    all_entries: list[tuple[str, str, int]] = []
    skipped_bad_mat: list[str] = []
    n_skipped_existing = 0

    for split in ["train", "valid", "test"]:
        split_meta = meta_available[meta_available["split"] == split]
        for _, row in tqdm(
            split_meta.iterrows(), total=len(split_meta),
            desc=f"Lead {args.lead} {split}",
        ):
            study_id = int(row["study_id"])
            if study_id not in valid_ids:
                continue
            base_name = row["base_name"]
            expected = expected_5s_paths(seg_dir, base_name)
            
            if all(p.exists() for p in expected):
                n_skipped_existing += 1
                for p in expected:
                    all_entries.append((split, str(p.relative_to(seg_dir)), study_id))
                continue
            
            src = mats_10s / row["save_file"]
            try:
                out_paths = segment_10s_to_5s(src, seg_dir, base_name, study_id)
            except (MatReadError, OSError, ValueError) as e:
                skipped_bad_mat.append(f"{row['save_file']}: {e}")
                continue
                
            for p in out_paths:
                all_entries.append((split, str(p.relative_to(seg_dir)), study_id))

    if n_skipped_existing:
        log(f"\nSkipped re-segmenting {n_skipped_existing} 10s files (5s segments already present).")

    if skipped_bad_mat:
        log(f"\nSkipped {len(skipped_bad_mat)} bad/truncated .mat files.")
        skip_log = lead_dir / "skipped_bad_mat.txt"
        with open(skip_log, "w") as f:
            f.write("\n".join(skipped_bad_mat))

    N = len(all_entries)
    log(f"\nTotal 5s segments: {N}")

    y = np.zeros((N, n_labels), dtype=np.float32)
    for i, (_, _, study_id) in enumerate(all_entries):
        y[i] = get_label_row(study_id)
    np.save(man_dir / "y.npy", y)

    for i, (_, rel_path, _) in enumerate(tqdm(all_entries, desc="Write manifest idx into .mat")):
        mat_path = seg_dir / rel_path
        d = loadmat(str(mat_path))
        d["idx"] = np.array([[i]], dtype=np.int64)
        savemat(str(mat_path), d, format="5", do_compression=False)

    root_line = str(seg_dir.resolve())
    for split in ["train", "valid", "test"]:
        entries = [rel for s, rel, _ in all_entries if s == split]
        with open(man_dir / f"{split}.tsv", "w") as f:
            f.write(root_line + "\n")
            for rel in entries:
                f.write(rel + "\t2500\n")

    original_idx_order = np.array([e[2] for e in all_entries], dtype=np.int64)
    np.save(man_dir / "original_idx_order.npy", original_idx_order)

    if label_def_path.exists():
        shutil.copy2(str(label_def_path), str(man_dir / "label_def.csv"))
    pw_src = labels_final / "pos_weight.txt"
    if pw_src.exists():
        shutil.copy2(str(pw_src), str(man_dir / "pos_weight.txt"))

    test_sids = sorted(set(sid for s, _, sid in all_entries if s == "test"))
    test_labels = []
    for sid in test_sids:
        try:
            vals = get_label_row(sid)
            test_labels.append({"idx": sid, **dict(zip(label_cols, vals))})
        except (KeyError, IndexError):
            pass
    if test_labels:
        pd.DataFrame(test_labels).to_csv(lead_dir / "test_labels.csv", index=False)

    test_meta = meta_available[meta_available["split"] == "test"]
    test_file_list = []
    for _, row in test_meta.iterrows():
        if int(row["study_id"]) in valid_ids:
            test_file_list.append({
                "idx": len(test_file_list),
                "study_id": int(row["study_id"]),
                "save_file": row["save_file"],
            })
    if test_file_list:
        pd.DataFrame(test_file_list).to_csv(lead_dir / "test_file_list.csv", index=False)

    sid_mapping = (
        meta_available[["study_id", "save_file"]]
        .drop_duplicates()
        .sort_values("study_id")
    )
    sid_mapping.to_csv(lead_dir / "study_id_mapping.csv", index=False)

    log("\n--- Split Summary ---")
    for split in ["train", "valid", "test"]:
        seg_count = sum(1 for s, _, _ in all_entries if s == split)
        study_count = len(set(sid for s, _, sid in all_entries if s == split))
        log(f"  {split:5s}: {seg_count:>8d} segments, {study_count:>7d} studies")
    
    log(f"\nOutput complete: {lead_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
