#!/usr/bin/env python3
"""
Build test set and finetune set for Lead I duplicated only.

Reads: labels/labels/, split/data/meta_split.csv, preprocess/data/lead_1_duplicated/.
Writes: build/data/test_lead1_duplicated/, build/data/finetune_lead1_duplicated/.
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
from tqdm import tqdm

SEGMENT_SAMPLES = 2500
DIR_LEAD1 = "lead_1_duplicated"


def get_base_dir(args: argparse.Namespace) -> Path:
    return Path(args.base_dir or os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


def log(msg: str) -> None:
    print(msg, flush=True)


def run_step1_test_data(
    preprocess_lead1_dir: Path,
    build_data_dir: Path,
    meta: pd.DataFrame,
    get_label_row: Callable[[int], np.ndarray],
    valid_study_ids: set,
    label_cols: list,
    label_def_path: Path,
) -> None:
    out_parent = build_data_dir / "test_lead1_duplicated"
    mats_dir = out_parent / "mats"
    mats_dir.mkdir(parents=True, exist_ok=True)

    test_df = meta[meta["split"] == "test"].copy()
    test_df["study_id"] = test_df["save_file"].str.replace(".mat", "", regex=False).str.split("_").str[-1].astype(int)
    test_save_files = test_df["save_file"].unique().tolist()

    for sf in test_save_files:
        src = preprocess_lead1_dir / sf
        if not src.exists():
            continue
        dest = mats_dir / sf
        if not dest.exists():
            try:
                dest.symlink_to(os.path.relpath(str(src.resolve()), str(mats_dir.resolve())))
            except OSError:
                shutil.copy2(str(src), str(dest))

    test_list = []
    for _, row in test_df.iterrows():
        if (mats_dir / row["save_file"]).exists():
            test_list.append({"idx": len(test_list), "study_id": row["study_id"], "save_file": row["save_file"]})
    if test_list:
        pd.DataFrame(test_list).to_csv(out_parent / "test_file_list.csv", index=False)

    sid_set = set(test_df["study_id"].astype(int)) & valid_study_ids
    rows = []
    for s in sorted(sid_set):
        try:
            row_vals = get_label_row(s)
            rows.append({"idx": s, **dict(zip(label_cols, row_vals))})
        except (KeyError, IndexError):
            pass
    if rows:
        pd.DataFrame(rows).to_csv(out_parent / "test_labels.csv", index=False)
    if label_def_path.exists():
        shutil.copy2(str(label_def_path), str(out_parent / "label_def.csv"))
    test_df[["study_id", "save_file"]].drop_duplicates().to_csv(out_parent / "study_id_mapping.csv", index=False)
    log(f"Step 1: test_lead1_duplicated -> {out_parent} (mats={len(list(mats_dir.glob('*.mat')))})")


def segment_10s_to_5s(src_mat: Path, out_dir: Path, base_name: str, study_id: int) -> list:
    data = loadmat(str(src_mat), variable_names=["feats", "curr_sample_rate"])
    feats = data["feats"]
    rate = int(np.squeeze(data["curr_sample_rate"]))
    if rate != 500:
        raise ValueError(f"Expected 500 Hz, got {rate}")
    n_seg = feats.shape[-1] // SEGMENT_SAMPLES
    out_paths = []
    for seg_i in range(n_seg):
        seg_feats = feats[:, seg_i * SEGMENT_SAMPLES : (seg_i + 1) * SEGMENT_SAMPLES].copy()
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


def run_step2_finetune_data(
    preprocess_lead1_dir: Path,
    build_data_dir: Path,
    labels_dir: Path,
    meta: pd.DataFrame,
    get_label_row: Callable[[int], np.ndarray],
    valid_study_ids: set,
    label_cols: list,
    label_def_path: Path,
) -> None:
    out_root = build_data_dir / "finetune_lead1_duplicated"
    segmented_dir = out_root / "segmented_5s"
    manifests_dir = out_root / "manifests"
    segmented_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    n_labels = len(label_cols)

    meta = meta.copy()
    meta["base_name"] = meta["save_file"].str.replace(".mat", "", regex=False)
    meta["study_id"] = meta["base_name"].str.split("_").str[-1].astype(int)

    all_entries = []
    for split in ["train", "valid", "test"]:
        split_meta = meta[meta["split"] == split]
        for _, row in tqdm(split_meta.iterrows(), total=len(split_meta), desc=f"finetune_lead1 {split}", leave=False):
            src = preprocess_lead1_dir / row["save_file"]
            if not src.exists():
                continue
            study_id = int(row["study_id"])
            if study_id not in valid_study_ids:
                continue
            out_paths = segment_10s_to_5s(src, segmented_dir, row["base_name"], study_id)
            for p in out_paths:
                all_entries.append((split, str(p.relative_to(segmented_dir)), study_id))

    N = len(all_entries)
    y = np.zeros((N, n_labels), dtype=np.float32)
    for i, (_, _, study_id) in enumerate(all_entries):
        y[i] = get_label_row(study_id)
    np.save(manifests_dir / "y.npy", y)

    for i, (_, rel_path, _) in enumerate(tqdm(all_entries, desc="write idx", leave=False)):
        mat_path = segmented_dir / rel_path
        d = loadmat(str(mat_path))
        d["idx"] = np.array([[i]], dtype=np.int64)
        savemat(str(mat_path), d, format="5", do_compression=False)

    root = str(segmented_dir.resolve())
    for split in ["train", "valid", "test"]:
        entries = [rel for s, rel, _ in all_entries if s == split]
        with open(manifests_dir / f"{split}.tsv", "w") as f:
            f.write(root + "\n")
            for rel in entries:
                f.write(rel + "\t2500\n")
    np.save(manifests_dir / "original_idx_order.npy", np.array([e[2] for e in all_entries], dtype=np.int64))
    if label_def_path.exists():
        shutil.copy2(str(label_def_path), str(manifests_dir / "label_def.csv"))
    pos_weight_src = labels_dir / "pos_weight.txt"
    if pos_weight_src.exists():
        shutil.copy2(str(pos_weight_src), str(manifests_dir / "pos_weight.txt"))
    log(f"Step 2: finetune_lead1_duplicated -> {out_root} (segments={N})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build test + finetune data (Lead I duplicated only).")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--skip-step1", action="store_true")
    parser.add_argument("--skip-step2", action="store_true")
    args = parser.parse_args()

    base = get_base_dir(args)
    split_data = base / "split" / "data"
    preprocess_data = base / "preprocess" / "data"
    build_data = base / "build" / "data"
    labels_dir = base / "labels" / "computed_labels"
    meta_path = Path(args.meta) if args.meta else split_data / "meta_split.csv"
    preprocess_lead1_dir = preprocess_data / DIR_LEAD1

    if not meta_path.exists():
        log(f"ERROR: meta not found: {meta_path}")
        return 1
    if not labels_dir.exists():
        log(f"ERROR: labels dir not found: {labels_dir}")
        return 1
    if not preprocess_lead1_dir.exists():
        log(f"ERROR: preprocess output not found: {preprocess_lead1_dir}. Run preprocess script first.")
        return 1

    labels_csv = labels_dir / "labels.csv"
    if not labels_csv.exists():
        log(f"ERROR: {labels_csv} not found")
        return 1
    label_def_path = labels_dir / "label_def_recomputed.csv"
    if not label_def_path.exists():
        label_def_path = labels_dir / "label_def.csv"
    if not label_def_path.exists():
        log("ERROR: label_def_recomputed.csv or label_def.csv required in labels/computed_labels/")
        return 1

    meta = pd.read_csv(meta_path, low_memory=False)
    if "split" not in meta.columns or "save_file" not in meta.columns:
        log("ERROR: meta must have columns split, save_file")
        return 1
    label_cols = pd.read_csv(label_def_path)["name"].tolist()
    labels_df = pd.read_csv(labels_csv)
    if not all(c in labels_df.columns for c in label_cols):
        log("ERROR: labels.csv missing label_def columns")
        return 1
    labels_by_idx = labels_df.set_index("idx")[label_cols]
    mapping_path = labels_dir / "study_id_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        study_id_to_idx = mapping.set_index("study_id")["idx"].to_dict()
        valid_study_ids = set(int(s) for s in study_id_to_idx.keys())

        def get_label_row(study_id: int):
            return labels_by_idx.loc[study_id_to_idx[study_id]].values.astype(np.float32)
    else:
        valid_study_ids = set(labels_df["idx"].astype(int))

        def get_label_row(study_id: int):
            return labels_by_idx.loc[study_id].values.astype(np.float32)

    build_data.mkdir(parents=True, exist_ok=True)
    if not args.skip_step1:
        run_step1_test_data(preprocess_lead1_dir, build_data, meta, get_label_row, valid_study_ids, label_cols, label_def_path)
    if not args.skip_step2:
        run_step2_finetune_data(
            preprocess_lead1_dir, build_data, labels_dir, meta, get_label_row, valid_study_ids, label_cols, label_def_path
        )
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
