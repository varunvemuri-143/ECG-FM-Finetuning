#!/usr/bin/env python3
"""
create_labels_ecgfm.py — Label creator (ECG-FM pattern labeler).

Generates labels using the same logic as ECG-FM's labeler.ipynb:
  preprocess_texts → PatternLabelerConfig.from_json → PatternLabeler → labels_to_array.

Reads MIMIC machine_measurements.csv (report_0..report_17), runs the ECG-FM pattern
labeler (config and code live under labels/ecg_fm_labeler_config/ and labels/ecg_fm_labeler/),
and writes to labels/computed_labels/:
  - labels.csv (binary 0/1, N × 17)
  - y.npy
  - label_def_recomputed.csv, label_def_official_17.csv
  - pos_weight.txt
  - study_id_mapping.csv

All paths are relative to --base-dir (repo root). No HiperGator or external paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# pandas<2.x compatibility: ECG-FM uses DataFrame.map()
if not hasattr(pd.DataFrame, "map"):
    pd.DataFrame.map = pd.DataFrame.applymap  # type: ignore[attr-defined]


def _repo_paths(base_dir: Path) -> dict:
    """Paths relative to repo base_dir (labels/scripts -> base_dir = repo root)."""
    return {
        "labeler_package": base_dir / "labels" / "ecg_fm_labeler",
        "labeler_config_dir": base_dir / "labels" / "ecg_fm_labeler_config",
        "official_label_def": base_dir / "labels" / "ecg_fm_labeler_config" / "label_def.csv",
        "machine_meas": base_dir / "labels" / "label_inputs" / "machine_measurements.csv",
        "split_csv": base_dir / "split" / "data" / "meta_split.csv",
        "out_dir": base_dir / "labels" / "computed_labels",
    }


def build_text_column(df: pd.DataFrame, report_prefix: str = "report_") -> pd.Series:
    """Concatenate report_0..report_17 into a single free-text string per row."""
    report_cols = [c for c in df.columns if c.startswith(report_prefix)]
    if not report_cols:
        raise ValueError(f"No columns starting with '{report_prefix}' found.")

    report_cols = sorted(
        report_cols,
        key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 9999,
    )

    rows: list[str] = []
    for _, r in df[report_cols].iterrows():
        parts: list[str] = []
        for val in r:
            if isinstance(val, str):
                s = val.strip()
            elif pd.isna(val):
                continue
            else:
                s = str(val).strip()
            if s:
                parts.append(s)
        rows.append(" ".join(parts))

    return pd.Series(rows, index=df.index, dtype="string")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create labels using ECG-FM pattern labeler.")
    ap.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Repo root (default: parent of labels/ relative to this script).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Max rows to label after split filter (default: 0 = all).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "test", "all"],
        help="Which split to label when split CSV exists (default: all).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: base_dir/labels/computed_labels).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve base_dir: env ECG_FINETUNE_BASE, then --base-dir, then script-relative
    base_dir = os.environ.get("ECG_FINETUNE_BASE")
    if base_dir is None and args.base_dir is not None:
        base_dir = args.base_dir
    if base_dir is None:
        # This script lives at labels/scripts/create_labels_ecgfm.py
        base_dir = Path(__file__).resolve().parent.parent.parent
    base_dir = Path(base_dir)

    paths = _repo_paths(base_dir)
    labeler_package = paths["labeler_package"]
    labeler_config_dir = paths["labeler_config_dir"]
    official_label_def_path = paths["official_label_def"]
    machine_meas_path = paths["machine_meas"]
    split_path = paths["split_csv"]
    out_dir = Path(args.out_dir) if args.out_dir else paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ECG-FM label generation (create_labels_ecgfm)")
    print("=" * 80)
    print(f"base_dir:              {base_dir}")
    print(f"machine_measurements:  {machine_meas_path}")
    print(f"labeler config dir:   {labeler_config_dir}")
    print(f"labeler package:      {labeler_package}")
    print(f"split file:           {split_path}")
    print(f"max_rows:             {args.max_rows or 'all'}")
    print(f"split:                {args.split}")
    print(f"out_dir:              {out_dir}")
    print("=" * 80)

    if not labeler_package.is_dir():
        print(f"✗ Labeler package not found: {labeler_package}")
        return 1
    if not labeler_config_dir.is_dir():
        print(f"✗ Labeler config dir not found: {labeler_config_dir}")
        return 1
    if not official_label_def_path.is_file():
        print(f"✗ Official label_def not found: {official_label_def_path}")
        return 1
    if not machine_meas_path.is_file():
        print(f"✗ machine_measurements.csv not found: {machine_meas_path}")
        return 1

    sys.path.insert(0, str(labeler_package))
    from pattern_labeler import PatternLabelerConfig, PatternLabeler, labels_to_array  # type: ignore
    from preprocess import preprocess_texts  # type: ignore
    import labeler_utils  # type: ignore
    import preprocess as ecgfm_preprocess  # type: ignore

    # Robustness: avoid TypeError when text values are NaN/float in ECG-FM’s remove_after_substring
    def _remove_after_substring_safe(series: pd.Series, substring: str, case: bool = False):
        texts_str = series.fillna("").astype(str)
        if case:
            slc = texts_str.str.find(substring)
        else:
            slc = texts_str.str.lower().str.find(substring.lower())
        not_found_inds = slc == -1
        slc = slc.copy()
        slc.loc[not_found_inds] = texts_str.str.len().loc[not_found_inds]
        slice_int = slc.fillna(texts_str.str.len()).astype(int).to_numpy()
        texts_arr = texts_str.to_numpy()
        out = [t[:i] for t, i in zip(texts_arr, slice_int)]
        return pd.Series(out, index=series.index)

    labeler_utils.remove_after_substring = _remove_after_substring_safe  # type: ignore
    ecgfm_preprocess.remove_after_substring = _remove_after_substring_safe  # type: ignore

    print("\n📖 Loading machine_measurements.csv...")
    df = pd.read_csv(machine_meas_path, low_memory=False)
    print(f"   ✓ Loaded {len(df)} rows")

    if "study_id" not in df.columns:
        raise ValueError("machine_measurements.csv must have a 'study_id' column.")

    print("\n📝 Building texts from report_* ...")
    texts = build_text_column(df, report_prefix="report_").fillna("").astype(str)
    texts.name = "text"
    texts.index = pd.RangeIndex(len(texts), name="level_0")

    if split_path.is_file():
        split_df = pd.read_csv(split_path)
        split_df["study_id"] = (
            split_df["save_file"]
            .str.replace(".mat", "", regex=False)
            .str.split("_")
            .str[-1]
            .astype(int)
        )
        split_df = split_df[["study_id", "split"]]
        df = df.merge(split_df, on="study_id", how="left")
    else:
        df["split"] = "train"
        print(f"   (split file not found; treating all as train)")

    if args.split != "all":
        before = len(df)
        df = df[df["split"] == args.split].copy()
        print(f"\n🔎 Split filter: {args.split} -> {len(df)}/{before} rows")
    else:
        print(f"\n🔎 Split filter: all -> {len(df)} rows")

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()
        print(f"   Using first {len(df)} rows")

    # Rebuild texts for the filtered df (same index alignment as df)
    texts = build_text_column(df, report_prefix="report_").fillna("").astype(str)
    texts.name = "text"
    texts.index = pd.RangeIndex(len(texts), name="level_0")

    print("\n🔍 Loading labeler config (JSON)...")
    labeler_config = PatternLabelerConfig.from_json(str(labeler_config_dir))
    labeler = PatternLabeler(labeler_config)
    print("   ✓ Labeler created")

    print("\n🔄 Preprocessing texts (preprocess_texts)...")
    texts_pp = preprocess_texts(texts.copy())
    texts_pp.rename("text", inplace=True)
    texts_pp.index.name = "level_0"
    print(f"   ✓ Preprocessed {len(texts_pp)} texts")

    print("\n🚀 Running PatternLabeler...")
    res = labeler(texts=texts_pp.copy())
    labels_flat = res.labels_flat.copy()
    print(f"   ✓ labels_flat: {len(labels_flat)} rows")

    official_label_def = pd.read_csv(official_label_def_path)
    target_labels = official_label_def["name"].tolist()
    labels_flat_17 = labels_flat[labels_flat["name"].isin(target_labels)].copy()
    print(f"\n🎯 Filtering to ECG-FM 17 labels: {len(labels_flat_17)}/{len(labels_flat)} rows")

    labels_in_data = set(labels_flat_17["name"].unique())
    labels_required = set(target_labels)
    missing_labels = labels_required - labels_in_data
    if missing_labels:
        print(f"   ⚠️ Official labels not in data (will get zero counts): {missing_labels}")

    print("\n📊 Converting to arrays (labels_to_array)...")
    try:
        labels_df_flat, label_def, y_soft, y = labels_to_array(
            labels_flat_17,
            num_samples=len(texts_pp),
            label_order=target_labels,
        )
    except AssertionError:
        print("   ⚠️ Some official labels missing in data; building 17 columns manually.")
        labels_df_flat, label_def_temp, y_soft_temp, y_temp = labels_to_array(
            labels_flat_17,
            num_samples=len(texts_pp),
        )
        label_name_to_col = {name: idx for idx, name in enumerate(label_def_temp["name"].tolist())}
        label_def_rows = []
        y_cols = []
        y_soft_cols = []
        for label_name in target_labels:
            if label_name in label_name_to_col:
                col_idx = label_name_to_col[label_name]
                label_def_rows.append(label_def_temp.iloc[col_idx].to_dict())
                y_cols.append(y_temp[:, col_idx])
                y_soft_cols.append(y_soft_temp[:, col_idx])
            else:
                label_def_rows.append({
                    "name": label_name,
                    "count": 0,
                    "pos_percent": 0.0,
                    "pos_weight": 0.0,
                })
                y_cols.append(np.zeros(y_temp.shape[0], dtype=y_temp.dtype))
                y_soft_cols.append(np.zeros(y_soft_temp.shape[0], dtype=y_soft_temp.dtype))
        label_def = pd.DataFrame(label_def_rows)
        y = np.column_stack(y_cols)
        y_soft = np.column_stack(y_soft_cols)

    print(f"   ✓ y shape: {y.shape}")

    np.save(out_dir / "y.npy", y.astype("float32"))
    label_def.to_csv(out_dir / "label_def_recomputed.csv", index=False)
    official_label_def.to_csv(out_dir / "label_def_official_17.csv", index=False)
    with open(out_dir / "pos_weight.txt", "w") as f:
        f.write(" ".join(f"{w:.8f}" for w in label_def["pos_weight"].values))

    label_names = label_def["name"].tolist()
    labels_matrix = pd.DataFrame(y.astype(int), columns=label_names, index=texts_pp.index)
    labels_matrix.index.name = "idx"
    labels_matrix.to_csv(out_dir / "labels.csv")

    mapping = pd.DataFrame({
        "idx": np.arange(len(df), dtype=int),
        "study_id": df["study_id"].values,
        "split": df["split"].values,
    })
    mapping.to_csv(out_dir / "study_id_mapping.csv", index=False)

    print("\n✅ Done. Outputs:")
    for name in ["labels.csv", "y.npy", "label_def_recomputed.csv", "pos_weight.txt", "study_id_mapping.csv"]:
        print(f"   {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
