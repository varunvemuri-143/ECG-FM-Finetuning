# Label-creation scripts

**create_labels_ecgfm.py** — Builds label files using the ECG-FM pattern labeler. Same logic as ECG-FM’s `labeler.ipynb`.

- **Reads:** `labels/raw_for_labeling/machine_measurements.csv` (must have `study_id` and `report_0`..`report_17`), `labels/ecg_fm_labeler_config/` (JSONs + label_def.csv), and the Python package in `labels/ecg_fm_labeler/`. Optionally uses `split/data/meta_split.csv` to filter by split.
- **Writes:** `labels/computed_labels/` — labels.csv, y.npy, y_soft.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv.

Run from repo root:  
`python labels/scripts/create_labels_ecgfm.py --base-dir .`  
Optional: `--split test --max-rows 10000` to limit to a split and/or row count.

All paths are relative to `--base-dir` (or env `ECG_FINETUNE_BASE`). No HiperGator or external paths.
