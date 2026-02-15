# Scripts and data by stage

All scripts use **`--base-dir`** (or env **`ECG_FINETUNE_BASE`**) as the repo root. Run from the **repository root**. See **FILE_LOCATIONS.md** for what to put where manually.

---

## labels/

- **labels/raw_for_labeling/** — Raw inputs: put **machine_measurements.csv** here (MIMIC-IV-ECG, with `study_id` and `report_0`..`report_17`). See labels/raw_for_labeling/README.md.
- **labels/ecg_fm_labeler_config/** — ECG-FM pattern labeler config (JSONs + official **label_def.csv**). Bundled so the label step is self-contained; do not edit unless syncing with ECG-FM.
- **labels/ecg_fm_labeler/** — ECG-FM pattern labeler Python package (pattern_labeler, preprocess, etc.). Bundled so the label step runs without an external ECG-FM clone.
- **labels/scripts/** — Script that creates labels: **create_labels_ecgfm.py** (see below).
- **labels/computed_labels/** — Output of the label script: labels.csv, y.npy, y_soft.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv. Build script reads from here.

### labels/scripts/create_labels_ecgfm.py

**Purpose:** Build label files using the ECG-FM pattern labeler (same logic as ECG-FM’s labeler.ipynb). Reads machine_measurements, builds text from report_0..report_17, runs the labeler, and writes the 17 diagnostic labels and arrays.

**Inputs:**
- **labels/raw_for_labeling/machine_measurements.csv** (required)
- **labels/ecg_fm_labeler_config/** (JSONs + label_def.csv)
- **labels/ecg_fm_labeler/** (Python package)
- **split/data/meta_split.csv** (optional; if present, used to filter by `--split` and to attach split in study_id_mapping)

**Outputs:** **labels/computed_labels/** — labels.csv, y.npy, y_soft.npy, label_def_recomputed.csv, label_def_official_17.csv, pos_weight.txt, study_id_mapping.csv.

**Example:**  
`python labels/scripts/create_labels_ecgfm.py --base-dir .`  
`python labels/scripts/create_labels_ecgfm.py --base-dir . --split test --max-rows 10000`

---

## split/

### split/scripts/create_split.py

**Purpose:** Create train/valid/test split and write meta CSV.

**Inputs:** **split/data/record_list.csv** (columns: `study_id`, `path`).

**Outputs:** **split/data/meta_split.csv** (columns: `save_file`, `split`).

**Example:** `python split/scripts/create_split.py --base-dir .`

---

## preprocess/

### preprocess/scripts/preprocess_ecgfm.py

**Purpose:** WFDB → 10 s .mat (Lead I duplicated to 12 ch, 500 Hz, standardized). Raw data is not in the repo; pass MIMIC-IV-ECG root via `--raw-root`.

**Inputs:** split/data/meta_split.csv, split/data/record_list.csv, **--raw-root** (path to downloaded MIMIC-IV-ECG).

**Outputs:** **preprocess/data/lead_1_duplicated/** (one .mat per record).

**Example:** `python preprocess/scripts/preprocess_ecgfm.py --base-dir . --raw-root /path/to/mimic-iv-ecg`

---

## build/

### build/scripts/build_test_and_finetune_data.py

**Purpose:** Build test set and finetune set (5 s segments, manifests, y.npy, etc.) for fairseq.

**Inputs:** labels/computed_labels/, split/data/meta_split.csv, preprocess/data/lead_1_duplicated/.

**Outputs:** build/data/test_lead1_duplicated/, build/data/finetune_lead1_duplicated/.

**Example:** `python build/scripts/build_test_and_finetune_data.py --base-dir .`

---

## Fine-tuning (fairseq_signals)

Run inside the fairseq-signals clone. Use **build/data/finetune_lead1_duplicated/manifests/** as `task.data`. See README for the full `fairseq-hydra-train` command.

---

## eval/

### eval/scripts/eval_ecgfm.py

**Purpose:** Run ECG-FM inference on test set and compute per-label metrics.

**Inputs:** build/data/test_lead1_duplicated/, labels/computed_labels/, **--model-path** (checkpoint .pt). You can put checkpoints in **finetuned_model/** and pass e.g. `--model-path finetuned_model/checkpoint_best_loss_*.pt`.

**Outputs:** eval/data/ (CSV predictions and metrics).

**Example:** `python eval/scripts/eval_ecgfm.py --base-dir . --model-path /path/to/checkpoint.pt`

### eval/scripts/eval_finetuned.py

**Purpose:** Same as eval_ecgfm.py with a torch.load patch for finetuned checkpoints (PyTorch ≥ 2.6).

**Inputs / Outputs:** Same as eval_ecgfm.py.

**Example:** `python eval/scripts/eval_finetuned.py --base-dir . --model-path /path/to/checkpoint.pt`
