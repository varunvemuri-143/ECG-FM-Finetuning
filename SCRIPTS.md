# Scripts and data by stage

Scripts use **`--base-dir`** (or env **`ECG_FINETUNE_BASE`**) as the repository root. [FILE_LOCATIONS.md](FILE_LOCATIONS.md) summarizes where files live.

---

## labels/

- **labels/label_inputs/** — Inputs for the label script (e.g. machine_measurements.csv, MIMIC-IV-ECG format).
- **labels/ecg_fm_labeler_config/** — ECG-FM pattern labeler config (JSONs + label_def.csv).
- **labels/ecg_fm_labeler/** — ECG-FM pattern labeler Python package.
- **labels/scripts/create_labels_ecgfm.py** — Builds label files; reads label_inputs + config; writes to computed_labels.
- **labels/computed_labels/** — Output: labels.csv, y.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv. Consumed by build.

### create_labels_ecgfm.py

**Purpose:** Build label files using the ECG-FM pattern labeler. Reads machine_measurements, builds text from report_0..report_17, runs the labeler; writes 17 diagnostic labels and arrays.

**Inputs:** labels/label_inputs/machine_measurements.csv, labels/ecg_fm_labeler_config/, labels/ecg_fm_labeler/. Optional: split/data/meta_split.csv (for --split filter).

**Outputs:** labels/computed_labels/ (labels.csv, y.npy, label_def_recomputed.csv, label_def_official_17.csv, pos_weight.txt, study_id_mapping.csv).

---

## split/

### create_split.py

**Purpose:** Create train/valid/test split; write meta CSV.

**Inputs:** split/data/record_list.csv (columns: study_id, path).

**Outputs:** split/data/meta_split.csv (columns: save_file, split).

---

## preprocess/

### preprocess_ecgfm.py

**Purpose:** WFDB → 10 s .mat (Lead I duplicated to 12 ch, 500 Hz). Requires MIMIC-IV-ECG root via --raw-root.

**Inputs:** split/data/meta_split.csv, split/data/record_list.csv, --raw-root.

**Outputs:** preprocess/data/lead_1_duplicated/ (one .mat per record).

---

## build/

### build_test_and_finetune_data.py

**Purpose:** Build test set and finetune set (5 s segments, manifests, y.npy) for fairseq.

**Inputs:** labels/computed_labels/, split/data/meta_split.csv, preprocess/data/lead_1_duplicated/.

**Outputs:** build/data/test_lead1_duplicated/, build/data/finetune_lead1_duplicated/.

---

## Fine-tuning

Done inside the fairseq-signals clone. task.data points to build/data/finetune_lead1_duplicated/manifests/. See README for fairseq-hydra-train command.

---

## eval/

### eval_ecgfm.py

**Purpose:** Run ECG-FM inference on test set; compute per-label metrics.

**Inputs:** build/data/test_lead1_duplicated/, labels/computed_labels/, --model-path (checkpoint .pt).

**Outputs:** eval/data/ (CSV predictions and metrics).

### eval_finetuned.py

**Purpose:** Same as eval_ecgfm.py with a torch.load patch for finetuned checkpoints (PyTorch ≥ 2.6).

**Inputs / Outputs:** Same as eval_ecgfm.py.
