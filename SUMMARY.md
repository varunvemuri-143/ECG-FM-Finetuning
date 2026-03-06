# Project summary: ECG-FM multi-lead fine-tuning pipeline

This document summarizes the pipeline, what each stage does, and where the **final files and outputs** live. For a **step-by-step reproduction guide** (exact commands and checks), see **[README.md → How to reproduce (step-by-step)](README.md#how-to-reproduce-step-by-step)**.

---

## What this repo is

A self-contained pipeline to:

1. **Create labels** from MIMIC-IV-ECG text reports (ECG-FM pattern labeler → 17 diagnostic labels).
2. **Split** records into train / valid / test (from record list or provided meta split).
3. **Preprocess** raw WFDB into 10 s Lead-1 or Lead-2 .mat files.
4. **Build manifests** for fairseq (train/valid/test TSVs, y.npy, label_def, pos_weight).
5. **Fine-tune** ECG-FM with fairseq-signals (external step).
6. **Evaluate** a checkpoint on the test set.

All paths are relative to the repo root (`--base-dir` or `ECG_FINETUNE_BASE`). No hardcoded cluster or machine paths.

---

## Pipeline stages (in order)

| Step | Stage      | Script / action | Main output location |
|------|------------|-----------------|----------------------|
| 0    | **Labels** | `labels/scripts/create_labels.py` | `labels/computed_labels/` |
| 1    | **Split**  | `split/scripts/create_split.py` or use provided meta split | `split/data/meta_split.csv` |
| 2    | **Preprocess** | `preprocess/scripts/preprocess.py` | `preprocess/data/lead_{lead}/` |
| 3    | **Manifest** | `manifest/scripts/build_test_and_finetune_data.py` | `manifest/data/` |
| 4    | **Fine-tune** | fairseq-hydra-train (in fairseq-signals) | Your checkpoint directory |
| 5    | **Eval**   | `eval/scripts/evaluate.py` | `eval/data/` |

---

## Final files and outputs (by stage)

### Labels (stage 0)

**Output directory:** `labels/computed_labels/`

| File | Description |
|------|-------------|
| `labels.csv` | Binary 0/1 labels, one row per sample, columns = 17 label names |
| `y.npy` | Label matrix (N × 17), float32 |
| `label_def_recomputed.csv` | Label names, counts, pos_weight (same order as labels.csv) |
| `label_def_official_17.csv` | Copy of official 17-label definition |
| `pos_weight.txt` | One weight per label, space-separated (for loss weighting) |
| `study_id_mapping.csv` | idx, study_id, split |

**Inputs used:** `labels/label_inputs/machine_measurements.csv`, `labels/ecg_fm_labeler_config/`, `labels/ecg_fm_labeler/`, optional `split/data/meta_split.csv`.

---

### Split (stage 1)

**Output (or provided):** `split/data/meta_split.csv`

| Column | Description |
|--------|-------------|
| `save_file` | e.g. mimic_iv_ecg_&lt;id&gt;.mat |
| `split` | train | valid | test |

**Alternative:** Use provided `split/data/meta_split_mimic_iv_ecg.csv` (same split as used in the reference run). Pass `--meta split/data/meta_split_mimic_iv_ecg.csv` to preprocess and manifest, or copy to `meta_split.csv`.

**Input:** `split/data/record_list.csv` (study_id, path) if generating split with `create_split.py`.

---

### Preprocess (stage 2)

**Output directory:** `preprocess/data/lead_{lead}/`

| Content | Description |
|---------|-------------|
| `*.mat` (one per record in `mats_10s`) | 10 s, 12 ch (Lead I or II duplicated), 500 Hz, standardized; `feats`, `curr_sample_rate` |

**Inputs:** `split/data/meta_split.csv`, `split/data/record_list.csv`, raw MIMIC-IV-ECG root (`--raw-root`), and `--lead`.

---

### Manifest (stage 3)

**Output directories:** `manifest/data/lead_{lead}/`

**Test set (for evaluation):**

| File | Description |
|------|-------------|
| `segmented_5s/` | Segmented .mat files for all splits |
| `test_file_list.csv` | idx, study_id, save_file |
| `test_labels.csv` | Labels for test set |
| `study_id_mapping.csv` | Copied from labels |

**Finetune set (for fairseq):**

| Path | Description |
|------|-------------|
| `manifests/` | **Main fairseq inputs:** |
| `manifests/train.tsv`, `valid.tsv`, `test.tsv` | Root path + list of rel_path &lt;tab&gt; 2500 |
| `manifests/y.npy` | Label matrix aligned to manifest rows |
| `manifests/label_def.csv` | Label names (same order as y.npy) |
| `manifests/pos_weight.txt` | Space-separated weights |
| `manifests/original_idx_order.npy` | Original study_id per row |

**Inputs:** `labels/computed_labels/`, `split/data/meta_split.csv`, `preprocess/data/lead_{lead}/`. 

---

### Fine-tuning (stage 4)

**Output:** Checkpoint(s) in a directory you choose (e.g. `finetuned_model/`). With `checkpoint.best_checkpoint_metric=loss` (default in README), the best checkpoint is typically `checkpoint_best.pt`; with `auroc` it is `checkpoint_best_auroc_*.pt` (or similar).

**Inputs:** `manifest/data/lead_{lead}/manifests/` as `task.data`, pretrained .pt as `model.model_path`.

---

### Eval (stage 5)

**Output directory:** `eval/data/`

| Content | Description |
|---------|-------------|
| Predictions CSV | Per-sample probabilities (per study_id) |
| Metrics CSV | Per-label + __macro__ (mean over labels) + __micro__ (pooled) AUROC, AUPRC, accuracy, recall, precision, F1, specificity, NPV |

**Inputs:** `manifest/data/lead_{lead}/`, `labels/computed_labels/`, checkpoint path (`--model-path`), and `--lead`.

---

## Inputs you must provide (outside repo)

| What | Where / how |
|------|-------------|
| **machine_measurements.csv** | Place in `labels/label_inputs/` (MIMIC-IV-ECG format: study_id, report_0..report_17) |
| **record_list.csv** | In repo at `split/data/record_list.csv` (or provide via --record-list) |
| **Meta split** | Generate with `create_split.py` → `meta_split.csv`, or use `meta_split_mimic_iv_ecg.csv` |
| **Raw MIMIC-IV-ECG** | Download from PhysioNet; pass path as `--raw-root` to preprocess |
| **Pretrained ECG-FM .pt** | Download from HuggingFace; pass to fairseq-hydra-train as `model.model_path` |

---

## Quick reference: where is the final …?

| Final output | Location |
|--------------|---------|
| 17-label definitions | `labels/computed_labels/label_def_recomputed.csv`, `pos_weight.txt` |
| Train/valid/test labels for fairseq | `manifest/data/lead_{lead}/manifests/y.npy` (+ label_def.csv, pos_weight.txt) |
| Fairseq data path | `manifest/data/lead_{lead}/manifests/` |
| Eval predictions and metrics | `eval/data/` |
| Fine-tuned checkpoint | Your choice; e.g. `finetuned_model/checkpoint_best.pt` |

---

## Other docs in this repo

- **[README.md](README.md)** — Setup, pipeline table, fairseq command, eval commands.
- **[SCRIPTS.md](SCRIPTS.md)** — Per-script purpose, inputs, outputs.
- **[FILE_LOCATIONS.md](FILE_LOCATIONS.md)** — Where each file lives and who writes it.
