# What to put where (manual)

Use this as a checklist. Anything not in the repo you must add or download yourself.

---

## 1. Labels (from your label-creation pipeline)

| What | Where | Notes |
|------|--------|------|
| **Raw inputs for labeler** | **`labels/raw_for_labeling/`** | Put **`machine_measurements.csv`** here (MIMIC-IV-ECG; must have `study_id` and `report_0`..`report_17`). |
| **Labeler config (JSON + label_def)** | **`labels/ecg_fm_labeler_config/`** | Already in repo (from ECG-FM). Do not edit unless syncing with ECG-FM. |
| **Labeler Python package** | **`labels/ecg_fm_labeler/`** | Already in repo (from ECG-FM). Do not edit unless syncing with ECG-FM. |
| **Label creator script** | **`labels/scripts/create_labels_ecgfm.py`** | Run this to produce computed_labels; uses `raw_for_labeling/`, `ecg_fm_labeler_config/`, `ecg_fm_labeler/`. |
| **Computed label files** | **`labels/computed_labels/`** | Written by `create_labels_ecgfm.py`: **labels.csv**, **y.npy**, **y_soft.npy**, **label_def_recomputed.csv**, **pos_weight.txt**, **study_id_mapping.csv**. Example-format files may already exist; re-run the script to regenerate. |

---

## 2. Split

| What | Where | Notes |
|------|--------|------|
| **Record list** | **`split/data/record_list.csv`** | Columns: `study_id`, `path` (path relative to raw WFDB root). You can copy from e.g. `ECG-Lead1-Finetuned/extra/record_list.csv` or generate from MIMIC-IV-ECG manifest. |
| **Meta split** | **`split/data/meta_split.csv`** | Written by `split/scripts/create_split.py`; no need to add manually unless you have your own. |

---

## 3. Preprocess

| What | Where | Notes |
|------|--------|------|
| **Raw WFDB root** | **External** | Download [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) and pass its path as `--raw-root` to the preprocess script. Not stored in the repo. |
| **Preprocessed .mat** | **`preprocess/data/lead_1_duplicated/`** | Written by `preprocess/scripts/preprocess_ecgfm.py`; no need to add manually. |

---

## 4. Build

| What | Where | Notes |
|------|--------|------|
| **Test / finetune sets** | **`build/data/test_lead1_duplicated/`**, **`build/data/finetune_lead1_duplicated/`** | Written by `build/scripts/build_test_and_finetune_data.py`; no need to add manually. |

---

## 5. Fine-tuning (fairseq)

| What | Where | Notes |
|------|--------|------|
| **Pretrained .pt** | **External** | Download [mimic_iv_ecg_physionet_pretrained.pt](https://huggingface.co/wanglab/ecg-fm). Pass its path to `fairseq-hydra-train` as `model.model_path`. |
| **Fine-tuned .pt** | **`finetuned_model/`** (optional) | After fine-tuning, copy your best checkpoint (e.g. `checkpoint_best_loss_*.pt`) here so eval can use `--model-path finetuned_model/checkpoint_best_loss_xxx.pt`. |

---

## 6. Eval

| What | Where | Notes |
|------|--------|------|
| **Predictions / metrics** | **`eval/data/`** | Written by `eval/scripts/eval_ecgfm.py` or `eval_finetuned.py`; no need to add manually. |

---

## Summary: you must add or do manually

1. **labels/raw_for_labeling/** — Add **`machine_measurements.csv`** (MIMIC-IV-ECG with `study_id` and `report_0`..`report_17`).
2. **labels/scripts/create_labels_ecgfm.py** — Already in repo. Run it (e.g. `python labels/scripts/create_labels_ecgfm.py --base-dir .`) to create labels.
3. **labels/computed_labels/** — Filled by the script. If you already have example files here, re-run the script to regenerate **labels.csv**, **y.npy**, **y_soft.npy**, etc.
4. **split/data/** — Add **`record_list.csv`** (e.g. copy from `ECG-Lead1-Finetuned/extra/record_list.csv`).
5. **Raw MIMIC-IV-ECG** — Download from PhysioNet and pass `--raw-root` to preprocess.
6. **Pretrained model** — Download from HuggingFace for fine-tuning.
7. **finetuned_model/** — Optionally copy your best checkpoint here after fine-tuning (e.g. `checkpoint_best_loss_*.pt`). Then run eval with `--model-path finetuned_model/checkpoint_best_loss_xxx.pt`.

Nothing else needs to be placed manually; the rest is produced by the pipeline scripts.

---

## If you missed anything

- **.gitignore** — Consider adding `finetuned_model/*.pt`, `labels/computed_labels/labels.csv`, `labels/computed_labels/*.npy`, `split/data/record_list.csv`, and large data dirs if you do not want to commit them.
- **Pretrained model** — Keep the downloaded HuggingFace `.pt` somewhere on disk and pass that path to `fairseq-hydra-train`; it does not need to live inside this repo.
- **Raw MIMIC-IV-ECG** — Stays wherever you downloaded it; only `--raw-root` is passed to the preprocess script.
