# File locations

Reference for where inputs and outputs live. Data not in the repo (raw ECG, pretrained model) is referenced by path when running scripts.

---

## 1. Labels

| What | Where |
|------|--------|
| Inputs for labeler | labels/label_inputs/ (e.g. machine_measurements.csv, MIMIC-IV-ECG: study_id, report_0..report_17) |
| Labeler config | labels/ecg_fm_labeler_config/ (JSONs + label_def.csv; from ECG-FM) |
| Labeler package | labels/ecg_fm_labeler/ (from ECG-FM) |
| Label script | labels/scripts/create_labels_ecgfm.py |
| Computed labels | labels/computed_labels/ (labels.csv, y.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv; written by create_labels_ecgfm.py) |

---

## 2. Split

| What | Where |
|------|--------|
| Record list | split/data/record_list.csv (study_id, path) |
| Meta split | split/data/meta_split.csv (save_file, split; written by create_split.py) |

---

## 3. Preprocess

| What | Where |
|------|--------|
| Raw WFDB root | External; [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/); passed as --raw-root |
| Preprocessed .mat | preprocess/data/lead_1_duplicated/ (written by preprocess_ecgfm.py) |

---

## 4. Build

| What | Where |
|------|--------|
| Test / finetune sets | build/data/test_lead1_duplicated/, build/data/finetune_lead1_duplicated/ (written by build_test_and_finetune_data.py) |

---

## 5. Fine-tuning (fairseq)

| What | Where |
|------|--------|
| Pretrained .pt | External; [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm); passed as model.model_path |
| Fine-tuned .pt | finetuned_model/ (optional; --model-path for eval) |

---

## 6. Eval

| What | Where |
|------|--------|
| Predictions / metrics | eval/data/ (written by eval_ecgfm.py or eval_finetuned.py) |
