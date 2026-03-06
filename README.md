# ECG-FM Generic Lead Fine-Tuning

This repository contains a pipeline for extracting **Lead I or Lead II** from MIMIC-IV-ECG and **fine-tuning ECG-FM** with [fairseq_signals](https://github.com/Jwoo5/fairseq-signals) (Bowen Lab [ECG-FM](https://github.com/bowang-lab/ecg-fm)). Everything you need to reproduce the fine-tuning run is here: scripts, configs, and step-by-step instructions below.

---

## How to reproduce (step-by-step)

Do these steps **in order**. Each step needs the outputs of the previous one. Run all commands from the **repository root** (or set `ECG_FINETUNE_BASE` to the repo root).

### Before you start (prerequisites)

| You need | Where to get it | Where it goes / how it’s used |
|----------|------------------|-------------------------------|
| **machine_measurements.csv** | MIMIC-IV-ECG (study_id, report_0…report_17) | Put in `labels/label_inputs/` |
| **record_list.csv** | Same cohort; columns: `study_id`, `path` (path = path under MIMIC-IV-ECG root) | Repo already has `split/data/record_list.csv` or use `--record-list` |
| **Meta split** | Either generate (Step 1) or use provided `split/data/meta_split_mimic_iv_ecg.csv` | Used in Steps 2 and 3 via `--meta` or as `split/data/meta_split.csv` |
| **Raw MIMIC-IV-ECG** | [PhysioNet MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) | Pass as `--raw-root` in Step 2 |
| **Pretrained ECG-FM .pt** | [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) | Pass as `model.model_path` in Step 4 |
| **fairseq_signals** | `git clone https://github.com/Jwoo5/fairseq-signals` + `pip install -e .` | Required for Step 4 (fine-tune) and Step 5 (eval) |

Also: `pip install -r requirements.txt` in this repo.

---

### Step 0 — Create labels

Uses the ECG-FM pattern labeler to turn text reports into 17 diagnostic labels.

```bash
python labels/scripts/create_labels.py --base-dir .
```

**Needs:** `labels/label_inputs/machine_measurements.csv`, and the labeler (config + code under `labels/ecg_fm_labeler_config/`, `labels/ecg_fm_labeler/`) — already in the repo.

**Produces:** `labels/computed_labels/` (labels.csv, y.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv).

**Check:** `ls labels/computed_labels/labels.csv labels/computed_labels/y.npy` should exist.

---

### Step 1 — Split (train / valid / test)

Either **A** or **B**:

- **A) Use the same split as HiPerGator**  
  Copy the provided file so it’s used as the default meta split:
  ```bash
  cp split/data/meta_split_mimic_iv_ecg.csv split/data/meta_split.csv
  ```
- **B) Generate a new random split** (80/10/10, seed 42):
  ```bash
  python split/scripts/create_split.py --base-dir .
  ```

**Needs:** `split/data/record_list.csv` (columns: study_id, path).

**Produces:** `split/data/meta_split.csv` (columns: save_file, split).

**Check:** `head -2 split/data/meta_split.csv` shows save_file and split.

---

### Step 2 — Preprocess (raw ECG → 10 s .mat per lead)

Extracts one lead (I or II), duplicates to 12 channels, 500 Hz, 10 s. Run **once per lead** you want (e.g. Lead 1 and Lead 2).

```bash
# Lead 1 (Lead I)
python preprocess/scripts/preprocess.py --base-dir . --lead 1 --raw-root /path/to/mimic-iv-ecg

# Lead 2 (Lead II)
python preprocess/scripts/preprocess.py --base-dir . --lead 2 --raw-root /path/to/mimic-iv-ecg
```

**Needs:** `split/data/meta_split.csv`, `split/data/record_list.csv`, and the MIMIC-IV-ECG root (unzipped) at `--raw-root`. Paths in record_list are relative to that root.

**Produces:** `preprocess/data/lead_1/mats_10s/*.mat` and/or `preprocess/data/lead_2/mats_10s/*.mat`.

**Check:** `ls preprocess/data/lead_1/mats_10s/ | head -5` shows .mat files.

---

### Step 3 — Build manifests (fairseq train/valid/test)

Builds 5 s segments and fairseq manifest TSVs + label files. Run **once per lead** you preprocessed.

```bash
# Lead 1
python manifest/scripts/build_test_and_finetune_data.py --base-dir . --lead 1

# Lead 2 (if you ran preprocess for lead 2)
python manifest/scripts/build_test_and_finetune_data.py --base-dir . --lead 2
```

If you use the provided meta split from a different path:
```bash
python manifest/scripts/build_test_and_finetune_data.py --base-dir . --lead 1 --meta split/data/meta_split_mimic_iv_ecg.csv
```

**Needs:** `labels/computed_labels/`, `split/data/meta_split.csv` (or `--meta`), and `preprocess/data/lead_{lead}/mats_10s/`.

**Produces:** `manifest/data/lead_1/manifests/` (and/or lead_2): train.tsv, valid.tsv, test.tsv, y.npy, label_def.csv, pos_weight.txt, plus segmented_5s/, test_labels.csv, etc.

**Check:** `test -f manifest/data/lead_1/manifests/train.tsv && echo OK`.

---

### Step 4 — Fine-tune (in fairseq_signals)

Run **inside your fairseq-signals clone**, once per lead. Set paths to this repo and your pretrained .pt. The same command works for Lead 1 or Lead 2; only `MANIFEST_DIR` and `OUTPUT_DIR` change (e.g. `lead_1` vs `lead_2`).

See the full **[Fine-tuning (fairseq_signals)](#fine-tuning-fairseq_signals)** block below for the exact `fairseq-hydra-train` command (matches the HiPerGator run: best by validation loss, lr=1e-6, batch 256).

**Needs:** `manifest/data/lead_{lead}/manifests/` from Step 3, and the pretrained model .pt.

**Produces:** Checkpoints in `OUTPUT_DIR` (e.g. `checkpoint_best.pt`).

**Check:** `ls $OUTPUT_DIR/checkpoint_best*.pt` after training.

---

### Step 5 — Evaluate on test set

Run inference and compute metrics (per-label + macro + micro). One script for either lead or both.

```bash
# One lead
python eval/scripts/evaluate.py --base-dir . --lead 1 --model-path /path/to/checkpoint_best.pt

# Both leads (default checkpoint paths: finetuned_model/checkpoint_best_lead_1.pt, .../checkpoint_best_lead_2.pt)
python eval/scripts/evaluate.py --base-dir . --both
```

**Needs:** `manifest/data/lead_{lead}/` (from Step 3), the checkpoint .pt, and fairseq_signals (for the model loader).

**Produces:** `eval/data/` — CSV with metrics (AUROC, AUPRC, F1, etc.) and optional prediction CSVs.

---

## Repository structure

```
git_ecg_finetuned/
├── README.md
├── requirements.txt
├── labels/
│   ├── label_inputs/          # Inputs for the label script (e.g. machine_measurements.csv)
│   ├── ecg_fm_labeler_config/ # ECG-FM labeler JSON config + label_def.csv
│   ├── ecg_fm_labeler/        # ECG-FM pattern labeler Python package
│   ├── scripts/
│   │   └── create_labels.py
│   └── computed_labels/       # Label script output (labels.csv, y.npy, etc.)
├── split/
│   ├── data/                  # record_list.csv, meta_split.csv
│   └── scripts/
│       └── create_split.py
├── preprocess/
│   ├── data/                  # lead_1/, lead_2/ (.mat output)
│   └── scripts/
│       └── preprocess.py      # Accepts --lead 1 or --lead 2
├── manifest/
│   ├── data/                  # lead_1/, lead_2/ (manifests in lead_1/manifests/)
│   └── scripts/
│       ├── README.md          
│       └── build_test_and_finetune_data.py # Accepts --lead 1 or --lead 2
└── eval/
    ├── data/                  # Predictions and metrics output
    └── scripts/
        └── evaluate.py        # Evaluation (--lead 1 or 2, or --both)
finetuned_model/               # Fine-tuned .pt checkpoints
```

- **labels/** — Label inputs (label_inputs), ECG-FM labeler config and package, label-creation script, and computed labels output.
- **split/** — Record list and split script; output meta_split.csv for preprocess and manifest. The same split used on HiperGator is provided as **meta_split_mimic_iv_ecg.csv** (use `--meta split/data/meta_split_mimic_iv_ecg.csv` or copy to meta_split.csv).
- **preprocess/** — Script that produces 10 s .mat (Lead I or II → 12 ch) from WFDB; output in preprocess/data/lead_{lead}/.
- **manifest/** — Script that builds test and finetune sets and creates fairseq manifests (train/valid/test TSVs, y.npy, etc.) in manifest/data/lead_{lead}/.
- **eval/** — Scripts to run ECG-FM inference and write predictions and metrics to eval/data/.
- **finetuned_model/** — Directory for fine-tuned checkpoints; eval scripts take a path via `--model-path`.

---

## Installation

- **fairseq_signals** (fine-tuning and evaluation):  
  `git clone https://github.com/Jwoo5/fairseq-signals` then `pip install --editable ./`
- **This repo:** `pip install -r requirements.txt`. Eval also requires `ecg-transform` and `fairseq_signals`.

---

## Data and model

- **Raw ECG:** [PhysioNet MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/). Pass its root as `--raw-root` to the preprocess script.
- **Pretrained model:** [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) — **mimic_iv_ecg_physionet_pretrained.pt** for Lead I fine-tuning.

---

## Pipeline

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 0 Labels | labels/scripts/create_labels.py | labels/label_inputs/, ecg_fm_labeler_config/, ecg_fm_labeler/, optional split/data/meta_split.csv | labels/computed_labels/ |
| 1 Split | split/scripts/create_split.py | split/data/record_list.csv | split/data/meta_split.csv |
| 2 Preprocess | preprocess/scripts/preprocess.py | split/data/, raw WFDB (--raw-root), --lead 1 or 2 | preprocess/data/lead_{lead}/ |
| 3 Manifest | manifest/scripts/build_test_and_finetune_data.py | labels/computed_labels/, split/data/, preprocess/data/, --lead 1 or 2 | manifest/data/lead_{lead}/ |
| 4 Fine-tune | fairseq-hydra-train (in fairseq-signals) | manifest/data/lead_{lead}/manifests/, pretrained .pt | Checkpoint directory |
| 5 Eval | eval/scripts/evaluate.py | manifest/data/lead_{lead}/, labels/computed_labels/, --model-path, --lead 1 or 2 | eval/data/ |

Run scripts from the repository root with `--base-dir .` (or set `ECG_FINETUNE_BASE`).

---

## Fine-tuning (fairseq_signals)

From a **fairseq-signals** clone. Set paths for the lead you want (e.g. `lead_1` or `lead_2`); the same command works for either by changing `MANIFEST_DIR` and `OUTPUT_DIR`. Args below match the HiPerGator run: best checkpoint by **validation loss**, lr=1e-6, batch 256, valid only.

```bash
FAIRSEQ_SIGNALS_ROOT="/path/to/fairseq-signals"
PRETRAINED_MODEL="/path/to/mimic_iv_ecg_physionet_pretrained.pt"
# For Lead 1:
MANIFEST_DIR="/path/to/git_ecg_finetuned/manifest/data/lead_1/manifests"
OUTPUT_DIR="/path/to/checkpoints_finetune_lead1"
# For Lead 2: use lead_2 in both paths.

LABEL_DIR="$MANIFEST_DIR"
NUM_LABELS=$(($(wc -l < "$LABEL_DIR/label_def.csv") - 1))
[ -f "$LABEL_DIR/pos_weight.txt" ] && POS_WEIGHT="[$(cat "$LABEL_DIR/pos_weight.txt" | tr ' ' ',')]" || POS_WEIGHT="null"

cd "$FAIRSEQ_SIGNALS_ROOT"
fairseq-hydra-train \
  task.data="$MANIFEST_DIR" \
  +task.label_file="${LABEL_DIR}/y.npy" \
  task.normalize=false task.enable_padding=true task.enable_padding_leads=false \
  model.model_path="$PRETRAINED_MODEL" model.num_labels=$NUM_LABELS model.no_pretrained_weights=false \
  model.dropout=0.0 model.attention_dropout=0.0 model.activation_dropout=0.1 \
  model.feature_grad_mult=0 model.freeze_finetune_updates=0 model.in_d=12 model.saliency=false \
  +criterion.pos_weight="$POS_WEIGHT" criterion.report_auc=true criterion.report_cinc_score=false \
  optimization.max_epoch=100 optimization.max_update=320000 optimization.lr=[1e-06] \
  lr_scheduler._name=fixed lr_scheduler.warmup_updates=0 \
  optimizer._name=adam optimizer.adam_betas='[0.9,0.98]' optimizer.adam_eps=1e-08 \
  dataset.num_workers=6 dataset.batch_size=256 dataset.max_tokens=null \
  'dataset.valid_subset="valid"' dataset.disable_validation=false \
  distributed_training.distributed_world_size=1 distributed_training.find_unused_parameters=true \
  checkpoint.save_dir="$OUTPUT_DIR" checkpoint.save_interval=1 checkpoint.save_interval_updates=0 \
  checkpoint.no_epoch_checkpoints=true checkpoint.no_last_checkpoints=true checkpoint.keep_last_epochs=0 \
  checkpoint.keep_best_checkpoints=1 checkpoint.no_save_optimizer_state=true \
  checkpoint.best_checkpoint_metric=loss checkpoint.maximize_best_checkpoint_metric=false \
  common.seed=1 common.fp16=false common.log_format=json common.log_interval=10 common.all_gather_list_size=2048000 \
  --config-dir "${FAIRSEQ_SIGNALS_ROOT}/examples/w2v_cmsc/config/finetuning/ecg_transformer" \
  --config-name diagnosis
```

To save the best checkpoint by **validation AUROC** instead, set `checkpoint.best_checkpoint_metric=auroc` and `checkpoint.maximize_best_checkpoint_metric=true`.

---

## Evaluation

One script **eval/scripts/evaluate.py** works for Lead 1, Lead 2, or both: same code path, only `--lead` and paths change.

```bash
# Single lead: pass --lead 1 or --lead 2 and the checkpoint for that lead
python eval/scripts/evaluate.py --base-dir . --lead 1 --model-path /path/to/checkpoint_best_lead1.pt
python eval/scripts/evaluate.py --base-dir . --lead 2 --model-path /path/to/checkpoint_best_lead2.pt

# Both leads: uses default paths finetuned_model/checkpoint_best_lead_1.pt and .../checkpoint_best_lead_2.pt
# or set --model-path-lead1 / --model-path-lead2
python eval/scripts/evaluate.py --base-dir . --both
```

Results are written to **eval/data/** (per-lead metrics and prediction CSVs).

---

## References

- [Bowen Lab ECG-FM](https://github.com/bowang-lab/ecg-fm), [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm), [Paper arXiv:2408.05178](https://arxiv.org/abs/2408.05178)
- [fairseq_signals](https://github.com/Jwoo5/fairseq-signals)
- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
